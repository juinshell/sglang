# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
import signal
import threading
from enum import Enum, auto

from networkx import min_cost_flow

import psutil
import setproctitle
import zmq

from typing import Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    ScaleUpDpGroupReq,
    CacheAwareInfoInput,
    CacheAwareInfoOutput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import bind_port, configure_logger, get_zmq_socket
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

log_path = "/workspace/moe-scaling/log"
import os
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    CACHE_AWARE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        # Parse args
        self.max_total_num_tokens = None
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.CACHE_AWARE: self.cache_aware_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers = [None] * server_args.dp_size
        # [cache-aware scheduler]
        self.cache_aware_receivers = [None] * server_args.dp_size

        # [moe-scaling]
        self.cur_gpus = 0
        self.cur_dp_size = 0
        self.workers_per_dp = server_args.tp_size // server_args.dp_size
        self.dp_groups_flag = [
            False
        ] * server_args.dp_size  # at beginning, only first dp group is serving

        if server_args.enable_dp_attention:
            dp_port_args = self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
            self.cur_gpus = self.workers_per_dp
            self.cur_dp_size = 1
            self.dp_groups_flag[0] = True
        else:
            dp_port_args = self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        if not self.server_args.enable_scaling:
            self.dp_groups_flag = [True] * server_args.dp_size

        # Only node rank 0 runs the real data parallel controller that dispatches the requests.
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    dp_port_args[dp_rank].scheduler_input_ipc_name,
                    True,
                )
                self.cache_aware_receivers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PULL,
                    dp_port_args[dp_rank].scheduler_output_ipc_name,
                    True,
                )
        self.max_req_input_len = None

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            dp_port_args.append(tmp_port_args)

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += server_args.tp_size * server_args.gpu_id_step

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

        return dp_port_args

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            pass

    def scaling_up_intra_node_dp_group(self, req: ScaleUpDpGroupReq):
        # assume initally, we can see all gpus, but only one dp group can serve
        self.dp_groups_flag[self.cur_dp_size] = True
        # done scaling up
        self.cur_dp_size += 1
        self.cur_gpus += self.workers_per_dp
        logger.info(f"Scaling up DP group to {self.cur_dp_size} numbers.")

    def launch_dp_attention_schedulers(self, server_args, port_args):
        self.launch_tensor_parallel_group(server_args, port_args, 0, None)
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            dp_port_args.append(PortArgs.init_new(server_args, dp_rank))
        return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        scaling: Optional[bool] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        # Launch tensor parallel scheduler processes
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        # if scaling is True:
        #     # tp_size is the minimum scaling unit
        #     tp_size_per_node = min(
        #         (server_args.tp_size + self.cur_gpus), server_args.gpus_per_node
        #     )
        #     tp_rank_range = range(self.cur_gpus, self.cur_gpus + server_args.tp_size)

        for tp_rank in tp_rank_range:
            rank_port_args = port_args

            if server_args.enable_dp_attention:
                # dp attention has different sharding logic
                attn_tp_rank, _, dp_rank = compute_dp_attention_world_info(
                    server_args.enable_dp_attention,
                    tp_rank,
                    server_args.tp_size,
                    server_args.dp_size,
                )
                # compute zmq ports for this dp rank
                rank_port_args = PortArgs.init_new(server_args, dp_rank)
                # Data parallelism resues the tensor parallelism group,
                # so all dp ranks should use the same nccl port.
                rank_port_args.nccl_port = port_args.nccl_port

            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + base_gpu_id
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )

            logger.info(
                f"Launching scheduler for DP{dp_rank}-TP{attn_tp_rank} starting at GPU #{gpu_id}."
            )
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, rank_port_args, gpu_id, tp_rank, dp_rank, writer),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            self.scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def round_robin_scheduler(self, req: Req):
        if (
            self.server_args.disaggregation_mode == "null"
            and not self.server_args.enable_scaling
        ):
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
        elif self.server_args.enable_scaling:
            self.workers[self.round_robin_counter].send_pyobj(req)
            self.round_robin_counter = (self.round_robin_counter + 1) % self.cur_dp_size
        else:
            self.workers[req.bootstrap_room % len(self.workers)].send_pyobj(req)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def cache_aware_scheduler(
        self, req: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput
    ):
        # logger.info(f"Cache-aware scheduler: dp_groups_flags:{self.dp_groups_flag}")
        # choose the worker with LPM prefix match
        max_prefix_len = 0
        max_prefix_idx = 0
        max_workload = 0
        # max_workload_idx = 0
        min_workload = 2**31 - 1
        min_workload_idx = 0
        max_available_size = 0
        max_available_size_idx = 0

        for i in range(len(self.workers)):
            if self.dp_groups_flag[i]:
                self.workers[i].send_pyobj(CacheAwareInfoInput(token_ids=req.input_ids))

        # receive the prefix length from each scheduler
        received_cnt = 0
        received_len = [-1] * len(self.workers)
        received_workload = [-1] * len(self.workers)
        received_available_size = [-1] * len(self.workers)
        while received_cnt < sum(self.dp_groups_flag):
            for i in range(len(self.workers)):
                if received_len[i] == -1 and self.dp_groups_flag[i]:
                    try:
                        recv_info_obj: CacheAwareInfoOutput = (
                            self.cache_aware_receivers[i].recv_pyobj(zmq.NOBLOCK)
                        )
                        prefix_len = recv_info_obj.prefix_len
                        received_len[i] = prefix_len
                        workload = recv_info_obj.running_req_num
                        received_workload[i] = workload
                        available_size = recv_info_obj.available_size
                        received_available_size[i] = available_size

                        received_cnt += 1
                        if prefix_len > max_prefix_len:
                            max_prefix_len = prefix_len
                            max_prefix_idx = i
                        if workload > max_workload:
                            max_workload = workload
                            # max_workload_idx = i
                        if workload < min_workload:
                            min_workload = workload
                            min_workload_idx = i
                        if available_size > max_available_size:
                            max_available_size = available_size
                            max_available_size_idx = i

                    except zmq.ZMQError:
                        continue

        ratio = max_prefix_len / len(req.input_ids)
        logger.info(
            f"Cache-aware scheduler: received_len: {received_len}, max_prefix_idx: {max_prefix_idx}, received_workload: {received_workload}, min_workload_idx: {min_workload_idx}, received_available_size: {received_available_size}, ratio: {ratio:.4f}"
        )

        # load balance logic
        if max_workload - min_workload >= 32 or max_workload > 1.0001 * min_workload:
            # unbalance, send to the least loaded worker
            self.workers[min_workload_idx].send_pyobj(req)
            return

        # balance

        if ratio >= 0.3:
            self.workers[max_prefix_idx].send_pyobj(req)
        else:
            self.workers[max_available_size_idx].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)

                elif isinstance(recv_req, ScaleUpDpGroupReq):
                    # scaling up dp group
                    self.scaling_up_intra_node_dp_group(recv_req)
                else:
                    # Send other control messages to first worker of tp group
                    for worker in self.workers[:: self.control_message_step]:
                        worker.send_pyobj(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
