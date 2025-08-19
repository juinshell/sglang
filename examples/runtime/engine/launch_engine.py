"""
This example demonstrates how to launch the offline engine.
"""

import sglang as sgl


def main():
    llm = sgl.Engine(model_path="/tmp/models/Llama-3.2-8B-Instruct")
    print(llm.generate("how to train an LLM?"))
    llm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
