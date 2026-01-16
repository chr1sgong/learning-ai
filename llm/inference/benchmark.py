import time
from vllm import LLM, SamplingParams
import os

# OPTIONAL: Point to a specific cache folder if you don't want to clutter home
# os.environ['HF_HOME'] = './model_cache'

def run_benchmark():
    # 1. Initialize Engine
    # gpu_memory_utilization=0.95 -> Uses ~22.8GB of your 24GB VRAM
    # THis maximizes the KV cache size (context length).
    print("--- 🚀 Loading Model on RTX 4090 ---")
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        gpu_memory_utilization=0.90,
        dtype="bfloat16"
    )

    # 2. Parameters
    # Maximize tokens to stress test generation speed
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512
    )

    # 3. Payload
    prompts = [
        "Write a highly concurrent Go implementation of a worker pool.",
        "Explain the CAP theorem in the context of distributed databases.",
        "What are the differences between LevelDB and RocksDB?",
        "Generate a Python script to monitor GPU memory usage using pynvml."
    ]

    # 4. Inference (Batch)
    print("---🔥 Starting Inference Batch ---")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # 5. Metrics
    total_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
    duration = end_time - start_time
    tps = total_tokens / duration

    print(f"\n--- 📊 Results ---")
    print(f"Total Tokens: {total_tokens}")
    print(f"Duration:     {duration:.4f}s")
    print(f"Throughput:   {tps:.2f} tokens/s")

if __name__ == "__main__":
    run_benchmark()