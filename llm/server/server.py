import json  # JSON 序列化，用于构造 SSE 消息
import uuid  # 生成唯一请求 ID
from contextlib import asynccontextmanager  # lifespan 装饰器，管理应用生命周期

import uvicorn  # ASGI 服务器，运行 FastAPI 应用
from fastapi import FastAPI, Request  # Web 框架核心组件
from fastapi.responses import StreamingResponse  # SSE 流式响应
from vllm import SamplingParams  # vLLM 采样参数
from vllm.engine.arg_utils import AsyncEngineArgs  # 异步引擎参数配置
from vllm.engine.async_llm_engine import AsyncLLMEngine  # 异步推理引擎

# Global Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model ONCE on startup.
    This prevents reloading 15GB of weights on every request.
    """
    global engine
    # Configure the engine arguments (not LLM instance!)
    engine_args = AsyncEngineArgs(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        max_model_len=4096
    )
    print("--- 🚀 Initializing Async Engine ---")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield  # This yield is required for @asynccontextmanager
    # Cleanup code would go here (after yield)
    print("--- 🛑 Shutting down ---")

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_stream(request: Request):
    """
    Accepts JSON: {"prompts": "...", "temperature": 0.7}
    Returns: Server-Sent Events (SSE) stream
    """
    data = await request.json()
    prompts = data.get("prompts")

    # Validation (Senior SDE habit: fail fast)
    if not prompts:
        return {"error": "Prompt is required"}
    
    # Define Sampling Parameters
    # We default to streaming one token at a time
    sampling_params = SamplingParams(
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 512)
    )

    # Every request needs a unique ID for the engine's internal tracker
    request_id = str(uuid.uuid4())

    # The Generator Function
    async def stream_results():
        # Add requests to the engine's queue
        # This returns an AsyncInterator
        results_generator = engine.generate(prompts, sampling_params, request_id)

        # Iterate as tokens are generated
        async for request_output in results_generator:
            # vLLM returns the FULL text every time by default.
            # We only want the *new* text (delta) for streaming.
            # (In production, we would calculate diffs, but vLLM objects
            # often simplify this by giving access to latest token).

            # Simple approach: Return the full JSON object for the frontend to parse
            text_output = request_output.outputs[0].text
            yield f"data: {json.dumps({'text': text_output})}\n\n"

    async def delta_stream_results():
        results_generator = engine.generate(prompts, sampling_params, request_id)
        previous_text_len = 0
        async for request_output in results_generator:
            # vLLM always gives us the FULL text generated so far
            curr_text = request_output.outputs[0].text
            # Slice to get only the new part
            delta = curr_text[previous_text_len:]
            # Update our tracking of the full text
            previous_text_len = len(curr_text)
            
            # Send only the delta part
            if delta:
                payload = json.dumps({'delta': delta})
                yield f"data: {payload}\n\n"
    # Return the generator as a StreamingResponse
    return StreamingResponse(delta_stream_results(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)