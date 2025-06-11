import os
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Initialize vLLM with CPU
try:
    print("Loading vLLM model...")
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        device="cpu",
        max_model_len=512,
        disable_custom_all_reduce=True,
        dtype="float32"
    )
    print("vLLM loaded successfully on CPU!")
except Exception as e:
    print(f"Error loading vLLM: {e}")
    print(f"Full traceback: {traceback.format_exc()}")
    llm = None

class CompletionRequest(BaseModel):
    model: str = "facebook/opt-125m"
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    if llm is None:
        return {"error": "Model not loaded"}
    
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return {
            "choices": [{"text": generated_text}]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)