from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Intro Server")

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class GenResponse(BaseModel):
    engine: str
    text: str

# vLLM
@app.post("/vllm", response_model=GenResponse)
def generate_vllm(req: GenRequest):
    try:
        resp = requests.post(
            "http://localhost:8001/v1/completions",
            json={
                "model": "facebook/opt-125m",
                "prompt": req.prompt, 
                "max_tokens": req.max_tokens,
                "temperature": 0.7
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            text = data["choices"][0].get("text", "")
        else:
            text = data.get("error", "No response")
            
        return GenResponse(engine="vLLM", text=text)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="vLLM server not available")
    except Exception as e:
        logger.error(f"vLLM error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ollama
@app.post("/ollama", response_model=GenResponse)
def generate_ollama(req: GenRequest):
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2", 
                "prompt": req.prompt,
                "options": {"num_predict": req.max_tokens},
                "stream": False
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return GenResponse(engine="Ollama", text=data.get("response", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
