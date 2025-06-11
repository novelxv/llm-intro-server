# LLM Intro Server

This is a simple FastAPI server that integrates with two different local LLM backends: **vLLM** and **Ollama**. It allows you to send prompts to each engine and receive generated responses via RESTful API endpoints.

---

## 🚀 Features

- 🔌 **FastAPI-based** backend server.
- 🤖 Integration with:
  - `vLLM` (tested with `facebook/opt-125m`)
  - `Ollama` (tested with `llama2`)
- 🧪 Simple JSON-based POST endpoints for prompt completion.

---

## 📦 Requirements

Make sure you have Python 3.8+ installed.

### Install dependencies

```bash
pip install -r requirements.txt
```

If not using a `requirements.txt`:

```bash
pip install fastapi uvicorn requests pydantic vllm torch torchvision
```

### Install Ollama

1. **Download Ollama:**
   - Visit [https://ollama.ai](https://ollama.ai)
   - Download for your platform (Linux, macOS, or Windows)

2. **Pull a model:**
   ```bash
   ollama pull llama2
   ```
   
   Other lightweight options:
   ```bash
   ollama pull phi3.5
   ollama pull gemma2:2b
   ```

3. **Start Ollama service:**
   ```bash
   ollama serve
   ```
   This runs on `http://localhost:11434`

---

## ⚙️ Usage

### 1. Start Ollama Server (if using Ollama endpoint)

```bash
ollama serve
```

### 2. Run vLLM Engine (if using vLLM endpoint)

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --port 8001
```
Make sure your device has GPU support if using vLLM.

### 3. Run FastAPI Server

```bash
python main.py
```

By default, the server runs on `http://localhost:8000`.

### 3. Test Endpoints

#### `/vllm`

```bash
POST /vllm
Content-Type: application/json

{
  "prompt": "Hello, how are you?",
  "max_tokens": 50
}
```

#### `/ollama`

```bash
POST /ollama
Content-Type: application/json

{
  "prompt": "Hello, how are you?",
  "max_tokens": 50
}
```

---

## 📝 Notes

- Ensure the vLLM/Ollama servers are running before calling the endpoints.

- vLLM runs on port `8001` by default, while Ollama runs on port `11434`.

---

## 🔍 API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.