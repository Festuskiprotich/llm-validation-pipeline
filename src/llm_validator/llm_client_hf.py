import os
import time
from huggingface_hub import InferenceClient

def ask_hf(prompt: str, model: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Uses Hugging Face Inference API instead of local Ollama."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is missing.")
        
    client = InferenceClient(model=model, token=token)
    start_time = time.time()
    
    # HF Inference API call
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
    )
    
    latency = time.time() - start_time
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "model": model,
        "latency_seconds": round(latency, 2)
    }
