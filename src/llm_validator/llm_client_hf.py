import os
import time
from groq import Groq

def ask_hf(prompt: str, model: str = "openai/gpt-oss-20b") -> dict:
    """Uses Groq API instead of Hugging Face (much faster and more reliable)."""
    token = os.getenv("GROQ_API_KEY")
    if not token:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
        
    client = Groq(api_key=token)
    start_time = time.time()
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
        max_tokens=500,
    )
    
    latency = time.time() - start_time
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "model": model,
        "latency_seconds": round(latency, 2)
    }
