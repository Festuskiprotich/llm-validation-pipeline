import time
import httpx
from pydantic import BaseModel

# 1. Define what a response from Ollama looks like
class OllamaResponse(BaseModel):
    model: str
    response: str
    done: bool

# 2. Create a function to talk to Ollama
def ask_ollama(prompt: str, model: str = "qwen2.5:3b") -> dict:
    """
    Sends a prompt to the local Ollama server and returns the response + latency.
    """
    # The URL of the Ollama API (we learned about this port earlier!)
    url = "http://localhost:11434/api/generate"
    
    # The data we send to the server
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False # We want the whole answer at once, not piece by piece
    }
    
    # Start the timer
    start_time = time.time()
    
    # Send the HTTP POST request to Ollama
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status() # Throws an error if the server failed
        
    # Stop the timer
    latency = time.time() - start_time
    
    # Parse the JSON response using our Pydantic model
    data = OllamaResponse(**response.json())
    
    return {
        "answer": data.response,
        "model": data.model,
        "latency_seconds": round(latency, 2)
    }

# 3. This block only runs if you execute this file directly
if __name__ == "__main__":
    print("🤖 Testing Ollama connection...")
    
    # Call our function with a test prompt
    result = ask_ollama("Explain what an LLM is in one short sentence.")
    
    print("\n--- RESPONSE ---")
    print(result["answer"].strip())
    print("\n--- METRICS ---")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_seconds']} seconds")