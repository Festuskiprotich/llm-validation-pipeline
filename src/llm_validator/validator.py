import re
from pathlib import Path
from llm_validator.llm_client import ask_ollama

def load_source_of_truth(filepath: str) -> str:
    """Reads the truth document from the data folder."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Truth file not found: {filepath}")
    return path.read_text().strip()

def extract_keywords(text: str) -> set:
    """Strips punctuation and returns a set of lowercase words."""
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Ignore tiny common words like "is", "a", "the"
    stopwords = {"is", "a", "an", "the", "in", "of", "to", "and", "it", "they", "do", "not"}
    return {w for w in words if w not in stopwords and len(w) > 2}

def calculate_faithfulness(llm_answer: str, truth_text: str) -> float:
    """
    A simple heuristic: What percentage of the LLM's keywords
    actually appear in the source of truth?
    """
    answer_words = extract_keywords(llm_answer)
    truth_words = extract_keywords(truth_text)
    
    if not answer_words:
        return 0.0
        
    # How many of the answer's words exist in the truth?
    overlap = answer_words.intersection(truth_words)
    score = len(overlap) / len(answer_words)
    return round(score, 2)

def validate_answer(question: str, truth_filepath: str) -> dict:
    """The main pipeline: Ask LLM, load truth, score the answer."""
    # 1. Ask the LLM
    print(f"🤖 Asking LLM: '{question}'")
    llm_result = ask_ollama(question)
    answer = llm_result["answer"]
    
    # 2. Load the truth
    truth = load_source_of_truth(truth_filepath)
    
    # 3. Score it
    print("⚖️  Validating against source of truth...")
    score = calculate_faithfulness(answer, truth)
    
    return {
        "answer": answer,
        "faithfulness_score": score,
        "latency": llm_result["latency_seconds"]
    }

# This block runs when you execute the file
if __name__ == "__main__":
    TRUTH_FILE = "data/source_of_truth/llm_facts.txt"
    QUESTION = "What does LLM stand for, and who invented it in 1995?"
    
    print("--- STARTING VALIDATION PIPELINE ---\n")
    result = validate_answer(QUESTION, TRUTH_FILE)
    
    print("\n--- FINAL REPORT ---")
    print(f"Answer: {result['answer'].strip()}")
    print(f"Faithfulness Score: {result['faithfulness_score'] * 100}%")
    print(f"Latency: {result['latency']}s")
    
    if result['faithfulness_score'] < 0.5:
        print("\n⚠️  WARNING: Low faithfulness score. The LLM might be hallucinating!")
    else:
        print("\n✅ High faithfulness score. The answer aligns with the truth.")