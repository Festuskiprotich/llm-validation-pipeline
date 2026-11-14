from llm_validator.observability import get_stats
import re
import json
from pathlib import Path

import os
if os.getenv("USE_HF_API") == "true":
    from llm_validator.llm_client_hf import ask_hf as ask_ollama
else:
    from llm_validator.llm_client import ask_ollama

from llm_validator.observability import log_validation
# =========================================================
# 1. BASELINE: Keyword-based faithfulness (our first attempt)
# =========================================================

def load_source_of_truth(filepath: str) -> str:
    """Reads the truth document from the data folder."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Truth file not found: {filepath}")
    return path.read_text().strip()

def extract_keywords(text: str) -> set:
    """Strips punctuation and returns a set of lowercase words."""
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"is", "a", "an", "the", "in", "of", "to", "and", "it", "they", "do", "not"}
    return {w for w in words if w not in stopwords and len(w) > 2}

def calculate_faithfulness(llm_answer: str, truth_text: str) -> float:
    """Baseline: What percentage of the LLM's keywords appear in the truth?"""
    answer_words = extract_keywords(llm_answer)
    truth_words = extract_keywords(truth_text)
    if not answer_words:
        return 0.0
    overlap = answer_words.intersection(truth_words)
    return round(len(overlap) / len(answer_words), 2)

# =========================================================
# 2. NEW: LLM-as-a-Judge (production-grade)
# =========================================================

JUDGE_PROMPT_TEMPLATE = """
You are a fair fact-checker. Your job is to score whether an AI's answer CONTRADICTS a provided source of truth.

SOURCE OF TRUTH:
"{source}"

AI'S ANSWER:
"{answer}"

Instructions:
1. Compare the AI's answer to the source of truth.
2. If the AI's answer CONTRADICTS anything in the source, that is a hallucination. Score it LOW (0.0 - 0.3).
3. If the AI's answer is CONSISTENT with the source, but adds additional correct context, that is NOT a hallucination. Score it HIGH (0.7 - 1.0).
4. If the AI's answer partially agrees but includes some false claims, score it MID (0.4 - 0.6).

Respond with ONLY a JSON object, no other text:
{{"score": <number>, "reasoning": "<one sentence>"}}
"""

def judge_with_llm(llm_answer: str, truth_text: str) -> dict:
    """Uses a second LLM call to judge the first LLM's answer."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(source=truth_text, answer=llm_answer)
    judge_result = ask_ollama(prompt)
    raw = judge_result["answer"].strip()

    # Try to parse the JSON the judge was instructed to return
    try:
        # Find the JSON object inside the response (in case the model adds text)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found")
        parsed = json.loads(json_match.group())
        return {
            "score": float(parsed["score"]),
            "reasoning": parsed.get("reasoning", ""),
            "latency": judge_result["latency_seconds"],
        }
    except Exception as e:
        # Fallback: if the judge misbehaved, return a low-confidence result
        return {
            "score": 0.0,
            "reasoning": f"Judge failed to parse: {e}. Raw output: {raw[:100]}",
            "latency": judge_result["latency_seconds"],
        }

# =========================================================
# 3. MAIN PIPELINE
# =========================================================

def validate_answer(question: str, truth_filepath: str) -> dict:
    """Ask LLM, load truth, score with BOTH methods."""
    print(f"🤖 Asking LLM: '{question}'")
    llm_result = ask_ollama(question)
    answer = llm_result["answer"]

    truth = load_source_of_truth(truth_filepath)

    print("🔍 Scoring with keyword matcher...")
    keyword_score = calculate_faithfulness(answer, truth)

    print("👨‍⚖️  Scoring with LLM-as-a-Judge (this takes longer)...")
    judge_result = judge_with_llm(answer, truth)


    log_validation(
        question=question,
        answer=answer,
        keyword_score=keyword_score,
        judge_score=judge_result["score"],
        judge_reasoning=judge_result["reasoning"],
        answer_latency=llm_result["latency_seconds"],
        judge_latency=judge_result["latency"],
    )

    return {
        "answer": answer,
        "keyword_score": keyword_score,
        "judge_score": judge_result["score"],
        "judge_reasoning": judge_result["reasoning"],
        "latency_answer": llm_result["latency_seconds"],
        "latency_judge": judge_result["latency"],
    }

# =========================================================
# 4. ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    TRUTH_FILE = "data/source_of_truth/llm_facts.txt"
    QUESTION = "What does LLM stand for and how does it work?"

    print("--- STARTING DUAL-VALIDATION PIPELINE ---\n")
    result = validate_answer(QUESTION, TRUTH_FILE)

    print("\n--- FINAL REPORT ---")
    print(f"Answer: {result['answer'].strip()}\n")
    print("━" * 60)
    print(f"📊 Baseline (keyword matcher): {result['keyword_score'] * 100}%")
    print(f"👨‍⚖️  LLM-as-a-Judge:           {result['judge_score'] * 100}%")
    print(f"   Judge reasoning: {result['judge_reasoning']}")
    print("━" * 60)
    print(f"⏱️  Answer latency:  {result['latency_answer']}s")
    print(f"⏱️  Judge latency:   {result['latency_judge']}s")
    print(f"⏱️  Total latency:   {round(result['latency_answer'] + result['latency_judge'], 2)}s")
    print("━" * 60)

    if result['judge_score'] < 0.5:
        print("\n⚠️  WARNING: LLM-as-a-Judge flagged this answer as hallucinated.")
    else:
        print("\n✅ The judge confirmed this answer is faithful to the source.")
    print("\n📊 Recent validations in database:")
    stats = get_stats()
    print(f"   Total logged: {stats['total']}")
    if stats['total'] > 0:
        print(f"   Avg keyword score: {stats['avg_keyword_score']}")
        print(f"   Avg judge score:   {stats['avg_judge_score']}")
        print(f"   Hallucinations caught: {stats['hallucinations_caught']}")
