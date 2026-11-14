from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_validator.validator import validate_answer

app = FastAPI(
    title="LLM Validation Pipeline",
    description="Production-grade LLM output validation with dual scoring",
    version="0.1.0",
)

class ValidationRequest(BaseModel):
    question: str

class ValidationResponse(BaseModel):
    question: str
    answer: str
    keyword_score: float
    judge_score: float
    judge_reasoning: str
    latency_seconds: float

@app.get("/")
def root():
    return {"status": "ok", "service": "LLM Validation Pipeline"}

@app.post("/validate", response_model=ValidationResponse)
def validate(request: ValidationRequest):
    """Validate an LLM answer against the source of truth."""
    try:
        result = validate_answer(
            question=request.question,
            truth_filepath="data/source_of_truth/llm_facts.txt",
        )
        return ValidationResponse(
            question=request.question,
            answer=result["answer"],
            keyword_score=result["keyword_score"],
            judge_score=result["judge_score"],
            judge_reasoning=result["judge_reasoning"],
            latency_seconds=round(result["latency_answer"] + result["latency_judge"], 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
