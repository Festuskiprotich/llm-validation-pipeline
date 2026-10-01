# LLM Validation Pipeline

A production-grade pipeline for validating LLM outputs, tracking latency and cost, and catching hallucinations before they reach users.

## What It Does

- Wraps any LLM (local via Ollama, or cloud via API) behind a validated interface
- Checks every response against a source of truth
- Logs latency, token usage, cost, and validation scores
- Exposes a FastAPI service for any app to use
- Ships with a dashboard for monitoring

## Why

Large Language Models hallucinate. Studies in 2026 report hallucination rates between 22% and 94%. This is unacceptable for production use. This project builds the missing layer between LLMs and real users.

## Tech Stack

- **Python 3.11**
- **Ollama** (local LLM inference)
- **FastAPI** (HTTP API)
- **SQLite** (metrics storage)
- **Streamlit** (dashboard)
- **Docker** (deployment)

## Status

🚧 In development. Built in public.

## License

MIT