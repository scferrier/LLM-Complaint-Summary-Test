# CLAUDE.md

This file provides context for Claude Code when working on this project.

## Project Overview

LLM Complaint Summary Test is a benchmarking framework for evaluating Large Language Model performance on legal document summarization tasks. The focus is on complex civil complaints from securities and commodities litigation.

### Goals

1. **Test One**: Evaluate LLM ability to extract objective information (plaintiffs, defendants, ticker symbols, causes of action, key factual allegations)
2. **Test Two**: Assess LLM summarization quality and ability to predict motion-to-dismiss outcomes
3. **Fine-tuning**: Improve top models using Legal-BERT embeddings and re-evaluate

## Tech Stack

- **Python**: 3.13+
- **Package Manager**: uv (pyproject.toml)
- **Key Dependencies**:
  - `litellm` - LLM API abstraction
  - `evaluate` - Evaluation metrics framework
  - `summac` - Summarization consistency evaluation
  - `qafacteval` - QA-based factual evaluation
  - `requests` - HTTP requests
  - `pandas` - Data manipulation
  - `python-dotenv` - Environment variables

## Project Structure

```
├── main.py              # Entry point (currently stub)
├── scrappers.py         # Data collection from Court Listener, GovInfo, Justia APIs
├── pyproject.toml       # Project configuration
├── requirements         # Dependencies list
├── .env                 # API keys (not in repo): COURT_LISTNER
├── Selected Cases/
│   ├── Compliants/      # Legal complaint PDFs for evaluation
│   └── Orders/
│       └── PDFs/        # Court order PDFs for evaluation
```

## Environment Variables

Required in `.env`:
- `COURT_LISTNER` - Court Listener API token

## External APIs

- **Court Listener API**: `https://www.courtlistener.com/api/rest/v4/`
- **GovInfo API**: `https://www.govinfo.gov/wssearch/getContentDetail`
- **Justia**: For legal document retrieval

## Evaluation Metrics

- **ROUGE, BLEU, METEOR**: Surface-level text similarity
- **BERTScore**: Semantic similarity
- **SummaC**: Summarization consistency
- **QAFactEval**: Question-answering factual evaluation

## Data

- **Evaluation set**: 25 complaints & orders
- **Fine-tuning set**: 100 complaints & orders
- **Source**: Securities/commodities cases filed in last 5 years
- **Cleaning applied**: Headers/footers removed, case captions stripped, OCR applied

## Commands

```bash
# Run the main script
python main.py

# Install dependencies (with uv)
uv pip install -r requirements
```

## Code Style

- Use type hints where appropriate
- Follow PEP 8 conventions
- Document functions with docstrings for complex logic
- Keep API keys in `.env`, never commit secrets

## Current Status

The project is in early development:
- `main.py` is a stub
- `scrappers.py` has partial implementation for Court Listener API
- README sections marked #TODO: Models/Libraries/Code, Model Results, Finetuning, Finetuned Model Results, Conclusion

## Domain Context

Legal complaints are lengthy, fact-dense documents with layered narratives. Key challenges:
- Long context lengths that degrade LLM performance
- Multiple (sometimes contradictory) theories of liability
- Structured legal terminology and citation formats
- Motion to dismiss analysis requires legal reasoning

## TODOs 

