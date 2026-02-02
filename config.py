import os
from dotenv import load_dotenv
load_dotenv()

PROCESSED_COMPLAINTS_DIR = "data/Processed_Text/Compliants"
PROCESSED_ORDERS_DIR = "data/Processed_Text/Orders_PDFs"
GROUND_TRUTH_TEST1_PATH = "data/ground_truth_test1.xlsx"
GROUND_TRUTH_TEST2_PATH = "data/ground_truth_test2.xlsx"
COURT_SUMMARIES_PATH = "Selected Cases/Court Summaries.xlsx"
RESULTS_DIR = "results"

MODELS = {
    "claude-opus": "claude-opus-4-5-20251101",
    "gemini": "gemini-3-flash-preview",
    "perplexity": "perplexity/sonar-pro",
    "grok": "xai/grok-4-1-fast-reasoning",
    "gpt-5.2": "gpt-5.2",
    "gpt-4.1-finetuned": "ft:gpt-4.1-2025-04-14:personal:complaint-test:D4Z39cJI",
    "gpt-4.1-finetuned-v2": "ft:gpt-4.1-2025-04-14:personal:complmtd3:D4asBVQw",
}
DIRECT_SDK_MODELS = {"gemini"}
MODEL_API_KEYS = {"claude-opus": "ANTHROPIC_API_KEY", "gpt-5.2": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "perplexity": "PERPLEXITY_API_KEY", "grok": "XAI_API_KEY", "gpt-4.1-finetuned": "OPENAI_API_KEY", "gpt-4.1-finetuned-v2": "OPENAI_API_KEY"}

LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 4096
LLM_TIMEOUT = 600
RATE_LIMIT_DELAY = 1.0
MODEL_RATE_LIMITS = {"claude-opus": 1.0, "gemini": 15.0, "perplexity": 2.0, "gpt-5.2": 1.0, "grok": 1.0, "gpt-4.1-finetuned": 1.0, "gpt-4.1-finetuned-v2": 1.0}
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
