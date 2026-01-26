"""
Configuration settings for LLM Complaint Summarization Evaluation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROCESSED_COMPLAINTS_DIR = "Processed_Text/Compliants"
PROCESSED_ORDERS_DIR = "Processed_Text/Orders_PDFs"
GROUND_TRUTH_TEST1_PATH = "data/ground_truth_test1.xlsx"
GROUND_TRUTH_TEST2_PATH = "data/ground_truth_test2.xlsx"
COURT_SUMMARIES_PATH = "Selected Cases/Court Summaries.xlsx"
RESULTS_DIR = "results"

# LLM Models to test (litellm format)
# See: https://docs.litellm.ai/docs/providers
MODELS = {
    "claude-opus": "claude-opus-4-5-20251101",  # Anthropic Claude Opus 4.5
    "gpt-5.2": "gpt-5.2",                        # OpenAI GPT-5.2
    "gemini": "gemini-3-flash-preview",          # Google Gemini (direct SDK passthrough)
    "perplexity": "perplexity/llama-3.1-sonar-large-128k-online",
    "grok": "xai/grok-4-1-fast-reasoning",       # xAI Grok 4.1 Fast Reasoning
}

# Models that use direct SDK instead of litellm
DIRECT_SDK_MODELS = {"gemini"}

# Model-specific API key environment variable names
MODEL_API_KEYS = {
    "claude-opus": "ANTHROPIC_API_KEY",
    "gpt-5.2": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "grok": "XAI_API_KEY",
}

# LLM settings
LLM_TEMPERATURE = 0.6  # Deterministic outputs for reproducibility
LLM_MAX_TOKENS = 4096  # Max response tokens
LLM_TIMEOUT = 600  # Seconds

# Rate limiting (seconds between API calls)
RATE_LIMIT_DELAY = 1.0

# Evaluation settings
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SUMMAC_MODEL = "vitc"
