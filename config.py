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
# Order: Slow models staggered (claude-opus first, gpt-5.2 last) to avoid end bottleneck
MODELS = {
    "claude-opus": "claude-opus-4-5-20251101",  # Anthropic Claude Opus 4.5 (SLOW - starts first)
    "gemini": "gemini-3-flash-preview",          # Google Gemini (direct SDK passthrough)
    "perplexity": "perplexity/sonar-pro",        # Perplexity Sonar Pro
    "grok": "xai/grok-4-1-fast-reasoning",       # xAI Grok 4.1 Fast Reasoning
    "gpt-5.2": "gpt-5.2",                        # OpenAI GPT-5.2 (SLOW - starts last)
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

# Model-specific rate limits (override default)
MODEL_RATE_LIMITS = {
    "claude-opus": 1.0,   # 450K tokens/min - high rate limit
    "gemini": 15.0,       # 15 seconds for free tier (5 req/min)
    "perplexity": 2.0,    # Perplexity has moderate rate limits
    "gpt-5.2": 1.0,       # GPT-5.2 has high rate limits
    "grok": 1.0,          # Grok has high rate limits
}

# Evaluation settings
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
SUMMAC_MODEL = "vitc"
