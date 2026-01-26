# Test 1: Structured Data Extraction
TEST1_SYSTEM_PROMPT = """You are a legal analyst specializing in extracting key information from securities class action complaints.
Your task is to accurately extract specific information from legal complaints.
Always respond with valid JSON only"""

TEST1_USER_PROMPT = """Extract the following information from this securities class action complaint:

1. **Plaintiffs**: List all named plaintiffs (lead plaintiffs and any other named parties)
2. **Defendants**: List all named defendants (companies and individuals)
3. **Ticker Symbol**: If the Defendant is a public company, the stock ticker symbol if mentioned (or null if not found)
4. **Class Period**: The start and end dates of the class period
5. **Causes of Action**: List each cause of action/claim asserted, with a 1-2 sentence summary of the factual basis supporting that claim

Return your response as a JSON object with this exact structure:
{{
    "plaintiffs": ["Plaintiff 1", "Plaintiff 2", ...],
    "defendants": ["Defendant 1", "Defendant 2", ...],
    "ticker": "XXXX" or null,
    "class_period": {{
        "start": "YYYY-MM-DD",
        "end": "YYYY-MM-DD"
    }},
    "causes_of_action": [
        {{
            "claim": "Name of the legal claim (e.g., Section 10(b) and Rule 10b-5)",
            "factual_basis": "1-2 sentence summary of the facts supporting this claim"
        }},
        ...
    ]
}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""


# Test 2: Summarization + MTD Prediction
TEST2_SYSTEM_PROMPT = """You are a neutral legal analyst that specializes in evaluating legal complaints.
Your task is to summarize complaints and predict motion to dismiss outcomes based on legal standards.
Always respond with valid JSON only."""

TEST2_USER_PROMPT = """Analyze this securities class action complaint and provide:

1. **Summary**: A neutral, objective summary of the complaint in 1-3 paragraphs covering:
   - Who the parties are
   - What the alleged misconduct was
   - The key factual allegations
   - The claims being asserted

2. **MTD Predictions**: For each cause of action, predict the likely outcome of a motion to dismiss:
   - Outcome: "granted" (claim dismissed), or "denied" (claim survives)
   - Reasoning: 2-3 sentences explaining why, referencing relevant legal standards (e.g., scienter, materiality, loss causation for 10b-5 claims) and how the facts allege meet or do not meet those standards.

Return your response as a JSON object with this exact structure:
{{
    "summary": "Your 1-3 paragraph neutral summary here...",
    "mtd_predictions": [
        {{
            "claim": "Name of the legal claim",
            "predicted_outcome": "granted" | "denied",
            "reasoning": "2-3 sentence explanation of the prediction"
        }},
        ...
    ]
}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""


def format_test1_prompt(complaint_text: str) -> list:
    """Format messages for Test 1 extraction."""
    return [
        {"role": "system", "content": TEST1_SYSTEM_PROMPT},
        {"role": "user", "content": TEST1_USER_PROMPT.format(complaint_text=complaint_text)}
    ]


def format_test2_prompt(complaint_text: str) -> list:
    """Format messages for Test 2 summarization + MTD prediction."""
    return [
        {"role": "system", "content": TEST2_SYSTEM_PROMPT},
        {"role": "user", "content": TEST2_USER_PROMPT.format(complaint_text=complaint_text)}
    ]
