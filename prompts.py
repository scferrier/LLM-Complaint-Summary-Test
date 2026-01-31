# Test 1: Structured Data Extraction
TEST1_SYSTEM_PROMPT = """You are a legal analyst specializing in extracting key information from securities class action complaints.
Your task is to accurately extract specific information from legal complaints.
Always respond with valid JSON only"""

TEST1_USER_PROMPT = """Extract the following information from this securities class action complaint:

1. **Plaintiffs**: List all named plaintiffs (lead plaintiffs and any other named parties)
2. **Defendants**: List all named defendants (companies and individuals)
3. **Ticker Symbol**: If the Defendant is a public company, the stock ticker symbol if mentioned (or null if not found)
4. **Class Period**: The start and end dates of the class period
5. **Causes of Action**: List each cause of action/claim asserted

Return your response as a JSON object with this exact structure:
{{
    "plaintiffs": ["Plaintiff 1", "Plaintiff 2", ...],
    "defendants": ["Defendant 1", "Defendant 2", ...],
    "ticker": "XXXX" or null,
    "class_period": {{
        "start": "YYYY-MM-DD",
        "end": "YYYY-MM-DD"
    }},
    "causes_of_action": ["Section 10(b) and Rule 10b-5", "Section 20(a)", ...]
}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""


# Test 2: Judicial Officer - Complaint Summary & Claim Rulings
TEST2_SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint.
Your task is to summarize the complaint and issue rulings on each claim asserted.
Always respond with valid JSON only."""

TEST2_USER_PROMPT = """You are presiding over this securities class action case. The defendants have filed a motion to dismiss.
Review the complaint and provide:

1. **Summary**: A comprehensive summary of the complaint covering:
   - The parties involved (plaintiffs and defendants)
   - The alleged fraudulent conduct or misconduct
   - The key factual allegations supporting the claims
   - The legal claims being asserted
   - The class period and any significant dates

2. **Claim Rulings**: For each cause of action asserted in the complaint, issue your ruling:
   - Ruling: "dismissed" (motion granted as to this claim), "sustained" (motion denied, claim survives), or "dismissed_in_part" (partially granted)
   - Reasoning: Provide detailed legal analysis explaining your ruling. Reference the applicable legal standards (e.g., for Section 10(b) claims: material misrepresentation or omission, scienter, reliance, loss causation; for Section 20(a) claims: primary violation and control person liability). Explain how the factual allegations do or do not satisfy each element.

Return your response as a JSON object with this exact structure:
{{
    "summary": "Your comprehensive summary of the complaint...",
    "claim_rulings": [
        {{
            "claim": "Name of the legal claim (e.g., Section 10(b) and Rule 10b-5)",
            "ruling": "dismissed" | "sustained" | "dismissed_in_part",
            "reasoning": "Detailed legal analysis explaining your ruling..."
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
