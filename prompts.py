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


# Background Extraction: Extract and clean factual background from court orders
BACKGROUND_EXTRACTION_SYSTEM_PROMPT = """You are a legal document processor specializing in extracting factual content from court orders.
Your task is to extract the factual background section and produce clean, readable prose.
Always respond with valid JSON only."""

BACKGROUND_EXTRACTION_USER_PROMPT = """Extract the factual background from this court order (motion to dismiss ruling).

Court orders typically contain these sections:
- BACKGROUND / FACTUAL BACKGROUND / FACTS (this is what we want)
- LEGAL STANDARD
- DISCUSSION / ANALYSIS
- CONCLUSION / ORDER

Your task:
1. **Identify** the factual background section (usually labeled "Background", "Factual Background", "Facts", or "I. Background")
2. **Extract** all factual content including:
   - Description of the parties (plaintiffs, defendants, their roles)
   - The company's business and operations
   - The alleged misconduct or fraudulent scheme
   - Key dates and events
   - Stock price movements and financial impacts
   - Any confidential witness allegations
3. **Clean** the text by removing:
   - All legal citations (e.g., "Id. ¶ 45", "(Doc. 36 ¶ 18)", "Id. at 5")
   - Paragraph symbols and numbers (¶, ¶¶)
   - Case citations (e.g., "(9th Cir. 2021)")
   - Footnote references
   - Parenthetical annotations like "(cleaned up)", "(emphasis added)"
   - ECF references
4. **Preserve**:
   - All factual content and narrative
   - Defined terms like ("UANPF") or (the "Class Period")
   - Quotations from the complaint or company statements
   - Section structure (A. The Parties, B. The Scheme, etc.) converted to prose

Return your response as a JSON object:
{{
    "background": "The complete, cleaned factual background as continuous prose. Include all parties, allegations, dates, and events. This should read as a comprehensive factual summary without any legal citations or paragraph references."
}}

COURT ORDER TEXT:
{order_text}

JSON RESPONSE:"""


def format_background_extraction_prompt(order_text: str) -> list:
    """Format messages for background extraction from court orders."""
    return [
        {"role": "system", "content": BACKGROUND_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": BACKGROUND_EXTRACTION_USER_PROMPT.format(order_text=order_text)}
    ]


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
