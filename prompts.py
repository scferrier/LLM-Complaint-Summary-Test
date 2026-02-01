TEST1_SYSTEM_PROMPT = """You are a legal analyst specializing in extracting key information from securities class action complaints. Always respond with valid JSON only"""

TEST1_USER_PROMPT = """Extract from this securities class action complaint:
1. Plaintiffs: All named plaintiffs
2. Defendants: All named defendants
3. Ticker Symbol: Stock ticker if mentioned (or null)
4. Class Period: Start and end dates
5. Causes of Action: List each claim asserted

Return JSON: {{"plaintiffs": [...], "defendants": [...], "ticker": "XXXX" or null, "class_period": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}, "causes_of_action": [...]}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""

TEST2_SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint. Always respond with valid JSON only."""

TEST2_USER_PROMPT = """Review the complaint and provide:
1. Summary: Comprehensive summary covering parties, alleged misconduct, key facts, legal claims, class period
2. Claim Rulings: For each cause of action, ruling ("dismissed"/"sustained") with reasoning

Return JSON: {{"summary": "...", "claim_rulings": [{{"claim": "...", "ruling": "dismissed"|"sustained"|"dismissed_in_part", "reasoning": "..."}}]}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""

BACKGROUND_EXTRACTION_SYSTEM_PROMPT = """You are a legal document processor. Extract factual background from court orders. Always respond with valid JSON only."""

BACKGROUND_EXTRACTION_USER_PROMPT = """Extract the factual background section from this court order. Clean by removing legal citations, paragraph symbols, case citations. Preserve all factual content.

Return JSON: {{"background": "Complete cleaned factual background..."}}

COURT ORDER TEXT:
{order_text}

JSON RESPONSE:"""

def format_test1_prompt(complaint_text: str) -> list:
    return [{"role": "system", "content": TEST1_SYSTEM_PROMPT}, {"role": "user", "content": TEST1_USER_PROMPT.format(complaint_text=complaint_text)}]

def format_test2_prompt(complaint_text: str) -> list:
    return [{"role": "system", "content": TEST2_SYSTEM_PROMPT}, {"role": "user", "content": TEST2_USER_PROMPT.format(complaint_text=complaint_text)}]

def format_background_extraction_prompt(order_text: str) -> list:
    return [{"role": "system", "content": BACKGROUND_EXTRACTION_SYSTEM_PROMPT}, {"role": "user", "content": BACKGROUND_EXTRACTION_USER_PROMPT.format(order_text=order_text)}]
