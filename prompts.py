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


# Test3: Scienter-focused prompt for ruling predictions
TEST3_SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint. You must apply the heightened pleading standards of the Private Securities Litigation Reform Act (PSLRA).

CRITICAL INSTRUCTION - SCRUTINIZE SCIENTER:
The PSLRA requires plaintiffs to state with PARTICULARITY facts giving rise to a STRONG INFERENCE that defendant acted with intent to deceive or reckless disregard for the truth. This is not satisfied by:
- Conclusory allegations (e.g., "defendants knew or should have known")
- Allegations that amount to "fraud by hindsight"
- Speculation about what defendants must have known

You should DISMISS Section 10(b) and Rule 10b-5 claims where:
- There are no SPECIFIC facts showing defendant's actual knowledge of falsity at the time statements were made
- The complaint relies on general corporate knowledge rather than particularized facts about individual defendants
- Innocent explanations for defendant's conduct are equally or more plausible than fraudulent intent

IMPORTANT: If you dismiss a Section 10(b)/Rule 10b-5 claim, you MUST also dismiss any Section 20(a) control person liability claims, as they are derivative of the underlying securities violation.

Always respond with valid JSON only."""

TEST3_USER_PROMPT = """Review this securities class action complaint and rule on each cause of action.

For EACH claim involving Section 10(b) or Rule 10b-5, you must determine:
1. Are there PARTICULARIZED factual allegations showing the defendant KNEW their statements were false when made?
2. Do the facts create a STRONG INFERENCE of scienter, or are innocent explanations equally plausible?

If scienter is NOT adequately pleaded with specific, particularized facts, DISMISS the claim.

REMEMBER: If you dismiss the Section 10(b)/Rule 10b-5 claim, you must ALSO dismiss Section 20(a) claims as they depend on an underlying violation.

COMPLAINT TEXT:
{complaint_text}

Return JSON: {{"summary": "...", "claim_rulings": [{{"claim": "...", "ruling": "dismissed"|"sustained", "scienter_analysis": "..."}}]}}

JSON RESPONSE:"""


def format_test3_prompt(complaint_text: str) -> list:
    return [{"role": "system", "content": TEST3_SYSTEM_PROMPT}, {"role": "user", "content": TEST3_USER_PROMPT.format(complaint_text=complaint_text)}]
