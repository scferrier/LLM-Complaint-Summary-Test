"""
Build JSONL for fine-tuning using full complaint texts as input
and cleaned backgrounds + rulings as output.

Focus: Improve ruling predictions by training on full complaint context.
"""

import json
import re
import pandas as pd
from pathlib import Path
from clean_pdf import process_pdf
import tiktoken

COMPLAINTS_DIR = Path("data/training_dataset/complaints_matched")
MATCHED_FILE = Path("data/training_dataset/matched_fillenames.xlsx")
TRAINING_DATA = Path("data/training_with_rulings.xlsx")
OUTPUT_JSONL = Path("data/fine_tune_complaints.jsonl")

MAX_TOKENS = 100000  # Eliminate anything over 100K tokens

# Prompts matching Test2 format for consistency
SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint. Always respond with valid JSON only."""

USER_PROMPT = """Review the complaint and provide:
1. Summary: Comprehensive summary covering parties, alleged misconduct, key facts, legal claims, class period
2. Claim Rulings: For each cause of action, predict ruling ("dismissed" or "sustained") with reasoning

Return JSON: {{"summary": "...", "claim_rulings": [{{"claim": "...", "ruling": "dismissed"|"sustained", "reasoning": "..."}}]}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""


def normalize_case_id(case_id: str) -> str:
    """Normalize case ID for matching."""
    case_id = re.sub(r'\.pdf$', '', case_id, flags=re.IGNORECASE)
    case_id = case_id.replace('_', '-')
    case_id = re.sub(r'-(amended-)?complaint$', '', case_id, flags=re.IGNORECASE)
    case_id = re.sub(r'-(first|second|third)-amended-complaint$', '', case_id, flags=re.IGNORECASE)
    case_id = case_id.lower()
    # Normalize cv number
    case_id = re.sub(r'-cv-0*(\d+)', lambda m: f'-cv-{m.group(1).zfill(5)}', case_id)
    # Remove trailing version numbers
    case_id = re.sub(r'(-cv-\d+)-\d+$', r'\1', case_id)
    return case_id


def parse_rulings(rulings_text: str) -> list:
    """Parse rulings text into structured format."""
    if pd.isna(rulings_text) or not rulings_text:
        return []

    rulings = []
    items = re.split(r'\d+\.\s+', str(rulings_text))

    for item in items:
        if not item.strip():
            continue

        match = re.match(r'(.+?):\s*(dismissed|sustained|dismissed[_\s]in[_\s]part)\s*[-–]\s*(.+)',
                        item.strip(), re.IGNORECASE | re.DOTALL)
        if match:
            claim = match.group(1).strip()
            ruling_raw = match.group(2).lower().replace(' ', '_')
            # Binary: dismissed_in_part -> sustained
            ruling = 'dismissed' if ruling_raw == 'dismissed' else 'sustained'
            reasoning = match.group(3).strip()
            rulings.append({
                "claim": claim,
                "ruling": ruling,
                "reasoning": reasoning[:500]  # Truncate long reasoning
            })

    return rulings


def main():
    enc = tiktoken.encoding_for_model('gpt-4')

    # Load matched filenames
    matched_df = pd.read_excel(MATCHED_FILE)
    print(f"Matched pairs: {len(matched_df)}")

    # Load training data with backgrounds and rulings
    training_df = pd.read_excel(TRAINING_DATA)
    training_df['norm_id'] = training_df['filename'].apply(normalize_case_id)
    print(f"Training data with rulings: {len(training_df)}")

    examples = []
    skipped_no_complaint = []
    skipped_no_match = []
    skipped_no_rulings = []
    skipped_too_large = []

    for _, row in matched_df.iterrows():
        complaint_file = row['complaint filename']
        order_file = row['order filename']

        # Normalize IDs for matching
        order_norm = normalize_case_id(order_file)

        # Find matching training data
        match = training_df[training_df['norm_id'] == order_norm]
        if match.empty:
            skipped_no_match.append(order_file)
            continue

        match_row = match.iloc[0]

        # Check for rulings
        if pd.isna(match_row.get('rulings')):
            skipped_no_rulings.append(order_file)
            continue

        # Extract complaint text
        complaint_path = COMPLAINTS_DIR / complaint_file
        if not complaint_path.exists():
            skipped_no_complaint.append(complaint_file)
            continue

        try:
            complaint_text, _ = process_pdf(str(complaint_path))
        except Exception as e:
            print(f"Error processing {complaint_file}: {e}")
            skipped_no_complaint.append(complaint_file)
            continue

        # Check token count
        user_content = USER_PROMPT.format(complaint_text=complaint_text)
        tokens = len(enc.encode(SYSTEM_PROMPT + user_content))

        if tokens > MAX_TOKENS:
            # Truncate complaint text
            max_complaint_tokens = MAX_TOKENS - 2000  # Leave room for prompt and response
            complaint_tokens = enc.encode(complaint_text)
            if len(complaint_tokens) > max_complaint_tokens:
                complaint_text = enc.decode(complaint_tokens[:max_complaint_tokens])
                user_content = USER_PROMPT.format(complaint_text=complaint_text)
                tokens = len(enc.encode(SYSTEM_PROMPT + user_content))

        if tokens > MAX_TOKENS:
            skipped_too_large.append((complaint_file, tokens))
            continue

        # Build response
        background = match_row.get('background_clean', '')
        if pd.isna(background):
            background = ''

        rulings = parse_rulings(match_row.get('rulings'))

        if not rulings:
            skipped_no_rulings.append(order_file)
            continue

        response = {
            "summary": background[:8000] if len(background) > 8000 else background,
            "claim_rulings": rulings
        }

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)}
            ]
        }
        examples.append(example)

        print(f"Added: {complaint_file} ({tokens:,} tokens, {len(rulings)} rulings)")

    # Write output
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Summary
    print("\n" + "=" * 60)
    print("FINE-TUNING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated examples: {len(examples)}")
    print(f"Output: {OUTPUT_JSONL}")
    print()
    print(f"Skipped - no complaint file: {len(skipped_no_complaint)}")
    print(f"Skipped - no training match: {len(skipped_no_match)}")
    print(f"Skipped - no rulings: {len(skipped_no_rulings)}")
    print(f"Skipped - too large: {len(skipped_too_large)}")

    # Ruling distribution
    all_rulings = []
    for ex in examples:
        resp = json.loads(ex['messages'][2]['content'])
        for r in resp.get('claim_rulings', []):
            all_rulings.append(r['ruling'])

    print()
    print("Ruling distribution:")
    print(f"  dismissed: {all_rulings.count('dismissed')}")
    print(f"  sustained: {all_rulings.count('sustained')}")


if __name__ == "__main__":
    main()
