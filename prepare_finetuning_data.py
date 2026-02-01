import json, os
from pathlib import Path
import pandas as pd
import tiktoken
from clean_pdf import process_pdf
from llm_inference import call_llm
from config import MODELS

TRAINING_COMPLAINTS_DIR = "data/training_dataset/complaints_matched"
TRAINING_ORDERS_DIR = "data/training_dataset/orders_matched"
MAX_TOKENS = 120000  # Leave buffer under 128K

SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint. Always respond with valid JSON only."""

USER_PROMPT = """Review the complaint and provide:
1. Summary: Comprehensive summary covering parties (Plaintiffs/Defendants), alleged misconduct, class period, legal claims and key facts supporting each claim.
2. Claim Rulings: For each cause of action, state if you think it should be dismissed or sustained and provide your reasoning.

Return JSON: {{"summary": "...", "claim_rulings": [{{"claim": "...", "ruling": "dismissed"|"sustained"|"dismissed_in_part", "reasoning": "..."}}]}}

COMPLAINT TEXT:
{complaint_text}

JSON RESPONSE:"""

EXTRACT_PROMPT = """Extract from this court order:
1. The factual background section (the judge's summary of the case facts, parties, and allegations)
2. The ruling for each cause of action/claim

Return JSON: {{"summary": "Complete factual background...", "claim_rulings": [{{"claim": "name of claim", "ruling": "dismissed"|"sustained"|"dismissed_in_part", "reasoning": "brief reasoning"}}]}}

COURT ORDER TEXT:
{order_text}

JSON RESPONSE:"""

def get_matching_files() -> list:
    complaints = {f.stem.replace('-amended-complaint', '').replace('-complaint', '').replace('-first-amended-complaint', '').replace('-second-amended-complaint', ''): f
                  for f in Path(TRAINING_COMPLAINTS_DIR).glob("*.pdf")}
    orders = {f.stem: f for f in Path(TRAINING_ORDERS_DIR).glob("*.pdf")}

    matches = []
    for order_id, order_path in orders.items():
        # Try to find matching complaint
        complaint_path = None
        for comp_id, comp_path in complaints.items():
            # Normalize both IDs for comparison
            norm_order = order_id.replace('_', '-').replace(' ', '-').lower()
            norm_comp = comp_id.replace('_', '-').replace(' ', '-').lower()
            if norm_order in norm_comp or norm_comp in norm_order:
                complaint_path = comp_path
                break
        if complaint_path:
            matches.append((order_id, complaint_path, order_path))
    return matches

def extract_ground_truth_from_order(order_text: str, model: str = "gpt-5.2") -> dict:
    messages = [{"role": "system", "content": "You are a legal document processor. Extract information from court orders. Always respond with valid JSON only."},
                {"role": "user", "content": EXTRACT_PROMPT.format(order_text=order_text[:50000])}]  # Truncate long orders
    resp = call_llm(messages, MODELS.get(model, model), model_name=model, max_tokens=8000, temperature=0.1)
    if resp.get('success'):
        return resp.get('response', {})
    return {}

def generate_finetuning_jsonl(output_path: str = "data/fine_tune.jsonl", model: str = "gpt-5.2", skip_extraction: bool = False, cache_path: str = "data/training_cache.json") -> int:
    enc = tiktoken.encoding_for_model('gpt-4')
    matches = get_matching_files()
    print(f"Found {len(matches)} matched complaint-order pairs")

    # Load cache if exists
    cache = {}
    if Path(cache_path).exists():
        cache = json.load(open(cache_path, 'r', encoding='utf-8'))
        print(f"Loaded {len(cache)} cached extractions")

    examples, skipped_large, skipped_extraction = [], [], []

    for case_id, complaint_path, order_path in matches:
        print(f"\nProcessing {case_id}...")

        # Extract complaint text
        try:
            complaint_text, _ = process_pdf(str(complaint_path))
        except Exception as e:
            print(f"  SKIP: Failed to process complaint - {e}")
            continue

        # Check token count
        user_content = USER_PROMPT.format(complaint_text=complaint_text)
        tokens = len(enc.encode(SYSTEM_PROMPT + user_content))
        if tokens > MAX_TOKENS:
            print(f"  SKIP: Too large ({tokens:,} tokens > {MAX_TOKENS:,})")
            skipped_large.append((case_id, tokens))
            continue

        # Get ground truth from order (use cache or extract)
        if case_id in cache and not skip_extraction:
            ground_truth = cache[case_id]
            print(f"  Using cached extraction")
        elif skip_extraction:
            print(f"  SKIP: No cache and skip_extraction=True")
            continue
        else:
            try:
                order_text, _ = process_pdf(str(order_path))
                ground_truth = extract_ground_truth_from_order(order_text, model)
                if not ground_truth.get('summary'):
                    print(f"  SKIP: Failed to extract summary from order")
                    skipped_extraction.append(case_id)
                    continue
                cache[case_id] = ground_truth
                # Save cache after each extraction
                json.dump(cache, open(cache_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  SKIP: Failed to process order - {e}")
                skipped_extraction.append(case_id)
                continue

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(ground_truth, ensure_ascii=False)}
            ]
        }
        examples.append(example)
        print(f"  OK: {tokens:,} tokens, {len(ground_truth.get('claim_rulings', []))} rulings")

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Generated {len(examples)} fine-tuning examples to {output_path}")
    print(f"Skipped {len(skipped_large)} (too large):")
    for case_id, tokens in skipped_large:
        print(f"  - {case_id}: {tokens:,} tokens")
    print(f"Skipped {len(skipped_extraction)} (extraction failed)")
    return len(examples)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate fine-tuning JSONL from training dataset")
    p.add_argument("--output", "-o", default="data/fine_tune.jsonl")
    p.add_argument("--model", "-m", default="gpt-5.2", help="Model for extracting ground truth from orders")
    p.add_argument("--skip-extraction", action="store_true", help="Only use cached extractions")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    count = generate_finetuning_jsonl(args.output, args.model, args.skip_extraction)

    if args.validate and count > 0:
        enc = tiktoken.encoding_for_model('gpt-4')
        with open(args.output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                data = json.loads(line)
                tokens = len(enc.encode(''.join(m['content'] for m in data['messages'])))
                assert tokens < 128000, f"Line {i}: {tokens} tokens exceeds limit"
        print(f"Validation passed: all {count} examples under 128K tokens")
