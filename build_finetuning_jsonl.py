"""
Merge rulings from extraction_review.xlsx into training data and build JSONL for fine-tuning.
"""

import json
import re
import argparse
import pandas as pd
from pathlib import Path


SYSTEM_PROMPT = """You are a federal district court judge reviewing a motion to dismiss a securities class action complaint. Analyze the complaint's factual background and provide a summary with rulings on each claim."""

USER_PROMPT = """Review this securities class action complaint background and provide:
1. A comprehensive summary of the case
2. Your ruling on each cause of action (dismissed, sustained, or dismissed_in_part) with reasoning

COMPLAINT BACKGROUND:
{background}

Respond with JSON only:"""


def normalize_case_id(case_id: str) -> str:
    """Normalize case ID for matching between files."""
    # Remove .pdf extension
    case_id = re.sub(r'\.pdf$', '', case_id, flags=re.IGNORECASE)
    # Replace underscores with hyphens
    case_id = case_id.replace('_', '-')
    # Lowercase first
    case_id = case_id.lower()

    # Normalize court codes (map different formats to canonical form)
    court_mappings = {
        'dma': 'mad',      # Massachusetts
        'ddc': 'dcd',      # DC District
        'sdfl': 'flsd',    # Southern District of Florida
        'sdin': 'insd',    # Southern District of Indiana
    }

    # Extract and normalize court code
    match = re.match(r'^([a-z]+)-', case_id)
    if match:
        court = match.group(1)
        if court in court_mappings:
            case_id = court_mappings[court] + case_id[len(court):]

    # Normalize cv number format (e.g., cv-486 -> cv-00486, cv-05937 -> cv-05937)
    case_id = re.sub(r'-cv-0*(\d+)', lambda m: f'-cv-{m.group(1).zfill(5)}', case_id)

    # Remove trailing document version numbers AFTER cv number (e.g., -cv-00111-1 -> -cv-00111)
    case_id = re.sub(r'(-cv-\d+)-\d+$', r'\1', case_id)

    return case_id


def parse_rulings_to_json(rulings_text: str) -> list:
    """Parse rulings text into structured JSON format.

    Rulings are normalized to binary: 'dismissed' or 'sustained' only.
    'dismissed_in_part' maps to 'dismissed'.
    """
    if pd.isna(rulings_text) or not rulings_text:
        return []

    rulings = []
    # Split by numbered items (1. , 2. , etc.)
    items = re.split(r'\d+\.\s+', rulings_text)

    for item in items:
        if not item.strip():
            continue

        # Parse format: "Claim Name: ruling - reasoning"
        match = re.match(r'(.+?):\s*(dismissed|sustained|dismissed[_\s]in[_\s]part)\s*[-–]\s*(.+)', item.strip(), re.IGNORECASE)
        if match:
            claim = match.group(1).strip()
            ruling_raw = match.group(2).lower().replace(' ', '_')
            # Normalize to binary: dismissed_in_part -> sustained (partial survival)
            if ruling_raw == 'dismissed':
                ruling = 'dismissed'
            else:  # sustained or dismissed_in_part
                ruling = 'sustained'
            reasoning = match.group(3).strip()
            rulings.append({
                "claim": claim,
                "ruling": ruling,
                "reasoning": reasoning
            })
        else:
            # Try simpler format - skip if can't parse ruling
            continue

    return rulings


def merge_and_build_jsonl(
    training_path: str = "data/training_extracted_backgrounds_cleaned.xlsx",
    review_path: str = "data/extraction_review.xlsx",
    output_excel: str = "data/training_with_rulings.xlsx",
    output_jsonl: str = "data/fine_tune_backgrounds.jsonl"
) -> tuple:
    """Merge rulings into training data and build JSONL."""

    # Load data
    training = pd.read_excel(training_path)
    review = pd.read_excel(review_path)

    print(f"Training data: {len(training)} rows")
    print(f"Review data: {len(review)} rows")

    # Create normalized ID columns for matching
    training['norm_id'] = training['filename'].apply(normalize_case_id)
    review['norm_id'] = review['case_id'].apply(normalize_case_id)

    # Merge on normalized ID
    merged = training.merge(
        review[['norm_id', 'rulings', 'ready_for_jsonl', 'causes_of_action']],
        on='norm_id',
        how='left'
    )

    # Check matches
    matched = merged['rulings'].notna().sum()
    print(f"Matched cases with rulings: {matched}/{len(training)}")

    # Show unmatched for debugging
    unmatched = merged[merged['rulings'].isna()]['filename'].tolist()
    if unmatched:
        print(f"\nUnmatched files ({len(unmatched)}):")
        for f in unmatched[:10]:
            print(f"  - {f}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # Save merged Excel
    merged.to_excel(output_excel, index=False)
    print(f"\nSaved merged data to: {output_excel}")

    # Build JSONL for fine-tuning
    examples = []
    skipped = []

    for idx, row in merged.iterrows():
        if pd.isna(row['rulings']) or pd.isna(row['background_clean']):
            skipped.append((row['filename'], 'missing rulings or background'))
            continue

        if row.get('ready_for_jsonl') == 'No':
            skipped.append((row['filename'], 'marked not ready'))
            continue

        background = row['background_clean']
        rulings_json = parse_rulings_to_json(row['rulings'])

        if not rulings_json:
            skipped.append((row['filename'], 'could not parse rulings'))
            continue

        # Build the assistant response
        assistant_response = {
            "summary": background[:5000] if len(background) > 5000 else background,  # Summary from background
            "claim_rulings": rulings_json
        }

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(background=background)},
                {"role": "assistant", "content": json.dumps(assistant_response, ensure_ascii=False)}
            ]
        }
        examples.append(example)

    # Write JSONL
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nGenerated {len(examples)} fine-tuning examples to: {output_jsonl}")
    if skipped:
        print(f"Skipped {len(skipped)} cases:")
        for filename, reason in skipped[:10]:
            print(f"  - {filename}: {reason}")

    return len(examples), len(skipped)


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning JSONL from training data with rulings")
    parser.add_argument("--training", "-t", default="data/training_extracted_backgrounds_cleaned.xlsx",
                        help="Path to training backgrounds Excel")
    parser.add_argument("--review", "-r", default="data/extraction_review.xlsx",
                        help="Path to extraction review Excel with rulings")
    parser.add_argument("--output-excel", "-e", default="data/training_with_rulings.xlsx",
                        help="Output path for merged Excel")
    parser.add_argument("--output-jsonl", "-j", default="data/fine_tune_backgrounds.jsonl",
                        help="Output path for JSONL")

    args = parser.parse_args()

    merge_and_build_jsonl(
        args.training,
        args.review,
        args.output_excel,
        args.output_jsonl
    )


if __name__ == "__main__":
    main()
