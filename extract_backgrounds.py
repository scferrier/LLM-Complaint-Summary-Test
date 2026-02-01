"""
Extract and clean background sections from court orders using LLM.

This script processes court order PDFs, extracts the factual background
section, and cleans it of legal citations to produce ground truth summaries.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional

from clean_pdf import process_pdf
from prompts import format_background_extraction_prompt
from llm_inference import call_llm
from config import MODELS


def extract_background_from_order(
    order_text: str,
    model: str = "perplexity",
    max_tokens: int = 8000
) -> Optional[str]:
    """
    Extract and clean background section from a court order using LLM.

    Args:
        order_text: The full text of the court order
        model: LLM model name (e.g., "perplexity", "gpt4o", "claude")
        max_tokens: Maximum tokens for response

    Returns:
        Cleaned background text, or None if extraction fails
    """
    messages = format_background_extraction_prompt(order_text)

    # Look up model ID from config
    model_id = MODELS.get(model, model)

    try:
        response = call_llm(
            messages=messages,
            model=model_id,
            model_name=model,
            max_tokens=max_tokens,
            temperature=0.1  # Low temperature for consistent extraction
        )

        if response and response.get('success'):
            # call_llm returns {'success': True, 'response': {...parsed JSON...}}
            data = response.get('response', {})

            # Handle case where response is already parsed
            if isinstance(data, dict):
                background = data.get('background', '')
                if background:
                    return background
                # Try raw_response if background not found
                if 'raw_response' in data:
                    return data['raw_response']

            return None
        else:
            error = response.get('error', 'Unknown error') if response else 'No response'
            print(f"  LLM call failed: {error}")
            return None

    except Exception as e:
        print(f"  Error calling LLM: {e}")
        return None


def process_all_orders(
    orders_dir: str = "Selected Cases/Orders/PDFs",
    output_path: str = "data/extracted_backgrounds.csv",
    model: str = "perplexity"
) -> pd.DataFrame:
    """
    Process all order PDFs and extract background sections.

    Args:
        orders_dir: Directory containing order PDFs
        output_path: Path to save extracted backgrounds
        model: LLM model to use

    Returns:
        DataFrame with case_id and extracted background
    """
    results = []

    pdf_files = sorted([f for f in os.listdir(orders_dir) if f.endswith('.pdf')])
    print(f"Found {len(pdf_files)} order PDFs to process")

    for i, pdf_file in enumerate(pdf_files):
        case_id = pdf_file.replace('.pdf', '')
        pdf_path = os.path.join(orders_dir, pdf_file)

        print(f"\n[{i+1}/{len(pdf_files)}] Processing {case_id}...")

        try:
            # Extract and clean text from PDF
            order_text, _ = process_pdf(pdf_path)
            print(f"  Extracted {len(order_text)} chars from PDF")

            # Use LLM to extract and clean background
            background = extract_background_from_order(order_text, model=model)

            if background:
                print(f"  Extracted background: {len(background)} chars")
                results.append({
                    'case_id': case_id,
                    'background': background,
                    'background_length': len(background)
                })
            else:
                print(f"  Failed to extract background")
                results.append({
                    'case_id': case_id,
                    'background': None,
                    'background_length': 0
                })

        except Exception as e:
            print(f"  Error processing {case_id}: {e}")
            results.append({
                'case_id': case_id,
                'background': None,
                'background_length': 0
            })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} extracted backgrounds to {output_path}")

    # Summary
    successful = df['background'].notna().sum()
    print(f"Successfully extracted: {successful}/{len(df)}")

    return df


def update_ground_truth(
    extracted_path: str = "data/extracted_backgrounds.csv",
    ground_truth_path: str = "data/ground_truth_test2.xlsx"
):
    """
    Update ground truth Excel with extracted backgrounds.

    Args:
        extracted_path: Path to extracted backgrounds CSV
        ground_truth_path: Path to ground truth Excel
    """
    # Load extracted backgrounds
    extracted_df = pd.read_csv(extracted_path)

    # Load existing ground truth
    gt_df = pd.read_excel(ground_truth_path)
    gt_df.columns = gt_df.columns.str.strip()

    # Update summaries with extracted backgrounds
    for _, row in extracted_df.iterrows():
        case_id = row['case_id']
        background = row['background']

        if pd.notna(background):
            mask = gt_df['case_id'] == case_id
            if mask.any():
                gt_df.loc[mask, 'summary'] = background
                print(f"Updated {case_id}")

    # Save updated ground truth
    gt_df.to_excel(ground_truth_path, index=False)
    print(f"\nUpdated {ground_truth_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract backgrounds from court orders")
    parser.add_argument("--model", default="perplexity", help="LLM model to use")
    parser.add_argument("--orders-dir", default="Selected Cases/Orders/PDFs", help="Orders directory")
    parser.add_argument("--output", default="data/extracted_backgrounds.csv", help="Output CSV path")
    parser.add_argument("--update-gt", action="store_true", help="Update ground truth Excel")
    parser.add_argument("--single", type=str, help="Process a single case ID")

    args = parser.parse_args()

    if args.single:
        # Process single case
        pdf_path = os.path.join(args.orders_dir, f"{args.single}.pdf")
        if os.path.exists(pdf_path):
            order_text, _ = process_pdf(pdf_path)
            background = extract_background_from_order(order_text, model=args.model)
            print("\n=== EXTRACTED BACKGROUND ===")
            print(background)
        else:
            print(f"PDF not found: {pdf_path}")
    else:
        # Process all orders
        df = process_all_orders(
            orders_dir=args.orders_dir,
            output_path=args.output,
            model=args.model
        )

        if args.update_gt:
            update_ground_truth(args.output)
