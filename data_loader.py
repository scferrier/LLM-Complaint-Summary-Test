"""
Data loading utilities for LLM Complaint Summarization Evaluation.

Handles loading complaint texts, ground truth data, and building evaluation DataFrames.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional


# Default paths
PROCESSED_COMPLAINTS_DIR = "Processed_Text/Compliants"
PROCESSED_ORDERS_DIR = "Processed_Text/Orders_PDFs"
GROUND_TRUTH_TEST1_PATH = "data/ground_truth_test1.xlsx"
GROUND_TRUTH_TEST2_PATH = "data/ground_truth_test2.xlsx"
COURT_SUMMARIES_PATH = "Selected Cases/Court Summaries.xlsx"


def parse_case_id(filename: str) -> dict:
    """
    Parse case ID from filename to extract court, year, and case number.

    Examples:
        cand_22_cv_02094 -> {court: 'cand', year: '22', case_number: '02094'}
        nysd_1_21-cv-11222 -> {court: 'nysd', division: '1', year: '21', case_number: '11222'}
    """
    # Remove file extension
    case_id = filename.replace('.txt', '').replace('.pdf', '')

    # Try to parse with regex - handles various formats
    # Format: [court][-_][division_]?[year][-_]cv[-_][number]
    pattern = r'^([a-z]+)[-_]?(\d)?[-_]?(\d{2})[-_]?cv[-_]?(\d+)[-_]?\d?$'
    match = re.match(pattern, case_id, re.IGNORECASE)

    if match:
        court = match.group(1)
        division = match.group(2) if match.group(2) else None
        year = match.group(3)
        case_number = match.group(4)

        return {
            'case_id': case_id,
            'court': court,
            'division': division,
            'year': f"20{year}",
            'case_number': case_number
        }

    # Fallback: return case_id with empty parsed fields
    return {
        'case_id': case_id,
        'court': None,
        'division': None,
        'year': None,
        'case_number': None
    }


def load_complaint_texts(processed_dir: str = PROCESSED_COMPLAINTS_DIR) -> dict:
    """
    Load all complaint text files from the processed directory.

    Returns:
        dict mapping case_id -> complaint text
    """
    texts = {}
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed complaints directory not found: {processed_dir}")

    for file_path in processed_path.glob("*.txt"):
        case_id = file_path.stem  # filename without extension
        with open(file_path, 'r', encoding='utf-8') as f:
            texts[case_id] = f.read()

    return texts


def load_order_texts(processed_dir: str = PROCESSED_ORDERS_DIR) -> dict:
    """
    Load all order text files from the processed directory.

    Returns:
        dict mapping case_id -> order text
    """
    texts = {}
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        return texts  # Orders may not exist for all cases

    for file_path in processed_path.glob("*.txt"):
        case_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            texts[case_id] = f.read()

    return texts


def load_ground_truth_test1(excel_path: str = GROUND_TRUTH_TEST1_PATH) -> pd.DataFrame:
    """
    Load Test 1 ground truth data (extraction targets).

    Returns:
        DataFrame with columns: case_id, plaintiffs, defendants, ticker,
        class_period_start, class_period_end, cause_1, cause_1_facts, etc.
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Ground truth file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    return df


def load_ground_truth_test2(excel_path: str = GROUND_TRUTH_TEST2_PATH) -> pd.DataFrame:
    """
    Load Test 2 ground truth data (MTD outcomes) from Excel.

    DEPRECATED: Use load_ground_truth_test2_orders() instead for order text comparison.

    Returns:
        DataFrame with columns: case_id, summary_reference, cause_1,
        cause_1_mtd_outcome, cause_1_reasoning, etc.
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Ground truth file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    return df


def load_ground_truth_test2_orders(orders_dir: str = PROCESSED_ORDERS_DIR) -> pd.DataFrame:
    """
    Load Test 2 ground truth from order text files.

    The order texts serve as the reference for evaluating LLM-generated
    complaint summaries and claim rulings.

    Returns:
        DataFrame with columns: case_id, order_text, order_text_length
    """
    texts = load_order_texts(orders_dir)

    if not texts:
        raise FileNotFoundError(f"No order texts found in: {orders_dir}")

    rows = []
    for case_id, text in texts.items():
        rows.append({
            'case_id': case_id,
            'order_text': text,
            'order_text_length': len(text)
        })

    return pd.DataFrame(rows).sort_values('case_id').reset_index(drop=True)


def load_court_summaries(excel_path: str = COURT_SUMMARIES_PATH) -> pd.DataFrame:
    """
    Load court summaries from the existing Excel file.

    Returns:
        DataFrame with case names and order summaries
    """
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Court summaries file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    # Rename columns for consistency
    df = df.rename(columns={'Name': 'case_id', 'Summaries': 'order_summary'})
    # Drop unnamed columns
    df = df[[col for col in df.columns if not col.startswith('Unnamed')]]
    return df


def build_complaints_df(
    complaints_dir: str = PROCESSED_COMPLAINTS_DIR,
    ground_truth_path: Optional[str] = GROUND_TRUTH_TEST1_PATH
) -> pd.DataFrame:
    """
    Build the main complaints DataFrame combining text, metadata, and ground truth.

    Returns:
        DataFrame with columns:
        - case_id, court, division, year, case_number (metadata)
        - complaint_text (full text)
        - text_length (character count)
        - Ground truth columns if available
    """
    # Load complaint texts
    texts = load_complaint_texts(complaints_dir)

    # Build base DataFrame
    rows = []
    for case_id, text in texts.items():
        parsed = parse_case_id(case_id)
        parsed['complaint_text'] = text
        parsed['text_length'] = len(text)
        rows.append(parsed)

    df = pd.DataFrame(rows)

    # Merge with ground truth if available
    if ground_truth_path and Path(ground_truth_path).exists():
        gt_df = load_ground_truth_test1(ground_truth_path)
        df = df.merge(gt_df, on='case_id', how='left')

    return df.sort_values('case_id').reset_index(drop=True)


def build_orders_df(
    orders_dir: str = PROCESSED_ORDERS_DIR,
    court_summaries_path: str = COURT_SUMMARIES_PATH,
    ground_truth_path: Optional[str] = GROUND_TRUTH_TEST2_PATH
) -> pd.DataFrame:
    """
    Build the orders DataFrame for Test 2.

    Returns:
        DataFrame with order text, court summaries, and MTD ground truth
    """
    # Load order texts
    texts = load_order_texts(orders_dir)

    # Build base DataFrame
    rows = []
    for case_id, text in texts.items():
        parsed = parse_case_id(case_id)
        parsed['order_text'] = text
        parsed['order_text_length'] = len(text)
        rows.append(parsed)

    df = pd.DataFrame(rows)

    # Merge with court summaries if available
    if Path(court_summaries_path).exists():
        summaries_df = load_court_summaries(court_summaries_path)
        # Normalize case_id formats for matching
        df = df.merge(summaries_df, on='case_id', how='left')

    # Merge with ground truth if available
    if ground_truth_path and Path(ground_truth_path).exists():
        gt_df = load_ground_truth_test2(ground_truth_path)
        df = df.merge(gt_df.drop(columns=['case_id'], errors='ignore'),
                      left_on='case_id', right_on=gt_df['case_id'] if 'case_id' in gt_df.columns else None,
                      how='left')

    return df.sort_values('case_id').reset_index(drop=True)


def get_case_ids() -> list:
    """Get list of all case IDs from processed complaints."""
    texts = load_complaint_texts()
    return sorted(texts.keys())


if __name__ == "__main__":
    # Test the data loader
    print("Testing data_loader.py...")

    # Test loading complaints
    complaints_df = build_complaints_df()
    print(f"\nLoaded {len(complaints_df)} complaints")
    print(f"Columns: {list(complaints_df.columns)}")
    print(f"\nSample case_ids:")
    for cid in complaints_df['case_id'].head(5):
        print(f"  - {cid}")

    print(f"\nText length stats:")
    print(f"  Min: {complaints_df['text_length'].min():,} chars")
    print(f"  Max: {complaints_df['text_length'].max():,} chars")
    print(f"  Avg: {complaints_df['text_length'].mean():,.0f} chars")
