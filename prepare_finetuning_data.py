"""
Prepare finetuning dataset by extracting and cleaning complaints and orders.

This script processes PDF documents for finetuning and creates a ground truth
spreadsheet template similar to ground_truth_test2.xlsx.

Usage:
    # Extract and clean all PDFs, create ground truth template
    python prepare_finetuning_data.py --complaints-dir path/to/complaints --orders-dir path/to/orders

    # Just process complaints
    python prepare_finetuning_data.py --complaints-dir path/to/complaints --complaints-only

    # Just process orders (with LLM background extraction)
    python prepare_finetuning_data.py --orders-dir path/to/orders --orders-only --extract-backgrounds

    # Create ground truth template from existing processed text
    python prepare_finetuning_data.py --create-template --output-dir data/finetuning
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from clean_pdf import process_pdf
from extract_backgrounds import extract_background_from_order


def get_case_id_from_filename(filename: str) -> str:
    """Extract case ID from PDF filename."""
    return filename.replace('.pdf', '').replace('.PDF', '')


def process_complaint_pdf(
    pdf_path: str,
    output_dir: str
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Process a single complaint PDF.

    Returns:
        Tuple of (case_id, cleaned_text, error_message)
    """
    case_id = get_case_id_from_filename(os.path.basename(pdf_path))

    try:
        text, _ = process_pdf(pdf_path, output_dir=output_dir)

        if text:
            # Save cleaned text
            output_path = os.path.join(output_dir, f"{case_id}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return case_id, text, None
        else:
            return case_id, None, "No text extracted"

    except Exception as e:
        return case_id, None, str(e)


def process_order_pdf(
    pdf_path: str,
    output_dir: str,
    extract_background: bool = False,
    model: str = "perplexity"
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Process a single order PDF.

    Returns:
        Tuple of (case_id, cleaned_text, background_text, error_message)
    """
    case_id = get_case_id_from_filename(os.path.basename(pdf_path))

    try:
        text, _ = process_pdf(pdf_path, output_dir=output_dir)

        if not text:
            return case_id, None, None, "No text extracted"

        # Save full cleaned text
        output_path = os.path.join(output_dir, f"{case_id}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Optionally extract background with LLM
        background = None
        if extract_background:
            print(f"  Extracting background for {case_id}...")
            background = extract_background_from_order(text, model=model)
            if background:
                bg_path = os.path.join(output_dir, f"{case_id}_background.txt")
                with open(bg_path, 'w', encoding='utf-8') as f:
                    f.write(background)

        return case_id, text, background, None

    except Exception as e:
        return case_id, None, None, str(e)


def process_complaints(
    complaints_dir: str,
    output_dir: str,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Process all complaint PDFs in a directory.

    Returns:
        DataFrame with case_id, text_length, success status
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = sorted([
        f for f in os.listdir(complaints_dir)
        if f.lower().endswith('.pdf')
    ])

    print(f"\nProcessing {len(pdf_files)} complaint PDFs...")
    results = []

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(complaints_dir, pdf_file)
        print(f"[{i+1}/{len(pdf_files)}] {pdf_file}")

        case_id, text, error = process_complaint_pdf(pdf_path, output_dir)

        results.append({
            'case_id': case_id,
            'text_length': len(text) if text else 0,
            'success': text is not None,
            'error': error
        })

        if text:
            print(f"  ✓ Extracted {len(text):,} chars")
        else:
            print(f"  ✗ Failed: {error}")

    df = pd.DataFrame(results)

    # Summary
    success_count = df['success'].sum()
    print(f"\nComplaints processed: {success_count}/{len(df)}")

    return df


def process_orders(
    orders_dir: str,
    output_dir: str,
    extract_backgrounds: bool = False,
    model: str = "perplexity"
) -> pd.DataFrame:
    """
    Process all order PDFs in a directory.

    Returns:
        DataFrame with case_id, text_length, background_length, success status
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = sorted([
        f for f in os.listdir(orders_dir)
        if f.lower().endswith('.pdf')
    ])

    print(f"\nProcessing {len(pdf_files)} order PDFs...")
    results = []

    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(orders_dir, pdf_file)
        print(f"[{i+1}/{len(pdf_files)}] {pdf_file}")

        case_id, text, background, error = process_order_pdf(
            pdf_path, output_dir,
            extract_background=extract_backgrounds,
            model=model
        )

        results.append({
            'case_id': case_id,
            'text_length': len(text) if text else 0,
            'background_length': len(background) if background else 0,
            'has_background': background is not None,
            'success': text is not None,
            'error': error
        })

        if text:
            msg = f"  ✓ Extracted {len(text):,} chars"
            if background:
                msg += f" (background: {len(background):,} chars)"
            print(msg)
        else:
            print(f"  ✗ Failed: {error}")

    df = pd.DataFrame(results)

    # Summary
    success_count = df['success'].sum()
    bg_count = df['has_background'].sum() if extract_backgrounds else 0
    print(f"\nOrders processed: {success_count}/{len(df)}")
    if extract_backgrounds:
        print(f"Backgrounds extracted: {bg_count}/{len(df)}")

    return df


def create_ground_truth_template(
    case_ids: List[str],
    output_path: str,
    num_causes: int = 15
) -> pd.DataFrame:
    """
    Create an empty ground truth spreadsheet template.

    The template has the same structure as ground_truth_test2.xlsx:
    - case_id: Case identifier
    - summary: Expected summary/background (to be filled in)
    - cause_1 through cause_N: Cause of action names
    - cause_1_mtd_outcome through cause_N_mtd_outcome: MTD rulings (granted/denied/partial)

    Args:
        case_ids: List of case IDs to include
        output_path: Path to save the Excel file
        num_causes: Maximum number of causes of action columns (default 15)

    Returns:
        DataFrame with the template
    """
    # Build column list
    columns = ['case_id', 'summary']
    for i in range(1, num_causes + 1):
        columns.append(f'cause_{i}')
        columns.append(f'cause_{i}_mtd_outcome')

    # Create empty DataFrame
    df = pd.DataFrame(columns=columns)

    # Add case IDs
    df['case_id'] = case_ids

    # Save to Excel
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"\nCreated ground truth template: {output_path}")
    print(f"  - {len(case_ids)} cases")
    print(f"  - {num_causes} cause of action slots")
    print("\nColumns to fill in:")
    print("  - summary: The expected summary/background text")
    print("  - cause_N: Name of the Nth cause of action")
    print("  - cause_N_mtd_outcome: MTD ruling (granted/denied/partial)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare finetuning dataset from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process complaints and orders, create ground truth template
    python prepare_finetuning_data.py \\
        --complaints-dir "Finetuning Cases/Complaints" \\
        --orders-dir "Finetuning Cases/Orders" \\
        --output-dir "data/finetuning"

    # Only process complaints
    python prepare_finetuning_data.py \\
        --complaints-dir "Finetuning Cases/Complaints" \\
        --complaints-only \\
        --output-dir "data/finetuning/complaints"

    # Process orders with LLM background extraction
    python prepare_finetuning_data.py \\
        --orders-dir "Finetuning Cases/Orders" \\
        --orders-only \\
        --extract-backgrounds \\
        --model perplexity \\
        --output-dir "data/finetuning/orders"

    # Create template from already processed files
    python prepare_finetuning_data.py \\
        --create-template \\
        --processed-dir "data/finetuning/complaints" \\
        --output-dir "data/finetuning"
        """
    )

    # Input directories
    parser.add_argument(
        "--complaints-dir",
        help="Directory containing complaint PDFs"
    )
    parser.add_argument(
        "--orders-dir",
        help="Directory containing order PDFs"
    )
    parser.add_argument(
        "--processed-dir",
        help="Directory with already-processed text files (for --create-template)"
    )

    # Processing options
    parser.add_argument(
        "--complaints-only",
        action="store_true",
        help="Only process complaints"
    )
    parser.add_argument(
        "--orders-only",
        action="store_true",
        help="Only process orders"
    )
    parser.add_argument(
        "--extract-backgrounds",
        action="store_true",
        help="Use LLM to extract background sections from orders"
    )
    parser.add_argument(
        "--model",
        default="perplexity",
        help="LLM model for background extraction (default: perplexity)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        default="data/finetuning",
        help="Output directory for processed text and template"
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Only create ground truth template (no PDF processing)"
    )
    parser.add_argument(
        "--template-name",
        default="ground_truth_finetuning.xlsx",
        help="Name for the ground truth template file"
    )
    parser.add_argument(
        "--num-causes",
        type=int,
        default=15,
        help="Number of cause of action columns in template (default: 15)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.create_template:
        # Need either processed-dir or one of the input dirs
        if not args.processed_dir and not args.complaints_dir and not args.orders_dir:
            parser.error("--create-template requires --processed-dir, --complaints-dir, or --orders-dir")
    elif not args.complaints_only and not args.orders_only:
        # Full processing - need both dirs
        if not args.complaints_dir or not args.orders_dir:
            parser.error("Provide both --complaints-dir and --orders-dir, or use --complaints-only/--orders-only")
    elif args.complaints_only and not args.complaints_dir:
        parser.error("--complaints-only requires --complaints-dir")
    elif args.orders_only and not args.orders_dir:
        parser.error("--orders-only requires --orders-dir")

    case_ids = []

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    if args.create_template:
        # Just create template from existing processed files
        source_dir = args.processed_dir or args.complaints_dir or args.orders_dir

        if args.processed_dir:
            # Get case IDs from processed text files
            case_ids = [
                f.replace('.txt', '')
                for f in os.listdir(source_dir)
                if f.endswith('.txt') and not f.endswith('_background.txt')
            ]
        else:
            # Get case IDs from PDF files
            case_ids = [
                get_case_id_from_filename(f)
                for f in os.listdir(source_dir)
                if f.lower().endswith('.pdf')
            ]
    else:
        # Process PDFs
        complaints_output = os.path.join(args.output_dir, "complaints")
        orders_output = os.path.join(args.output_dir, "orders")

        if not args.orders_only and args.complaints_dir:
            complaints_df = process_complaints(args.complaints_dir, complaints_output)
            case_ids.extend(complaints_df[complaints_df['success']]['case_id'].tolist())

            # Save processing summary
            complaints_df.to_csv(
                os.path.join(args.output_dir, "complaints_processing_log.csv"),
                index=False
            )

        if not args.complaints_only and args.orders_dir:
            orders_df = process_orders(
                args.orders_dir,
                orders_output,
                extract_backgrounds=args.extract_backgrounds,
                model=args.model
            )

            # Add order case IDs if not already from complaints
            order_case_ids = orders_df[orders_df['success']]['case_id'].tolist()
            for cid in order_case_ids:
                if cid not in case_ids:
                    case_ids.append(cid)

            # Save processing summary
            orders_df.to_csv(
                os.path.join(args.output_dir, "orders_processing_log.csv"),
                index=False
            )

    # Create ground truth template
    if case_ids:
        case_ids = sorted(set(case_ids))  # Deduplicate and sort
        template_path = os.path.join(args.output_dir, args.template_name)
        create_ground_truth_template(case_ids, template_path, args.num_causes)
    else:
        print("\nNo cases found to create template.")

    print(f"\n✓ Done! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
