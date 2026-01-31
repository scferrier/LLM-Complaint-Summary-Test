"""
LLM Complaint Summarization Evaluation - Main Pipeline

This script orchestrates the full evaluation pipeline:
1. Load complaint data and ground truth
2. Run Test 1 (extraction) with all models
3. Run Test 2 (summarization + MTD) with all models
4. Evaluate results against ground truth
5. Generate comparison report
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import MODELS, RESULTS_DIR
from data_loader import (
    build_complaints_df,
    load_ground_truth_test1,
    load_ground_truth_test2_orders,
    load_ground_truth_test2,
    GROUND_TRUTH_TEST1_PATH,
    GROUND_TRUTH_TEST2_PATH,
)
from llm_inference import (
    batch_run_test1,
    batch_run_test2,
    load_test_results,
    run_test1_extraction,
    run_test2_summary,
)
from evals import (
    evaluate_all_test1,
    evaluate_all_test2,
    generate_summary_report,
    generate_macro_f1_table,
    generate_test2_results_table,
)


def run_single_test(case_id: str, model: str, test: str) -> dict:
    """Run a single test on one case with one model."""
    df = build_complaints_df()
    case_row = df[df['case_id'] == case_id]

    if case_row.empty:
        print(f"Case not found: {case_id}")
        return {}

    text = case_row['complaint_text'].iloc[0]

    if test == "test1":
        result = run_test1_extraction(text, model)
    else:
        result = run_test2_summary(text, model)

    return result


def run_test1_pipeline(
    models: list = None,
    cases: list = None,
    verbose: bool = True,
    parallel: bool = True
) -> dict:
    """
    Run the Test 1 extraction pipeline.

    Args:
        models: List of model names (defaults to all)
        cases: List of case_ids to process (defaults to all)
        verbose: Print progress
        parallel: Run models in parallel

    Returns:
        dict with results for each model
    """
    # Load data
    df = build_complaints_df()
    if cases:
        df = df[df['case_id'].isin(cases)]

    # Run inference
    results = batch_run_test1(df, models=models, verbose=verbose, parallel=parallel)

    return results


def run_test2_pipeline(
    models: list = None,
    cases: list = None,
    verbose: bool = True,
    parallel: bool = True
) -> dict:
    """
    Run the Test 2 summarization + MTD pipeline.

    Args:
        models: List of model names (defaults to all)
        cases: List of case_ids to process (defaults to all)
        verbose: Print progress
        parallel: Run models in parallel

    Returns:
        dict with results for each model
    """
    # Load data
    df = build_complaints_df()
    if cases:
        df = df[df['case_id'].isin(cases)]

    # Run inference
    results = batch_run_test2(df, models=models, verbose=verbose, parallel=parallel)

    return results


def evaluate_results(compute_factual: bool = True) -> tuple:
    """
    Evaluate all saved results against ground truth.

    Args:
        compute_factual: Whether to compute SummaC and QAFactEval (slow)

    Returns:
        tuple of (test1_scores_df, test2_scores_df)
    """
    print("\n" + "="*60)
    print("EVALUATING RESULTS")
    print("="*60)

    # Load ground truth for Test 1
    try:
        gt1 = load_ground_truth_test1()
        print(f"Loaded Test 1 ground truth: {len(gt1)} cases")
    except FileNotFoundError:
        print("Warning: Test 1 ground truth not found")
        gt1 = pd.DataFrame()

    # Load ground truth for Test 2 (order texts)
    try:
        gt2 = load_ground_truth_test2_orders()
        print(f"Loaded Test 2 ground truth (order texts): {len(gt2)} cases")
    except FileNotFoundError:
        print("Warning: Test 2 order texts not found")
        gt2 = pd.DataFrame()

    # Load ground truth rulings from Excel (for F1 scoring)
    rulings_df = None
    try:
        rulings_df = load_ground_truth_test2()
        print(f"Loaded Test 2 rulings from Excel: {len(rulings_df)} cases")
    except FileNotFoundError:
        print("Warning: Test 2 rulings Excel not found (ruling F1 will be skipped)")

    # Load results for each model
    test1_results = {}
    test2_results = {}

    for model_name in MODELS.keys():
        t1 = load_test_results("test1", model_name)
        if t1:
            test1_results[model_name] = t1
            print(f"  Loaded Test 1 results for {model_name}: {len(t1)} cases")

        t2 = load_test_results("test2", model_name)
        if t2:
            test2_results[model_name] = t2
            print(f"  Loaded Test 2 results for {model_name}: {len(t2)} cases")

    # Evaluate Test 1
    test1_scores = pd.DataFrame()
    if test1_results and not gt1.empty:
        test1_scores = evaluate_all_test1(test1_results, gt1)
        output_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test1_scores.to_csv(output_path, index=False)
        print(f"\nTest 1 scores saved to {output_path}")

        # Generate Macro F1 results table
        generate_macro_f1_table(test1_scores, RESULTS_DIR)

    # Evaluate Test 2
    test2_scores = pd.DataFrame()
    if test2_results and not gt2.empty:
        # Load complaints for factual consistency metrics
        complaints_df = build_complaints_df() if compute_factual else None
        test2_scores = evaluate_all_test2(
            test2_results, gt2,
            complaints_df=complaints_df,
            rulings_df=rulings_df,
            compute_factual=compute_factual
        )
        output_path = Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test2_scores.to_csv(output_path, index=False)
        print(f"Test 2 scores saved to {output_path}")

        # Generate Test 2 results comparison table
        generate_test2_results_table(test2_scores, RESULTS_DIR)

    return test1_scores, test2_scores


def generate_report():
    """Generate the final summary report."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    # Load scores
    test1_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv"
    test2_path = Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"

    test1_scores = pd.read_csv(test1_path) if test1_path.exists() else pd.DataFrame()
    test2_scores = pd.read_csv(test2_path) if test2_path.exists() else pd.DataFrame()

    # Generate Macro F1 table for Test 1
    if not test1_scores.empty:
        generate_macro_f1_table(test1_scores, RESULTS_DIR)

    # Generate Test 2 results comparison table
    if not test2_scores.empty:
        generate_test2_results_table(test2_scores, RESULTS_DIR)

    # Generate report
    report = generate_summary_report(test1_scores, test2_scores)
    print(f"\nReport saved to {RESULTS_DIR}/summary_report.md")
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Complaint Summarization Evaluation Pipeline"
    )
    parser.add_argument(
        "--test1", action="store_true",
        help="Run Test 1 (extraction) pipeline"
    )
    parser.add_argument(
        "--test2", action="store_true",
        help="Run Test 2 (summarization + MTD) pipeline"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate saved results against ground truth"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate summary report from scores"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full pipeline (test1 + test2 + evaluate + report)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to test (default: all). Options: {list(MODELS.keys())}"
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Specific case IDs to process (default: all)"
    )
    parser.add_argument(
        "--single", nargs=3, metavar=("CASE_ID", "MODEL", "TEST"),
        help="Run single test: --single cand_22_cv_02094 gpt4 test1"
    )
    parser.add_argument(
        "--factual", action="store_true", default=True,
        help="Compute factual consistency scores (SummaC + QAFactEval) - enabled by default"
    )
    parser.add_argument(
        "--no-factual", action="store_true",
        help="Skip factual consistency scores (faster evaluation)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True,
        help="Run models in parallel (default: True)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run models sequentially instead of in parallel"
    )

    args = parser.parse_args()

    # Handle parallel/sequential flags
    parallel = not args.sequential

    # If no args provided, show help
    if not any([args.test1, args.test2, args.evaluate, args.report, args.all, args.single]):
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START")
        print("="*60)
        print("\n1. Fill in ground truth data:")
        print("   - data/ground_truth_test1.xlsx")
        print("   - data/ground_truth_test2.xlsx")
        print("\n2. Run a single test to verify setup:")
        print("   python main.py --single cand_22_cv_02094 gpt4 test1")
        print("\n3. Run full pipeline:")
        print("   python main.py --all --models gpt4 claude")
        print("\n4. Or run step by step:")
        print("   python main.py --test1 --models gpt4")
        print("   python main.py --test2 --models gpt4")
        print("   python main.py --evaluate")
        print("   python main.py --report")
        return

    verbose = not args.quiet

    # Single test mode
    if args.single:
        case_id, model, test = args.single
        print(f"\nRunning {test} on {case_id} with {model}...")
        result = run_single_test(case_id, model, test)
        print("\nResult:")
        print(json.dumps(result, indent=2, default=str))
        return

    # Determine whether to compute factual metrics
    compute_factual = args.factual and not args.no_factual

    # Full pipeline
    if args.all:
        run_test1_pipeline(models=args.models, cases=args.cases, verbose=verbose, parallel=parallel)
        run_test2_pipeline(models=args.models, cases=args.cases, verbose=verbose, parallel=parallel)
        evaluate_results(compute_factual=compute_factual)
        generate_report()
        return

    # Individual steps
    if args.test1:
        run_test1_pipeline(models=args.models, cases=args.cases, verbose=verbose, parallel=parallel)

    if args.test2:
        run_test2_pipeline(models=args.models, cases=args.cases, verbose=verbose, parallel=parallel)

    if args.evaluate:
        evaluate_results(compute_factual=compute_factual)

    if args.report:
        generate_report()


if __name__ == "__main__":
    main()
