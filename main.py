import argparse
import os
import warnings

# Suppress noisy output from dependencies
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

from pathlib import Path
import pandas as pd
from config import MODELS, RESULTS_DIR
from data_loader import build_complaints_df, load_ground_truth_test1, load_ground_truth_test2
from llm_inference import batch_run_test1, batch_run_test2, batch_run_test3, load_test_results
from evals import evaluate_all_test1, evaluate_all_test2, evaluate_all_test3, generate_summary_report, generate_macro_f1_table, generate_test2_results_table, generate_ruling_comparison_table


def run_test1_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:

    df = build_complaints_df()
    gt1 = load_ground_truth_test1()
    df = df[df['case_id'].isin(gt1['case_id'])]

    if cases:
        df = df[df['case_id'].isin(cases)]

    return batch_run_test1(df, models=models, verbose=verbose, parallel=parallel)

def run_test2_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    df = build_complaints_df()
    gt2 = load_ground_truth_test2()
    df = df[df['case_id'].isin(gt2['case_id'])]

    if cases:
        df = df[df['case_id'].isin(cases)]
    return batch_run_test2(df, models=models, verbose=verbose, parallel=parallel)

def run_test3_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    df = build_complaints_df()
    gt2 = load_ground_truth_test2()
    df = df[df['case_id'].isin(gt2['case_id'])]

    if cases:
        df = df[df['case_id'].isin(cases)]
    return batch_run_test3(df, models=models, verbose=verbose, parallel=parallel)

def evaluate_results(tests_to_eval: set = None) -> tuple:
    """Evaluate results. If tests_to_eval is provided, only evaluate those tests."""
    try:
        gt1 = load_ground_truth_test1()
    except FileNotFoundError:
        gt1 = pd.DataFrame()

    try:
        gt2 = load_ground_truth_test2()
    except FileNotFoundError:
        gt2 = pd.DataFrame()

    rulings_df = gt2 if not gt2.empty else None

    test1_results, test2_results, test3_results = {}, {}, {}

    for model_name in MODELS.keys():
        if tests_to_eval is None or 'test1' in tests_to_eval:
            t1 = load_test_results("test1", model_name)
            if t1: test1_results[model_name] = t1
        if tests_to_eval is None or 'test2' in tests_to_eval:
            t2 = load_test_results("test2", model_name)
            if t2: test2_results[model_name] = t2
        if tests_to_eval is None or 'test3' in tests_to_eval:
            t3 = load_test_results("test3", model_name)
            if t3: test3_results[model_name] = t3

    test1_scores = pd.DataFrame()

    if test1_results and not gt1.empty:
        test1_scores = evaluate_all_test1(test1_results, gt1)
        output_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test1_scores.to_csv(output_path, index=False)
        generate_macro_f1_table(test1_scores, RESULTS_DIR)

    test2_scores = pd.DataFrame()

    if test2_results and not gt2.empty:
        complaints_df = build_complaints_df()
        test2_scores = evaluate_all_test2(test2_results, gt2, complaints_df=complaints_df, rulings_df=rulings_df, compute_factual=True)
        output_path = Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test2_scores.to_csv(output_path, index=False)
        generate_test2_results_table(test2_scores, RESULTS_DIR)

    test3_scores = pd.DataFrame()

    if test3_results and not gt2.empty:
        test3_scores = evaluate_all_test3(test3_results, gt2)
        output_path = Path(RESULTS_DIR) / "test3" / "scores" / "test3_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test3_scores.to_csv(output_path, index=False)

    # Generate ruling comparison table (test2 vs test3)
    if not test2_scores.empty or not test3_scores.empty:
        generate_ruling_comparison_table(test2_scores, test3_scores, RESULTS_DIR)

    return test1_scores, test2_scores, test3_scores


def generate_report():

    test1_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv"
    test2_path = Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"
    test3_path = Path(RESULTS_DIR) / "test3" / "scores" / "test3_scores.csv"

    test1_scores = pd.read_csv(test1_path) if test1_path.exists() else pd.DataFrame()
    test2_scores = pd.read_csv(test2_path) if test2_path.exists() else pd.DataFrame()
    test3_scores = pd.read_csv(test3_path) if test3_path.exists() else pd.DataFrame()

    if not test1_scores.empty: generate_macro_f1_table(test1_scores, RESULTS_DIR)
    if not test2_scores.empty: generate_test2_results_table(test2_scores, RESULTS_DIR)
    if not test2_scores.empty or not test3_scores.empty:
        generate_ruling_comparison_table(test2_scores, test3_scores, RESULTS_DIR)

    generate_summary_report(test1_scores, test2_scores, test3_scores)


def main():

    p = argparse.ArgumentParser(description="LLM Complaint Summarization Evaluation Pipeline")
    p.add_argument("--test1", action="store_true"); p.add_argument("--test2", action="store_true"); p.add_argument("--test3", action="store_true")
    p.add_argument("--all", action="store_true"); p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--sequential", action="store_true")

    args = p.parse_args()
    parallel = not args.sequential

    tests_run = set()

    if args.all:
        run_test1_pipeline(models=args.models, cases=args.cases, parallel=parallel)
        run_test2_pipeline(models=args.models, cases=args.cases, parallel=parallel)
        run_test3_pipeline(models=args.models, cases=args.cases, parallel=parallel)
        tests_run = {'test1', 'test2', 'test3'}

    if args.test1: run_test1_pipeline(models=args.models, cases=args.cases, parallel=parallel); tests_run.add('test1')
    if args.test2: run_test2_pipeline(models=args.models, cases=args.cases, parallel=parallel); tests_run.add('test2')
    if args.test3: run_test3_pipeline(models=args.models, cases=args.cases, parallel=parallel); tests_run.add('test3')

    if tests_run:
        print("\nTest calls complete. Starting evaluation and report generation...")
        evaluate_results(tests_run)
        generate_report()
        print("Report generation complete.")

if __name__ == "__main__": main()
