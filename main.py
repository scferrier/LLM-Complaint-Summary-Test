import argparse, json
from pathlib import Path
import pandas as pd
from config import MODELS, RESULTS_DIR
from data_loader import build_complaints_df, load_ground_truth_test1, load_ground_truth_test2
from llm_inference import batch_run_test1, batch_run_test2, batch_run_test3, batch_run_test4, load_test_results, run_test1_extraction, run_test2_summary, run_test3_summary, run_test4_summary
from evals import evaluate_all_test1, evaluate_all_test2, generate_summary_report, generate_macro_f1_table, generate_test2_results_table


def run_single_test(case_id: str, model: str, test: str) -> dict:
    
    df = build_complaints_df()
    case_row = df[df['case_id'] == case_id]
    
    if case_row.empty: 
        return {}
    text = case_row['complaint_text'].iloc[0]
    
    return run_test1_extraction(text, model) if test == "test1" else run_test2_summary(text, model)

def run_test1_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    
    df = build_complaints_df()
    
    if cases: 
        df = df[df['case_id'].isin(cases)]
    
    return batch_run_test1(df, models=models, verbose=verbose, parallel=parallel)

def run_test2_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    df = build_complaints_df()
    if cases:
        df = df[df['case_id'].isin(cases)]
    return batch_run_test2(df, models=models, verbose=verbose, parallel=parallel)

def run_test3_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    df = build_complaints_df()
    if cases:
        df = df[df['case_id'].isin(cases)]
    return batch_run_test3(df, models=models, verbose=verbose, parallel=parallel)

def run_test4_pipeline(models: list = None, cases: list = None, verbose: bool = True, parallel: bool = True) -> dict:
    df = build_complaints_df()
    if cases:
        df = df[df['case_id'].isin(cases)]
    return batch_run_test4(df, models=models, verbose=verbose, parallel=parallel)


def evaluate_results(compute_factual: bool = True) -> tuple:
    try: 
        gt1 = load_ground_truth_test1()
    except FileNotFoundError: 
        gt1 = pd.DataFrame()
    
    try:
        gt2 = load_ground_truth_test2()
    except FileNotFoundError: 
        gt2 = pd.DataFrame()
    
    rulings_df = gt2 if not gt2.empty else None
    
    test1_results, test2_results = {}, {}
    
    for model_name in MODELS.keys():
        t1 = load_test_results("test1", model_name)
        if t1: test1_results[model_name] = t1
        t2 = load_test_results("test2", model_name)
        if t2: test2_results[model_name] = t2
    
    test1_scores = pd.DataFrame()
    
    if test1_results and not gt1.empty:
        test1_scores = evaluate_all_test1(test1_results, gt1)
        output_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test1_scores.to_csv(output_path, index=False)
        generate_macro_f1_table(test1_scores, RESULTS_DIR)
    
    test2_scores = pd.DataFrame()
    
    if test2_results and not gt2.empty:
        complaints_df = build_complaints_df() if compute_factual else None
        test2_scores = evaluate_all_test2(test2_results, gt2, complaints_df=complaints_df, rulings_df=rulings_df, compute_factual=compute_factual)
        output_path = Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test2_scores.to_csv(output_path, index=False)
        generate_test2_results_table(test2_scores, RESULTS_DIR)
   
    return test1_scores, test2_scores


def generate_report():
    
    test1_path, test2_path = Path(RESULTS_DIR) / "test1" / "scores" / "test1_scores.csv", Path(RESULTS_DIR) / "test2" / "scores" / "test2_scores.csv"
    test1_scores = pd.read_csv(test1_path) if test1_path.exists() else pd.DataFrame()
    test2_scores = pd.read_csv(test2_path) if test2_path.exists() else pd.DataFrame()
    
    if not test1_scores.empty: generate_macro_f1_table(test1_scores, RESULTS_DIR)
    if not test2_scores.empty: generate_test2_results_table(test2_scores, RESULTS_DIR)
    
    generate_summary_report(test1_scores, test2_scores)


def main():
   
    p = argparse.ArgumentParser(description="LLM Complaint Summarization Evaluation Pipeline")
    p.add_argument("--test1", action="store_true"); p.add_argument("--test2", action="store_true"); p.add_argument("--test3", action="store_true"); p.add_argument("--test4", action="store_true")
    p.add_argument("--evaluate", action="store_true"); p.add_argument("--report", action="store_true")
    p.add_argument("--all", action="store_true"); p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--cases", nargs="+", default=None); p.add_argument("--no-factual", action="store_true")
    p.add_argument("--sequential", action="store_true"); p.add_argument("--single", nargs=3, metavar=("CASE", "MODEL", "TEST"))
   
    args = p.parse_args()
    parallel = not args.sequential
    
    if args.single:
        print(json.dumps(run_single_test(*args.single), indent=2, default=str)); return
    
    compute_factual = not getattr(args, 'no_factual', False)
    
    if args.all:
        run_test1_pipeline(models=args.models, cases=args.cases, parallel=parallel)
        run_test2_pipeline(models=args.models, cases=args.cases, parallel=parallel)
        evaluate_results(compute_factual=compute_factual); generate_report(); return
    
    if args.test1: run_test1_pipeline(models=args.models, cases=args.cases, parallel=parallel)
    if args.test2: run_test2_pipeline(models=args.models, cases=args.cases, parallel=parallel)
    if args.test3: run_test3_pipeline(models=args.models, cases=args.cases, parallel=parallel)
    if args.test4: run_test4_pipeline(models=args.models, cases=args.cases, parallel=parallel)
    if args.evaluate: evaluate_results(compute_factual=compute_factual)
    if args.report: generate_report()

if __name__ == "__main__": main()
