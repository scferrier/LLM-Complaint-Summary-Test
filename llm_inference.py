import json
import time
import re
import os
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import litellm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    MODELS,
    DIRECT_SDK_MODELS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    RATE_LIMIT_DELAY,
    MODEL_RATE_LIMITS,
    RESULTS_DIR,
)
from prompts import format_test1_prompt, format_test2_prompt

# Load environment variables
load_dotenv()

# Thread-safe progress tracking
_progress_lock = threading.Lock()

# Suppress duplicate tqdm output in threading
tqdm.monitor_interval = 0


def call_gemini_direct(messages: list, model: str, temperature: float = LLM_TEMPERATURE,
                       max_tokens: int = LLM_MAX_TOKENS) -> dict:
    """
    Call Gemini via the direct Google AI SDK (genai.Client).
    """
    try:
        import google.generativeai as genai

        # Initialize client with API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # Convert messages to Gemini format
        # Gemini expects a single prompt or contents list
        system_prompt = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]

        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content

        # Call Gemini
        client = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        response = client.generate_content(full_prompt)
        content = response.text

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = {"raw_response": content}

        # Get usage if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
            }

        return {
            "success": True,
            "response": parsed,
            "usage": usage
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "usage": None
        }


def call_llm_litellm(messages: list, model: str, temperature: float = LLM_TEMPERATURE,
                     max_tokens: int = LLM_MAX_TOKENS, timeout: int = LLM_TIMEOUT) -> dict:
    """
    Call an LLM via litellm.
    """
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        content = response.choices[0].message.content

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = {"raw_response": content}

        return {
            "success": True,
            "response": parsed,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "usage": None
        }


def call_llm(messages: list, model: str, model_name: str = None,
             temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS,
             timeout: int = LLM_TIMEOUT) -> dict:
    """
    Route LLM calls to appropriate backend (litellm or direct SDK).
    """
    # Check if this model uses direct SDK
    if model_name and model_name in DIRECT_SDK_MODELS:
        if model_name == "gemini":
            return call_gemini_direct(messages, model, temperature, max_tokens)

    # Default to litellm
    return call_llm_litellm(messages, model, temperature, max_tokens, timeout)


def run_test1_extraction(complaint_text: str, model_name: str, model_id: Optional[str] = None) -> dict:
    """Run Test 1 extraction on a single complaint."""
    if model_id is None:
        model_id = MODELS.get(model_name)
        if model_id is None:
            return {"success": False, "error": f"Unknown model: {model_name}"}

    messages = format_test1_prompt(complaint_text)
    result = call_llm(messages, model_id, model_name=model_name)

    return {
        "model": model_name,
        "model_id": model_id,
        "test": "test1",
        **result
    }


def run_test2_summary(complaint_text: str, model_name: str, model_id: Optional[str] = None) -> dict:
    """Run Test 2 summarization + MTD prediction on a single complaint."""
    if model_id is None:
        model_id = MODELS.get(model_name)
        if model_id is None:
            return {"success": False, "error": f"Unknown model: {model_name}"}

    messages = format_test2_prompt(complaint_text)
    result = call_llm(messages, model_id, model_name=model_name)

    return {
        "model": model_name,
        "model_id": model_id,
        "test": "test2",
        **result
    }


def _run_model_test1(model_name: str, cases_df: pd.DataFrame, pbar: tqdm,
                     save_results: bool = True) -> tuple:
    """
    Run Test 1 for a single model across all cases.
    Returns (model_name, results_list).
    """
    results = []

    for idx, row in cases_df.iterrows():
        case_id = row['case_id']
        complaint_text = row['complaint_text']

        result = run_test1_extraction(complaint_text, model_name)
        result['case_id'] = case_id
        results.append(result)

        # Update progress bar (thread-safe)
        status = "OK" if result['success'] else "ERR"
        with _progress_lock:
            pbar.update(1)
            pbar.set_postfix_str(f"{model_name}: {case_id[:20]}... {status}")

        # Rate limiting (model-specific)
        delay = MODEL_RATE_LIMITS.get(model_name, RATE_LIMIT_DELAY)
        time.sleep(delay)

    # Save results for this model
    if save_results:
        save_test_results(results, "test1", model_name)

    return model_name, results


def _run_model_test2(model_name: str, cases_df: pd.DataFrame, pbar: tqdm,
                     save_results: bool = True, start_delay: float = 0.0) -> tuple:
    """
    Run Test 2 for a single model across all cases.
    Returns (model_name, results_list).

    Args:
        start_delay: Seconds to wait before starting (for staggered parallel execution)
    """
    # Staggered start to prevent slow models from bunching at the end
    if start_delay > 0:
        time.sleep(start_delay)

    results = []

    for idx, row in cases_df.iterrows():
        case_id = row['case_id']
        complaint_text = row['complaint_text']

        result = run_test2_summary(complaint_text, model_name)
        result['case_id'] = case_id
        results.append(result)

        # Update progress bar (thread-safe)
        status = "OK" if result['success'] else "ERR"
        with _progress_lock:
            pbar.update(1)
            pbar.set_postfix_str(f"{model_name}: {case_id[:20]}... {status}")

        # Rate limiting (model-specific)
        delay = MODEL_RATE_LIMITS.get(model_name, RATE_LIMIT_DELAY)
        time.sleep(delay)

    # Save results for this model
    if save_results:
        save_test_results(results, "test2", model_name)

    return model_name, results


def batch_run_test1(cases_df: pd.DataFrame, models: Optional[list] = None,
                    save_results: bool = True, verbose: bool = True,
                    parallel: bool = True) -> dict:
    """
    Run Test 1 extraction across all cases and models.

    Args:
        cases_df: DataFrame with case_id and complaint_text columns
        models: List of model names to test (default: all)
        save_results: Save results to files
        verbose: Print progress info
        parallel: Run models in parallel (default: True)

    Returns:
        dict mapping model_name -> list of results
    """
    if models is None:
        models = list(MODELS.keys())

    total_calls = len(cases_df) * len(models)

    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST 1: STRUCTURED DATA EXTRACTION")
        print(f"{'='*60}")
        print(f"Cases: {len(cases_df)} | Models: {len(models)} | Total calls: {total_calls}")
        print(f"Models: {', '.join(models)}")
        print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"{'='*60}\n")

    all_results = {}

    # Create progress bar
    pbar = tqdm(total=total_calls, desc="Test 1 Progress",
                unit="call", ncols=100, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                dynamic_ncols=True)

    if parallel and len(models) > 1:
        # Run models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(_run_model_test1, model, cases_df, pbar, save_results): model
                for model in models
            }

            for future in as_completed(futures):
                model_name, results = future.result()
                all_results[model_name] = results
                if verbose:
                    success_count = sum(1 for r in results if r.get('success'))
                    tqdm.write(f"  Completed {model_name}: {success_count}/{len(results)} successful")
    else:
        # Sequential execution
        for model_name in models:
            _, results = _run_model_test1(model_name, cases_df, pbar, save_results)
            all_results[model_name] = results
            if verbose:
                success_count = sum(1 for r in results if r.get('success'))
                tqdm.write(f"  Completed {model_name}: {success_count}/{len(results)} successful")

    pbar.close()

    if verbose:
        print(f"\n{'='*60}")
        print("TEST 1 COMPLETE")
        for model, results in all_results.items():
            success = sum(1 for r in results if r.get('success'))
            print(f"  {model}: {success}/{len(results)} successful")
        print(f"{'='*60}\n")

    return all_results


def batch_run_test2(cases_df: pd.DataFrame, models: Optional[list] = None,
                    save_results: bool = True, verbose: bool = True,
                    parallel: bool = True) -> dict:
    """
    Run Test 2 summarization + MTD prediction across all cases and models.

    Args:
        cases_df: DataFrame with case_id and complaint_text columns
        models: List of model names to test (default: all)
        save_results: Save results to files
        verbose: Print progress info
        parallel: Run models in parallel (default: True)

    Returns:
        dict mapping model_name -> list of results
    """
    if models is None:
        models = list(MODELS.keys())

    total_calls = len(cases_df) * len(models)

    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST 2: SUMMARIZATION + MTD PREDICTION")
        print(f"{'='*60}")
        print(f"Cases: {len(cases_df)} | Models: {len(models)} | Total calls: {total_calls}")
        print(f"Models: {', '.join(models)}")
        print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"{'='*60}\n")

    all_results = {}

    # Create progress bar
    pbar = tqdm(total=total_calls, desc="Test 2 Progress",
                unit="call", ncols=100, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                dynamic_ncols=True)

    if parallel and len(models) > 1:
        # Staggered start delays to spread out slow models
        # claude-opus starts first (0s), gpt-5.2 starts last (5s)
        stagger_delays = {
            "claude-opus": 0.0,   # Slow - starts first
            "gemini": 1.0,
            "perplexity": 2.0,
            "grok": 3.0,
            "gpt-5.2": 5.0,       # Slow - starts last
        }

        # Run models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(
                    _run_model_test2, model, cases_df, pbar, save_results,
                    stagger_delays.get(model, 0.0)
                ): model
                for model in models
            }

            for future in as_completed(futures):
                model_name, results = future.result()
                all_results[model_name] = results
                if verbose:
                    success_count = sum(1 for r in results if r.get('success'))
                    tqdm.write(f"  Completed {model_name}: {success_count}/{len(results)} successful")
    else:
        # Sequential execution
        for model_name in models:
            _, results = _run_model_test2(model_name, cases_df, pbar, save_results)
            all_results[model_name] = results
            if verbose:
                success_count = sum(1 for r in results if r.get('success'))
                tqdm.write(f"  Completed {model_name}: {success_count}/{len(results)} successful")

    pbar.close()

    if verbose:
        print(f"\n{'='*60}")
        print("TEST 2 COMPLETE")
        for model, results in all_results.items():
            success = sum(1 for r in results if r.get('success'))
            print(f"  {model}: {success}/{len(results)} successful")
        print(f"{'='*60}\n")

    return all_results


def save_test_results(results: list, test_name: str, model_name: str):
    """Save test results to JSON and Excel files."""
    output_dir = Path(RESULTS_DIR) / test_name / "raw_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"{model_name}_outputs.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON to {json_path}")

    # Save Excel for manual review
    excel_path = output_dir / f"{model_name}_outputs.xlsx"
    export_results_to_excel(results, test_name, excel_path)
    print(f"  Saved Excel to {excel_path}")


def export_results_to_excel(results: list, test_name: str, output_path: Path):
    """
    Export LLM results to Excel for manual review.
    Flattens the nested JSON structure into readable columns.
    Includes evaluation scores if ground truth is available.
    """
    from data_loader import load_ground_truth_test1, load_ground_truth_test2
    from evals import clean_ground_truth_row, evaluate_test1_case, evaluate_test2_case

    # Try to load ground truth for scoring
    ground_truth = None
    try:
        if test_name == 'test1':
            ground_truth = load_ground_truth_test1()
        elif test_name == 'test2':
            ground_truth = load_ground_truth_test2()
    except FileNotFoundError:
        pass

    rows = []

    for result in results:
        if not result.get('success'):
            rows.append({
                'case_id': result.get('case_id', ''),
                'model': result.get('model', ''),
                'success': False,
                'error': result.get('error', ''),
            })
            continue

        response = result.get('response', {})
        case_id = result.get('case_id', '')
        row = {
            'case_id': case_id,
            'model': result.get('model', ''),
            'success': True,
        }

        if test_name == 'test1':
            # Test 1: Extraction fields
            row['plaintiffs'] = '; '.join(response.get('plaintiffs', []))
            row['defendants'] = '; '.join(response.get('defendants', []))
            row['ticker'] = response.get('ticker', '')

            period = response.get('class_period', {}) or {}
            row['class_period_start'] = period.get('start', '')
            row['class_period_end'] = period.get('end', '')

            # Causes of action (flatten list)
            causes = response.get('causes_of_action', [])
            for i, cause in enumerate(causes[:15], 1):
                if isinstance(cause, str):
                    row[f'cause_{i}'] = cause
                elif isinstance(cause, dict):
                    row[f'cause_{i}'] = cause.get('claim', '')

            # Calculate scores if ground truth available
            if ground_truth is not None:
                gt_row = ground_truth[ground_truth['case_id'] == case_id]
                if not gt_row.empty:
                    gt_dict = gt_row.iloc[0].to_dict()
                    actual = clean_ground_truth_row(gt_dict)
                    scores = evaluate_test1_case(response, actual, apply_cleaning=True)
                    row['plaintiffs_f1'] = round(scores.get('plaintiffs_f1', 0), 3)
                    row['defendants_f1'] = round(scores.get('defendants_f1', 0), 3)
                    row['ticker_match'] = round(scores.get('ticker', 0), 3)
                    row['class_period_score'] = round((scores.get('class_period_start', 0) + scores.get('class_period_end', 0)) / 2, 3)
                    row['causes_f1'] = round(scores.get('causes_f1', 0), 3)
                    row['composite_f1'] = round(scores.get('overall', 0), 3)

        elif test_name == 'test2':
            # Test 2: Summary + Claim Rulings fields
            row['summary'] = response.get('summary', '')

            # Handle both new format (claim_rulings) and old format (mtd_predictions)
            claim_rulings = response.get('claim_rulings', [])
            if not claim_rulings:
                claim_rulings = response.get('mtd_predictions', [])

            for i, ruling in enumerate(claim_rulings[:15], 1):
                if isinstance(ruling, dict):
                    row[f'claim_{i}'] = ruling.get('claim', '')
                    row[f'ruling_{i}'] = ruling.get('ruling', ruling.get('predicted_outcome', ''))
                    row[f'reasoning_{i}'] = ruling.get('reasoning', '')

            # Note: Full scoring is done in evaluate_results() with order texts
            # Excel export only includes raw LLM outputs for manual review

        # Add token usage
        usage = result.get('usage', {}) or {}
        row['prompt_tokens'] = usage.get('prompt_tokens', '')
        row['completion_tokens'] = usage.get('completion_tokens', '')

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)


def load_test_results(test_name: str, model_name: str) -> list:
    """Load test results from JSON file."""
    output_path = Path(RESULTS_DIR) / test_name / "raw_outputs" / f"{model_name}_outputs.json"
    if not output_path.exists():
        return []

    with open(output_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test with a single case and single model
    from data_loader import build_complaints_df

    print("Testing LLM inference...")
    print(f"Available models: {list(MODELS.keys())}")

    # Load one case
    df = build_complaints_df()
    sample = df.head(1)

    print(f"\nTest case: {sample['case_id'].iloc[0]}")
    print(f"Text length: {sample['text_length'].iloc[0]:,} chars")

    # Test with GPT-5.2
    print("\nRunning Test 1 extraction with gpt-5.2...")
    result = run_test1_extraction(
        sample['complaint_text'].iloc[0],
        "gpt-5.2"
    )

    if result['success']:
        print("Success!")
        print(json.dumps(result['response'], indent=2)[:1000] + "...")
    else:
        print(f"Error: {result['error']}")
