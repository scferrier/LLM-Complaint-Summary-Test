import json
import time
import re
import os
from typing import Optional
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv

from config import (
    MODELS,
    DIRECT_SDK_MODELS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    RATE_LIMIT_DELAY,
    RESULTS_DIR,
)
from prompts import format_test1_prompt, format_test2_prompt

# Load environment variables
load_dotenv()


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


def batch_run_test1(cases_df: pd.DataFrame, models: Optional[list] = None, save_results: bool = True,
    verbose: bool = True) -> dict:
 
    if models is None:
        models = list(MODELS.keys())

    all_results = {model: [] for model in models}

    total_calls = len(cases_df) * len(models)
    call_count = 0

    for model_name in models:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running Test 1 with {model_name}")
            print(f"{'='*50}")

        for idx, row in cases_df.iterrows():
            call_count += 1
            case_id = row['case_id']
            complaint_text = row['complaint_text']

            if verbose:
                print(f"  [{call_count}/{total_calls}] Processing {case_id}...", end=" ")

            result = run_test1_extraction(complaint_text, model_name)
            result['case_id'] = case_id

            all_results[model_name].append(result)

            if verbose:
                status = "OK" if result['success'] else f"ERROR: {result.get('error', 'Unknown')}"
                print(status)

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        # Save intermediate results
        if save_results:
            save_test_results(all_results[model_name], "test1", model_name)

    return all_results


def batch_run_test2(cases_df: pd.DataFrame, models: Optional[list] = None,
save_results: bool = True,  verbose: bool = True) -> dict:

    if models is None:
        models = list(MODELS.keys())

    all_results = {model: [] for model in models}

    total_calls = len(cases_df) * len(models)
    call_count = 0

    for model_name in models:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running Test 2 with {model_name}")
            print(f"{'='*50}")

        for idx, row in cases_df.iterrows():
            call_count += 1
            case_id = row['case_id']
            complaint_text = row['complaint_text']

            if verbose:
                print(f"  [{call_count}/{total_calls}] Processing {case_id}...", end=" ")

            result = run_test2_summary(complaint_text, model_name)
            result['case_id'] = case_id

            all_results[model_name].append(result)

            if verbose:
                status = "OK" if result['success'] else f"ERROR: {result.get('error', 'Unknown')}"
                print(status)

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        # Save intermediate results
        if save_results:
            save_test_results(all_results[model_name], "test2", model_name)

    return all_results


def save_test_results(results: list, test_name: str, model_name: str):
    """Save test results to JSON file."""
    output_dir = Path(RESULTS_DIR) / test_name / "raw_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_outputs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved results to {output_path}")


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
