import json, os, re, time, threading
from typing import Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import litellm, pandas as pd
litellm.suppress_debug_info = True
litellm.set_verbose = False
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
from dotenv import load_dotenv
from tqdm import tqdm
from config import MODELS, DIRECT_SDK_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT, RATE_LIMIT_DELAY, MODEL_RATE_LIMITS, RESULTS_DIR, GROUND_TRUTH_TEST1_PATH, GROUND_TRUTH_TEST2_PATH
from prompts import format_test1_prompt, format_test2_prompt, format_test3_prompt
from data_loader import load_ground_truth_test1, load_ground_truth_test2
from evals import clean_ground_truth_row, evaluate_test1_case

load_dotenv()
_progress_lock = threading.Lock()
tqdm.monitor_interval = 0

def _parse_json(content: str) -> dict:
    m = re.match(r'^\s*(\{[\s\S]*\})\s*$', content)
    if m:
        result = json.loads(m.group(1))
        if isinstance(result, dict): return result
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    return json.loads(m.group(1)) if m else {"raw_response": content}

def call_llm(messages: list, model: str, model_name: str = None, temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS, timeout: int = LLM_TIMEOUT) -> dict:
    try:
        if model_name and model_name in DIRECT_SDK_MODELS and model_name == "gemini":
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            sys_prompt, user = next((m["content"] for m in messages if m["role"] == "system"), ""), next((m["content"] for m in messages if m["role"] == "user"), "")
            resp = client.models.generate_content(model=model, contents=f"{sys_prompt}\n\n{user}" if sys_prompt else user, config={"temperature": temperature, "max_output_tokens": max_tokens})
            usage = {"prompt_tokens": getattr(resp.usage_metadata, 'prompt_token_count', 0), "completion_tokens": getattr(resp.usage_metadata, 'candidates_token_count', 0), "total_tokens": getattr(resp.usage_metadata, 'total_token_count', 0)} if hasattr(resp, 'usage_metadata') and resp.usage_metadata else None
            return {"success": True, "response": _parse_json(resp.text), "usage": usage}
        resp = litellm.completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        return {"success": True, "response": _parse_json(resp.choices[0].message.content), "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens, "total_tokens": resp.usage.total_tokens}}
    except Exception as e: return {"success": False, "error": str(e), "usage": None}

def run_test1_extraction(text: str, model_name: str, model_id: Optional[str] = None) -> dict:
    model_id = model_id or MODELS.get(model_name)
    if not model_id: return {"success": False, "error": f"Unknown model: {model_name}"}
    return {"model": model_name, "model_id": model_id, "test": "test1", **call_llm(format_test1_prompt(text), model_id, model_name=model_name)}

def run_test2_summary(text: str, model_name: str, model_id: Optional[str] = None) -> dict:
    model_id = model_id or MODELS.get(model_name)
    if not model_id: return {"success": False, "error": f"Unknown model: {model_name}"}
    return {"model": model_name, "model_id": model_id, "test": "test2", **call_llm(format_test2_prompt(text), model_id, model_name=model_name)}

def run_test3_summary(text: str, model_name: str, model_id: Optional[str] = None) -> dict:
    model_id = model_id or MODELS.get(model_name)
    if not model_id: return {"success": False, "error": f"Unknown model: {model_name}"}
    return {"model": model_name, "model_id": model_id, "test": "test3", **call_llm(format_test3_prompt(text), model_id, model_name=model_name)}

def _run_model_batch(model: str, df: pd.DataFrame, run_func: Callable, pbar: tqdm, save: bool, test: str, delay: float = 0.0) -> tuple:
    if delay > 0: time.sleep(delay)
    results = []
    for _, row in df.iterrows():
        r = run_func(row['complaint_text'], model); r['case_id'] = row['case_id']; results.append(r)
        with _progress_lock: pbar.update(1); pbar.set_postfix_str(f"{model}: {row['case_id'][:20]}...")
        time.sleep(MODEL_RATE_LIMITS.get(model, RATE_LIMIT_DELAY))
    if save: save_test_results(results, test, model)
    return model, results

def _batch_run(df: pd.DataFrame, run_func: Callable, test: str, models: list = None, save: bool = True, parallel: bool = True, delays: dict = None) -> dict:
    models = models or list(MODELS.keys())
    pbar = tqdm(total=len(df) * len(models), desc=f"{test} Progress", unit="call", ncols=100)
    results = {}
    if parallel and len(models) > 1:
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            futures = {ex.submit(_run_model_batch, m, df, run_func, pbar, save, test, (delays or {}).get(m, 0.0)): m for m in models}
            for f in as_completed(futures): name, res = f.result(); results[name] = res
    else:
        for m in models: _, res = _run_model_batch(m, df, run_func, pbar, save, test); results[m] = res
    pbar.close()
    return results

def batch_run_test1(df: pd.DataFrame, models: list = None, save_results: bool = True, verbose: bool = True, parallel: bool = True) -> dict:
    return _batch_run(df, run_test1_extraction, "test1", models, save_results, parallel)

def batch_run_test2(df: pd.DataFrame, models: list = None, save_results: bool = True, verbose: bool = True, parallel: bool = True) -> dict:
    return _batch_run(df, run_test2_summary, "test2", models, save_results, parallel, {"claude-opus": 0.0, "gemini": 1.0, "perplexity": 2.0, "grok": 3.0, "gpt-5.2": 5.0})

def batch_run_test3(df: pd.DataFrame, models: list = None, save_results: bool = True, verbose: bool = True, parallel: bool = True) -> dict:
    return _batch_run(df, run_test3_summary, "test3", models, save_results, parallel, {"claude-opus": 0.0, "gemini": 1.0, "perplexity": 2.0, "grok": 3.0, "gpt-5.2": 5.0})

def save_test_results(results: list, test: str, model: str):
    out = Path(RESULTS_DIR) / test / "raw_outputs"; out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / f"{model}_outputs.json", 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    export_results_to_excel(results, test, out / f"{model}_outputs.xlsx")

def export_results_to_excel(results: list, test: str, path: Path):
    gt_path = Path(GROUND_TRUTH_TEST1_PATH if test == 'test1' else GROUND_TRUTH_TEST2_PATH)
    gt = (load_ground_truth_test1() if test == 'test1' else load_ground_truth_test2()) if gt_path.exists() else None
    rows = []
    for r in results:
        if not r.get('success'): rows.append({'case_id': r.get('case_id', ''), 'model': r.get('model', ''), 'success': False, 'error': r.get('error', '')}); continue
        resp, cid = r.get('response', {}), r.get('case_id', '')
        row = {'case_id': cid, 'model': r.get('model', ''), 'success': True}
        if test == 'test1':
            row['plaintiffs'], row['defendants'], row['ticker'] = '; '.join(resp.get('plaintiffs', [])), '; '.join(resp.get('defendants', [])), resp.get('ticker', '')
            p = resp.get('class_period', {}) or {}; row['class_period_start'], row['class_period_end'] = p.get('start', ''), p.get('end', '')
            for i, c in enumerate(resp.get('causes_of_action', [])[:15], 1): row[f'cause_{i}'] = c if isinstance(c, str) else c.get('claim', '')
            if gt is not None:
                gtr = gt[gt['case_id'] == cid]
                if not gtr.empty:
                    s = evaluate_test1_case(resp, clean_ground_truth_row(gtr.iloc[0].to_dict()))
                    row['plaintiffs_f1'], row['defendants_f1'], row['ticker_match'] = round(s['plaintiffs_f1'], 3), round(s['defendants_f1'], 3), round(s['ticker'], 3)
                    row['causes_f1'], row['composite_f1'] = round(s['causes_f1'], 3), round(s['overall'], 3)
        elif test == 'test2':
            row['summary'] = resp.get('summary', '')
            for i, ru in enumerate((resp.get('claim_rulings', []) or resp.get('mtd_predictions', []))[:15], 1):
                if isinstance(ru, dict): row[f'claim_{i}'], row[f'ruling_{i}'], row[f'reasoning_{i}'] = ru.get('claim', ''), ru.get('ruling', ru.get('predicted_outcome', '')), ru.get('reasoning', '')
        u = r.get('usage', {}) or {}; row['prompt_tokens'], row['completion_tokens'] = u.get('prompt_tokens', ''), u.get('completion_tokens', '')
        rows.append(row)
    pd.DataFrame(rows).to_excel(path, index=False)

def load_test_results(test: str, model: str) -> list:
    p = Path(RESULTS_DIR) / test / "raw_outputs" / f"{model}_outputs.json"
    return json.load(open(p, 'r', encoding='utf-8')) if p.exists() else []
