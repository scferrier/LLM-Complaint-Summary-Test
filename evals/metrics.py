import re
import os
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("evaluate").setLevel(logging.ERROR)

from .cleaning import normalize_name, normalize_ruling

_evaluate_metrics, _nli_pipeline = {}, None

def _get_metric(name: str):
    global _evaluate_metrics
    if name not in _evaluate_metrics:
        import evaluate
        evaluate.logging.set_verbosity_error()
        _evaluate_metrics[name] = evaluate.load(name)
    return _evaluate_metrics[name]

def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline, logging as tf_logging
        tf_logging.set_verbosity_error()
        _nli_pipeline = pipeline('text-classification', model='facebook/bart-large-mnli', device=-1)
    return _nli_pipeline

def exact_match(predicted: str, actual: str) -> float:
    if predicted is None and actual is None: return 1.0
    if predicted is None or actual is None: return 0.0
    return 1.0 if str(predicted).lower().strip() == str(actual).lower().strip() else 0.0

def list_f1_score(predicted_list: list, actual_list: list) -> dict:
    if not predicted_list and not actual_list: return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_list or not actual_list: return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred, actual = set(normalize_name(p) for p in predicted_list if p), set(normalize_name(a) for a in actual_list if a)
    matches = len(pred & actual)
    prec, rec = (matches / len(pred) if pred else 0.0), (matches / len(actual) if actual else 0.0)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0}

def date_match(predicted_date: str, actual_date: str) -> float:
    if not predicted_date and not actual_date: return 1.0
    if not predicted_date or not actual_date: return 0.0
    def parse(d):
        d = str(d).strip()
        m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', d)
        if m: return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', d)
        return (int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None
    p, a = parse(predicted_date), parse(actual_date)
    if p is None or a is None: return exact_match(predicted_date, actual_date)
    if p == a: return 1.0
    return 0.8 if p[0] == a[0] and p[1] == a[1] else 0.0

def ruling_f1_score(predicted_rulings: list, actual_rulings: list) -> dict:
    if not predicted_rulings or not actual_rulings: return {"accuracy": 0.0, "macro_f1": 0.0, "correct": 0, "total": 0, "per_class": {}}
    pred_norm, actual_norm = [normalize_ruling(p) for p in predicted_rulings], [normalize_ruling(a) for a in actual_rulings]
    pairs = [(p, a) for p, a in zip(pred_norm[:min(len(pred_norm), len(actual_norm))], actual_norm) if p and a]
    if not pairs: return {"accuracy": 0.0, "macro_f1": 0.0, "correct": 0, "total": 0, "per_class": {}}
    pf, af = [p for p, a in pairs], [a for p, a in pairs]
    correct = sum(1 for p, a in zip(pf, af) if p == a)
    per_class, f1s = {}, []
    for cls in set(pf) | set(af):
        tp = sum(1 for p, a in zip(pf, af) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(pf, af) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(pf, af) if p != cls and a == cls)
        prec, rec = tp / (tp + fp) if (tp + fp) > 0 else 0.0, tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1}; f1s.append(f1)
    return {"accuracy": correct / len(pf), "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "correct": correct, "total": len(pf), "per_class": per_class}

def rouge_scores(prediction: str, reference: str) -> dict:
    if not prediction or not reference: return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    r = _get_metric("rouge").compute(predictions=[prediction], references=[reference], use_stemmer=True)
    return {"rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"]}

def bleu_score(prediction: str, reference: str) -> float:
    if not prediction or not reference: return 0.0
    return _get_metric("bleu").compute(predictions=[prediction], references=[[reference]])["bleu"]

def meteor_score(prediction: str, reference: str) -> float:
    if not prediction or not reference: return 0.0
    return _get_metric("meteor").compute(predictions=[prediction], references=[reference])["meteor"]

def bert_score(prediction: str, reference: str) -> dict:
    if not prediction or not reference: return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    r = _get_metric("bertscore").compute(predictions=[prediction], references=[reference], lang="en", rescale_with_baseline=True)
    return {"precision": r["precision"][0], "recall": r["recall"][0], "f1": r["f1"][0]}

def faithfulness_score(source_text: str, summary: str) -> float:
    if not source_text or not summary: return 0.0
    try:
        nli = _get_nli_pipeline()
        src = source_text[:3000] if len(source_text) > 3000 else source_text
        sents = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip() and len(s.strip()) > 10][:10]
        if not sents: return 0.0
        scores = []
        for s in sents:
            label = nli(f"{src[:1500]}</s></s>{s}", truncation=True)[0]['label'].lower()
            scores.append(1.0 if label == 'entailment' else 0.5 if label == 'neutral' else 0.0)
        return sum(scores) / len(scores)
    except: return 0.0
