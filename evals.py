"""
Evaluation metrics for LLM Complaint Summarization.

Test 1 Metrics (Extraction):
- Exact match for ticker, class period
- List F1 for plaintiffs, defendants
- Semantic similarity for cause of action descriptions

Test 2 Metrics (Summarization + MTD):
- ROUGE, BERTScore for summary quality
- SummaC for factual consistency
- Accuracy for MTD outcome predictions
"""

import json
import re
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

# Lazy load heavy dependencies
_evaluate_metrics = {}
_summac_model = None


def _get_metric(name: str):
    """Lazy load evaluate metrics."""
    global _evaluate_metrics
    if name not in _evaluate_metrics:
        import evaluate
        _evaluate_metrics[name] = evaluate.load(name)
    return _evaluate_metrics[name]


def _get_summac():
    """Lazy load SummaC model."""
    global _summac_model
    if _summac_model is None:
        from summac.model_summac import SummaCConv
        _summac_model = SummaCConv(
            models=["vitc"],
            bins='percentile',
            granularity="sentence",
            nli_labels="e",
            device="cpu",
            start_file="default",
            agg="mean"
        )
    return _summac_model


# =============================================================================
# Text Cleaning / Normalization
# =============================================================================

def clean_text_for_comparison(text: str) -> str:
    """
    Clean text for comparison by normalizing common variations.
    - Replace section symbols (§ and encoding variants) with 'Section'
    - Replace '&' with 'and'
    - Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace section symbol and common encoding issues
    text = text.replace('§', 'Section ')
    text = text.replace('�', 'Section ')  # Common encoding issue
    text = text.replace('\u00a7', 'Section ')  # Unicode section symbol

    # Replace & with 'and' (but not in company names like "S&P")
    # Only replace standalone & or & surrounded by spaces
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_list_field(text: str) -> list:
    """
    Split a text field into a list, handling multiple separator types.
    Prioritizes ';', then ',' for defendant/plaintiff lists.
    """
    if not text or not isinstance(text, str):
        return []

    # Clean the text first
    text = clean_text_for_comparison(text)

    # Try semicolon first (preferred separator)
    if ';' in text:
        items = [item.strip() for item in text.split(';')]
    # Try comma (common for name lists like defendants/plaintiffs)
    elif ',' in text:
        # Remove trailing ", and Name" pattern first
        text = re.sub(r',\s+and\s+', ', ', text, flags=re.IGNORECASE)
        # Split on comma followed by space
        items = [item.strip() for item in text.split(',')]
    # Try " and " as separator (for simple "A and B" patterns)
    elif ' and ' in text.lower():
        items = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
        items = [item.strip() for item in items]
    else:
        # Single item
        items = [text.strip()]

    # Filter out empty items and duplicates
    seen = set()
    result = []
    for item in items:
        # Normalize for dedup check
        normalized = normalize_name(item)
        if item.strip() and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item.strip())
    return result


def normalize_cause_of_action(cause: str) -> str:
    """
    Normalize a cause of action string for comparison.
    Removes common prefixes and standardizes formatting.
    """
    if not cause:
        return ""

    cause = clean_text_for_comparison(cause)

    # Remove common prefixes
    prefixes_to_remove = [
        r'^violations?\s+of\s+',
        r'^claims?\s+under\s+',
        r'^claims?\s+for\s+',
    ]
    for prefix in prefixes_to_remove:
        cause = re.sub(prefix, '', cause, flags=re.IGNORECASE)

    # Remove trailing duplicates/garbage (like "10b-5" repeated)
    # Clean up any newlines or extra text
    cause = cause.split('\n')[0].strip()

    return cause.strip()


def clean_ground_truth_row(gt_row: dict) -> dict:
    """
    Clean a ground truth row for comparison.
    Returns a cleaned dict with standardized formats.
    """
    import pandas as pd

    cleaned = {
        'plaintiffs': [],
        'defendants': [],
        'ticker': None,
        'class_period': {'start': None, 'end': None},
        'causes_of_action': []
    }

    # Clean plaintiffs
    if 'plaintiffs' in gt_row and pd.notna(gt_row.get('plaintiffs')):
        cleaned['plaintiffs'] = split_list_field(str(gt_row['plaintiffs']))

    # Clean defendants
    if 'defendants' in gt_row and pd.notna(gt_row.get('defendants')):
        cleaned['defendants'] = split_list_field(str(gt_row['defendants']))

    # Clean ticker (uppercase, strip whitespace)
    if 'ticker' in gt_row and pd.notna(gt_row.get('ticker')):
        cleaned['ticker'] = str(gt_row['ticker']).strip().upper()

    # Class period dates
    if 'class_period_start' in gt_row and pd.notna(gt_row.get('class_period_start')):
        cleaned['class_period']['start'] = str(gt_row['class_period_start'])[:10]
    if 'class_period_end' in gt_row and pd.notna(gt_row.get('class_period_end')):
        cleaned['class_period']['end'] = str(gt_row['class_period_end'])[:10]

    # Clean causes of action
    for i in range(1, 16):
        col = f'cause_{i}'
        if col in gt_row and pd.notna(gt_row.get(col)):
            cause = normalize_cause_of_action(str(gt_row[col]))
            if cause:
                cleaned['causes_of_action'].append(cause)

    return cleaned


def clean_llm_response(response: dict) -> dict:
    """
    Clean LLM response for comparison.
    Applies same normalization as ground truth.
    """
    cleaned = {
        'plaintiffs': [],
        'defendants': [],
        'ticker': None,
        'class_period': {'start': None, 'end': None},
        'causes_of_action': []
    }

    # Clean plaintiffs
    plaintiffs = response.get('plaintiffs', [])
    if isinstance(plaintiffs, list):
        cleaned['plaintiffs'] = [clean_text_for_comparison(p) for p in plaintiffs if p]

    # Clean defendants
    defendants = response.get('defendants', [])
    if isinstance(defendants, list):
        cleaned['defendants'] = [clean_text_for_comparison(d) for d in defendants if d]

    # Clean ticker
    ticker = response.get('ticker')
    if ticker:
        cleaned['ticker'] = str(ticker).strip().upper()

    # Class period
    period = response.get('class_period', {}) or {}
    if period.get('start'):
        cleaned['class_period']['start'] = str(period['start'])[:10]
    if period.get('end'):
        cleaned['class_period']['end'] = str(period['end'])[:10]

    # Clean causes of action
    causes = response.get('causes_of_action', [])
    if isinstance(causes, list):
        for c in causes:
            if isinstance(c, str):
                cleaned['causes_of_action'].append(normalize_cause_of_action(c))
            elif isinstance(c, dict):
                claim = c.get('claim', '')
                cleaned['causes_of_action'].append(normalize_cause_of_action(claim))

    return cleaned


# =============================================================================
# Test 1: Extraction Metrics
# =============================================================================

def exact_match(predicted: str, actual: str) -> float:
    """
    Check if predicted value exactly matches actual (case-insensitive).
    Returns 1.0 for match, 0.0 for no match.
    """
    if predicted is None and actual is None:
        return 1.0
    if predicted is None or actual is None:
        return 0.0
    return 1.0 if str(predicted).lower().strip() == str(actual).lower().strip() else 0.0


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, remove common prefixes/suffixes)."""
    if not name:
        return ""
    name = name.lower().strip()

    # Remove parenthetical acronyms like (UANPF), (SHEPP), (the "Company")
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)

    # Remove common legal suffixes
    for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", corp.", " corp.", ", co.", " co."]:
        name = name.replace(suffix, "")

    # Normalize special characters
    name = name.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    name = name.replace("�", "'")  # Common encoding issue

    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def list_f1_score(predicted_list: list, actual_list: list) -> dict:
    """
    Calculate precision, recall, and F1 for list comparison.
    Uses normalized string matching.

    Returns:
        dict with 'precision', 'recall', 'f1'
    """
    if not predicted_list and not actual_list:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not actual_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Normalize all names
    pred_normalized = set(normalize_name(p) for p in predicted_list if p)
    actual_normalized = set(normalize_name(a) for a in actual_list if a)

    # Calculate matches
    matches = len(pred_normalized & actual_normalized)

    precision = matches / len(pred_normalized) if pred_normalized else 0.0
    recall = matches / len(actual_normalized) if actual_normalized else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def date_match(predicted_date: str, actual_date: str) -> float:
    """
    Check if dates match (flexible parsing).
    Returns 1.0 for match, partial score for close matches, 0.0 for no match.
    """
    if not predicted_date and not actual_date:
        return 1.0
    if not predicted_date or not actual_date:
        return 0.0

    # Simple normalization - extract year, month, day
    def parse_date(d):
        # Try to extract YYYY-MM-DD or similar
        d = str(d).strip()
        match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', d)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        # Try MM/DD/YYYY
        match = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', d)
        if match:
            return (int(match.group(3)), int(match.group(1)), int(match.group(2)))
        return None

    pred_parsed = parse_date(predicted_date)
    actual_parsed = parse_date(actual_date)

    if pred_parsed is None or actual_parsed is None:
        # Fallback to string match
        return exact_match(predicted_date, actual_date)

    if pred_parsed == actual_parsed:
        return 1.0

    # Partial credit for same year and month
    if pred_parsed[0] == actual_parsed[0] and pred_parsed[1] == actual_parsed[1]:
        return 0.8

    return 0.0


def evaluate_test1_case(predicted: dict, actual: dict, apply_cleaning: bool = True) -> dict:
    """
    Evaluate Test 1 extraction for a single case.

    Args:
        predicted: LLM output dict with plaintiffs, defendants, ticker, class_period, causes_of_action
        actual: Ground truth dict with same structure
        apply_cleaning: Whether to clean/normalize both inputs before comparing

    Returns:
        dict with scores for each field
    """
    # Apply cleaning if requested
    if apply_cleaning:
        predicted = clean_llm_response(predicted)
        # actual should already be cleaned by clean_ground_truth_row, but clean again to be safe
        if not all(k in actual for k in ['plaintiffs', 'defendants', 'ticker', 'class_period', 'causes_of_action']):
            actual = clean_ground_truth_row(actual)

    scores = {}

    # Ticker (exact match)
    scores['ticker'] = exact_match(
        predicted.get('ticker'),
        actual.get('ticker')
    )

    # Plaintiffs (list F1)
    plaintiff_scores = list_f1_score(
        predicted.get('plaintiffs', []),
        actual.get('plaintiffs', [])
    )
    scores['plaintiffs_precision'] = plaintiff_scores['precision']
    scores['plaintiffs_recall'] = plaintiff_scores['recall']
    scores['plaintiffs_f1'] = plaintiff_scores['f1']

    # Defendants (list F1)
    defendant_scores = list_f1_score(
        predicted.get('defendants', []),
        actual.get('defendants', [])
    )
    scores['defendants_precision'] = defendant_scores['precision']
    scores['defendants_recall'] = defendant_scores['recall']
    scores['defendants_f1'] = defendant_scores['f1']

    # Class period (date match)
    pred_period = predicted.get('class_period', {}) or {}
    actual_period = actual.get('class_period', {}) or {}

    scores['class_period_start'] = date_match(
        pred_period.get('start'),
        actual_period.get('start')
    )
    scores['class_period_end'] = date_match(
        pred_period.get('end'),
        actual_period.get('end')
    )

    # Causes of action (compare claim names)
    pred_causes = predicted.get('causes_of_action', []) or []
    actual_causes = actual.get('causes_of_action', []) or []

    # Handle both list of strings and list of dicts (for backwards compatibility)
    def extract_claims(causes):
        claims = []
        for c in causes:
            if isinstance(c, str):
                claims.append(c)
            elif isinstance(c, dict):
                claims.append(c.get('claim', ''))
        return claims

    pred_claims = extract_claims(pred_causes)
    actual_claims = extract_claims(actual_causes)

    cause_scores = list_f1_score(pred_claims, actual_claims)
    scores['causes_precision'] = cause_scores['precision']
    scores['causes_recall'] = cause_scores['recall']
    scores['causes_f1'] = cause_scores['f1']

    # Overall score (average of key metrics)
    key_scores = [
        scores['ticker'],
        scores['plaintiffs_f1'],
        scores['defendants_f1'],
        (scores['class_period_start'] + scores['class_period_end']) / 2,
        scores['causes_f1']
    ]
    scores['overall'] = sum(key_scores) / len(key_scores)

    return scores


# =============================================================================
# Test 2: Summarization Metrics
# =============================================================================

def rouge_scores(prediction: str, reference: str) -> dict:
    """
    Calculate ROUGE scores.

    Returns:
        dict with rouge1, rouge2, rougeL scores
    """
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge = _get_metric("rouge")
    result = rouge.compute(
        predictions=[prediction],
        references=[reference],
        use_stemmer=True
    )
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"]
    }


def bleu_score(prediction: str, reference: str) -> float:
    """
    Calculate BLEU score.

    Returns:
        float BLEU score between 0 and 1
    """
    if not prediction or not reference:
        return 0.0

    bleu = _get_metric("bleu")
    # Hugging Face evaluate expects strings (handles tokenization internally)
    result = bleu.compute(
        predictions=[prediction],
        references=[[reference]]
    )
    return result["bleu"]


def meteor_score(prediction: str, reference: str) -> float:
    """
    Calculate METEOR score.

    Returns:
        float METEOR score between 0 and 1
    """
    if not prediction or not reference:
        return 0.0

    meteor = _get_metric("meteor")
    result = meteor.compute(
        predictions=[prediction],
        references=[reference]
    )
    return result["meteor"]


def bert_score(prediction: str, reference: str) -> dict:
    """
    Calculate BERTScore for semantic similarity.

    Returns:
        dict with precision, recall, f1
    """
    if not prediction or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    bertscore = _get_metric("bertscore")
    result = bertscore.compute(
        predictions=[prediction],
        references=[reference],
        lang="en",
        rescale_with_baseline=True
    )
    return {
        "precision": result["precision"][0],
        "recall": result["recall"][0],
        "f1": result["f1"][0]
    }


def summac_score(source_text: str, summary: str) -> float:
    """
    Calculate SummaC consistency score.
    Measures if the summary is factually consistent with the source.

    Returns:
        float score between 0 and 1
    """
    if not source_text or not summary:
        return 0.0

    summac = _get_summac()
    result = summac.score([source_text], [summary])
    return result["scores"][0]


def normalize_ruling(outcome: str) -> str:
    """
    Normalize a ruling/outcome string to standard form.

    Maps various phrasings to: 'dismissed', 'sustained', 'dismissed_in_part'
    """
    if not outcome:
        return None
    o = str(outcome).lower().strip()

    # Map to standard terms
    if 'dismiss' in o and 'part' in o:
        return 'dismissed_in_part'
    elif 'grant' in o and 'part' in o:
        return 'dismissed_in_part'
    elif 'dismiss' in o:
        return 'dismissed'
    elif 'grant' in o:
        return 'dismissed'  # granted MTD = dismissed
    elif 'sustain' in o:
        return 'sustained'
    elif 'denied' in o or 'deny' in o:
        return 'sustained'  # denied MTD = sustained
    elif 'survive' in o:
        return 'sustained'
    return o


def ruling_f1_score(predicted_rulings: list, actual_rulings: list) -> dict:
    """
    Calculate F1 score for claim ruling predictions.

    Computes per-class precision, recall, F1 and macro-averaged F1.

    Args:
        predicted_rulings: List of LLM predictions ('dismissed', 'sustained', 'dismissed_in_part')
        actual_rulings: List of ground truth outcomes

    Returns:
        dict with per-class metrics and macro F1
    """
    if not predicted_rulings or not actual_rulings:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "correct": 0,
            "total": 0,
            "per_class": {}
        }

    # Normalize all rulings
    pred_norm = [normalize_ruling(p) for p in predicted_rulings]
    actual_norm = [normalize_ruling(a) for a in actual_rulings]

    # Align lists (match by position)
    min_len = min(len(pred_norm), len(actual_norm))
    pred_norm = pred_norm[:min_len]
    actual_norm = actual_norm[:min_len]

    # Filter out None values (missing predictions or ground truth)
    pairs = [(p, a) for p, a in zip(pred_norm, actual_norm) if p is not None and a is not None]
    if not pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "correct": 0, "total": 0, "per_class": {}}

    pred_filtered = [p for p, a in pairs]
    actual_filtered = [a for p, a in pairs]

    # Calculate accuracy
    correct = sum(1 for p, a in zip(pred_filtered, actual_filtered) if p == a)
    accuracy = correct / len(pred_filtered)

    # Calculate per-class precision, recall, F1
    classes = set(pred_filtered) | set(actual_filtered)
    per_class = {}
    f1_scores = []

    for cls in classes:
        tp = sum(1 for p, a in zip(pred_filtered, actual_filtered) if p == cls and a == cls)
        fp = sum(1 for p, a in zip(pred_filtered, actual_filtered) if p == cls and a != cls)
        fn = sum(1 for p, a in zip(pred_filtered, actual_filtered) if p != cls and a == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    # Macro F1 (unweighted average across classes)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": correct,
        "total": len(pred_filtered),
        "per_class": per_class
    }


def mtd_accuracy(predicted_outcomes: list, actual_outcomes: list) -> dict:
    """
    DEPRECATED: Use ruling_f1_score instead for F1 metrics.

    Calculate accuracy for MTD outcome predictions.

    Args:
        predicted_outcomes: List of predictions ('granted', 'denied', 'granted-in-part')
        actual_outcomes: List of actual outcomes

    Returns:
        dict with accuracy and per-class metrics
    """
    if not predicted_outcomes or not actual_outcomes:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    pred_norm = [normalize_ruling(p) for p in predicted_outcomes]
    actual_norm = [normalize_ruling(a) for a in actual_outcomes]

    # Count matches
    min_len = min(len(pred_norm), len(actual_norm))
    correct = sum(1 for p, a in zip(pred_norm[:min_len], actual_norm[:min_len]) if p == a)

    return {
        "accuracy": correct / min_len if min_len > 0 else 0.0,
        "correct": correct,
        "total": min_len
    }


def combine_llm_output_to_text(predicted: dict) -> str:
    """
    Combine LLM JSON output (summary + claim rulings) into a single text
    for comparison against order text.

    Args:
        predicted: LLM output with 'summary' and 'claim_rulings'

    Returns:
        Combined text string
    """
    parts = []

    # Add summary
    summary = predicted.get('summary', '')
    if summary:
        parts.append(summary)

    # Add claim rulings
    claim_rulings = predicted.get('claim_rulings', [])
    if not claim_rulings:
        # Fallback for old format with mtd_predictions
        claim_rulings = predicted.get('mtd_predictions', [])

    if isinstance(claim_rulings, list):
        for ruling in claim_rulings:
            if isinstance(ruling, dict):
                claim = ruling.get('claim', '')
                outcome = ruling.get('ruling', ruling.get('predicted_outcome', ''))
                reasoning = ruling.get('reasoning', '')

                if claim or reasoning:
                    ruling_text = f"{claim}: {outcome}. {reasoning}"
                    parts.append(ruling_text)

    return "\n\n".join(parts)


def evaluate_test2_case(
    predicted: dict,
    ground_truth_summary: str,
    source_text: Optional[str] = None,
    ground_truth_rulings: Optional[list] = None,
    compute_factual: bool = True
) -> dict:
    """
    Evaluate Test 2 summarization and claim rulings for a single case.

    Compares LLM summary against ground truth summary using ROUGE, BLEU, METEOR,
    BERTScore. Uses SummaC to check factual consistency against source complaint.

    Args:
        predicted: LLM output with 'summary' and 'claim_rulings'
        ground_truth_summary: Reference summary to compare against for text metrics
        source_text: Original complaint text (for SummaC factual consistency)
        ground_truth_rulings: List of actual ruling outcomes from Excel (for F1)
        compute_factual: Whether to compute SummaC (slow)

    Returns:
        dict with scores for all metrics
    """
    scores = {}

    # Get LLM summary for comparison
    pred_summary = predicted.get('summary', '')

    if not pred_summary or not ground_truth_summary:
        return {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'bleu': 0.0, 'meteor': 0.0,
            'bertscore_f1': 0.0,
            'summac': 0.0,
            'ruling_f1': 0.0, 'ruling_accuracy': 0.0,
            'overall': 0.0
        }

    # ROUGE scores (compare LLM summary vs ground truth summary)
    rouge = rouge_scores(pred_summary, ground_truth_summary)
    scores['rouge1'] = rouge['rouge1']
    scores['rouge2'] = rouge['rouge2']
    scores['rougeL'] = rouge['rougeL']

    # BLEU score (compare LLM summary vs ground truth summary)
    scores['bleu'] = bleu_score(pred_summary, ground_truth_summary)

    # METEOR score (compare LLM summary vs ground truth summary)
    scores['meteor'] = meteor_score(pred_summary, ground_truth_summary)

    # BERTScore (compare LLM summary vs ground truth summary)
    bert = bert_score(pred_summary, ground_truth_summary)
    scores['bertscore_f1'] = bert['f1']

    # SummaC: Is the LLM summary factually consistent with the source complaint?
    if compute_factual and source_text:
        scores['summac'] = summac_score(source_text, pred_summary)
    else:
        scores['summac'] = 0.0

    # Ruling F1 (compare predicted rulings against ground truth)
    if ground_truth_rulings:
        # Extract predicted rulings from LLM output
        claim_rulings = predicted.get('claim_rulings', [])
        if not claim_rulings:
            # Fallback for old format
            claim_rulings = predicted.get('mtd_predictions', [])

        predicted_rulings = []
        for ruling in claim_rulings:
            if isinstance(ruling, dict):
                outcome = ruling.get('ruling', ruling.get('predicted_outcome', ''))
                predicted_rulings.append(outcome)

        # Calculate F1
        ruling_metrics = ruling_f1_score(predicted_rulings, ground_truth_rulings)
        scores['ruling_f1'] = ruling_metrics['macro_f1']
        scores['ruling_accuracy'] = ruling_metrics['accuracy']
        scores['ruling_correct'] = ruling_metrics['correct']
        scores['ruling_total'] = ruling_metrics['total']
    else:
        scores['ruling_f1'] = 0.0
        scores['ruling_accuracy'] = 0.0

    # Overall score (weighted average of all metrics)
    # Surface metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR (5)
    # Semantic metrics: BERTScore (1)
    # Factual metrics: SummaC (1)
    # Ruling metrics: F1 (1)
    surface_score = (
        scores['rouge1'] + scores['rouge2'] + scores['rougeL'] +
        scores['bleu'] + scores['meteor']
    ) / 5
    semantic_score = scores['bertscore_f1']
    factual_score = scores['summac'] if compute_factual else 0.0
    ruling_score = scores['ruling_f1']

    # Weight: 30% surface, 20% semantic, 20% factual, 30% ruling
    if compute_factual and ground_truth_rulings:
        scores['overall'] = 0.3 * surface_score + 0.2 * semantic_score + 0.2 * factual_score + 0.3 * ruling_score
    elif ground_truth_rulings:
        # Without factual: 40% surface, 30% semantic, 30% ruling
        scores['overall'] = 0.4 * surface_score + 0.3 * semantic_score + 0.3 * ruling_score
    elif compute_factual:
        # Without rulings: 40% surface, 30% semantic, 30% factual
        scores['overall'] = 0.4 * surface_score + 0.3 * semantic_score + 0.3 * factual_score
    else:
        # Only surface and semantic: 60% surface, 40% semantic
        scores['overall'] = 0.6 * surface_score + 0.4 * semantic_score

    return scores


def evaluate_test2_case_legacy(
    predicted: dict,
    actual: dict,
    source_text: Optional[str] = None,
    compute_summac: bool = False
) -> dict:
    """
    DEPRECATED: Legacy Test 2 evaluation using Excel ground truth.
    Use evaluate_test2_case() with order text instead.

    Evaluate Test 2 summarization and MTD prediction for a single case.

    Args:
        predicted: LLM output with 'summary' and 'mtd_predictions'
        actual: Ground truth with 'summary_reference' and MTD outcomes
        source_text: Original complaint text (for SummaC)
        compute_summac: Whether to compute SummaC (slow)

    Returns:
        dict with scores for summary quality and MTD accuracy
    """
    scores = {}

    # Summary quality
    pred_summary = predicted.get('summary', '')
    ref_summary = actual.get('summary_reference', '')

    if pred_summary and ref_summary:
        # ROUGE
        rouge = rouge_scores(pred_summary, ref_summary)
        scores['rouge1'] = rouge['rouge1']
        scores['rouge2'] = rouge['rouge2']
        scores['rougeL'] = rouge['rougeL']

        # BERTScore
        bert = bert_score(pred_summary, ref_summary)
        scores['bertscore_f1'] = bert['f1']

        # SummaC (optional, slow)
        if compute_summac and source_text:
            scores['summac'] = summac_score(source_text, pred_summary)
    else:
        scores['rouge1'] = 0.0
        scores['rouge2'] = 0.0
        scores['rougeL'] = 0.0
        scores['bertscore_f1'] = 0.0

    # MTD predictions
    pred_mtd = predicted.get('mtd_predictions', [])
    if isinstance(pred_mtd, list):
        pred_outcomes = [p.get('predicted_outcome') for p in pred_mtd if isinstance(p, dict)]
    else:
        pred_outcomes = []

    # Get actual outcomes from ground truth
    actual_outcomes = []
    for i in range(1, 5):  # cause_1 through cause_4
        outcome = actual.get(f'cause_{i}_mtd_outcome')
        if outcome:
            actual_outcomes.append(outcome)

    mtd = mtd_accuracy(pred_outcomes, actual_outcomes)
    scores['mtd_accuracy'] = mtd['accuracy']
    scores['mtd_correct'] = mtd['correct']
    scores['mtd_total'] = mtd['total']

    # Overall score
    summary_score = (scores['rouge1'] + scores['rouge2'] + scores['rougeL'] + scores['bertscore_f1']) / 4
    scores['overall'] = (summary_score + scores['mtd_accuracy']) / 2

    return scores


# =============================================================================
# Batch Evaluation
# =============================================================================

def evaluate_all_test1(
    results: dict,
    ground_truth_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate all Test 1 results for all models.

    Args:
        results: dict mapping model_name -> list of result dicts
        ground_truth_df: DataFrame with ground truth (from ground_truth_test1.xlsx)

    Returns:
        DataFrame with scores for each case and model
    """
    all_scores = []

    for model_name, model_results in results.items():
        for result in model_results:
            if not result.get('success'):
                continue

            case_id = result['case_id']
            predicted = result.get('response', {})

            # Get ground truth for this case
            gt_row = ground_truth_df[ground_truth_df['case_id'] == case_id]
            if gt_row.empty:
                continue

            # Convert ground truth row to cleaned dict
            gt_dict = gt_row.iloc[0].to_dict()
            actual = clean_ground_truth_row(gt_dict)

            # Evaluate (cleaning is already applied to actual, will be applied to predicted)
            scores = evaluate_test1_case(predicted, actual, apply_cleaning=True)
            scores['model'] = model_name
            scores['case_id'] = case_id
            all_scores.append(scores)

    return pd.DataFrame(all_scores)


def evaluate_all_test2(
    results: dict,
    ground_truth_df: pd.DataFrame,
    complaints_df: Optional[pd.DataFrame] = None,
    rulings_df: Optional[pd.DataFrame] = None,
    compute_factual: bool = True
) -> pd.DataFrame:
    """
    Evaluate all Test 2 results for all models.

    Compares LLM summaries against ground truth summaries using ROUGE, BLEU,
    METEOR, BERTScore. Uses SummaC to check factual consistency against
    source complaint text.

    Args:
        results: dict mapping model_name -> list of result dicts
        ground_truth_df: DataFrame with ground truth summaries
                         Must have 'case_id' and 'summary' columns
        complaints_df: DataFrame with complaint texts (for SummaC factual consistency)
                       Must have 'case_id' and 'complaint_text' columns
        rulings_df: DataFrame with ground truth rulings (from Excel)
                    Should have 'case_id' and 'cause_X_mtd_outcome' columns
        compute_factual: Whether to compute SummaC scores (slow)

    Returns:
        DataFrame with scores for each case and model
    """
    all_scores = []

    for model_name, model_results in results.items():
        for result in model_results:
            if not result.get('success'):
                continue

            case_id = result['case_id']
            predicted = result.get('response', {})

            # Get ground truth summary for this case
            gt_row = ground_truth_df[ground_truth_df['case_id'] == case_id]
            if gt_row.empty:
                # Try matching with normalized case_id
                normalized_id = case_id.replace('-', '_').replace(' ', '_')
                gt_row = ground_truth_df[
                    ground_truth_df['case_id'].str.replace('-', '_').str.replace(' ', '_') == normalized_id
                ]

            if gt_row.empty:
                print(f"  Warning: No ground truth found for case {case_id}")
                continue

            ground_truth_summary = gt_row.iloc[0].get('summary', '')
            if not ground_truth_summary:
                print(f"  Warning: Empty ground truth summary for case {case_id}")
                continue

            # Get source complaint text for factual consistency metrics
            source_text = None
            if compute_factual and complaints_df is not None:
                complaint_row = complaints_df[complaints_df['case_id'] == case_id]
                if complaint_row.empty:
                    # Try normalized matching
                    normalized_id = case_id.replace('-', '_').replace(' ', '_')
                    complaint_row = complaints_df[
                        complaints_df['case_id'].str.replace('-', '_').str.replace(' ', '_') == normalized_id
                    ]
                if not complaint_row.empty:
                    source_text = complaint_row['complaint_text'].iloc[0]

            # Get ground truth rulings from Excel if available
            ground_truth_rulings = None
            if rulings_df is not None:
                ruling_row = rulings_df[rulings_df['case_id'] == case_id]
                if ruling_row.empty:
                    normalized_id = case_id.replace('-', '_').replace(' ', '_')
                    ruling_row = rulings_df[
                        rulings_df['case_id'].str.replace('-', '_').str.replace(' ', '_') == normalized_id
                    ]
                if not ruling_row.empty:
                    # Extract ruling columns (cause_1_mtd_outcome, cause_2_mtd_outcome, etc.)
                    row_data = ruling_row.iloc[0]
                    ground_truth_rulings = []
                    for i in range(1, 16):  # Up to 15 rulings
                        col = f'cause_{i}_mtd_outcome'
                        if col in row_data.index and pd.notna(row_data[col]):
                            ground_truth_rulings.append(str(row_data[col]))

            # Evaluate using ground truth summary as reference
            scores = evaluate_test2_case(
                predicted=predicted,
                ground_truth_summary=ground_truth_summary,
                source_text=source_text,
                ground_truth_rulings=ground_truth_rulings,
                compute_factual=compute_factual
            )
            scores['model'] = model_name
            scores['case_id'] = case_id
            all_scores.append(scores)

    return pd.DataFrame(all_scores)


def generate_summary_report(
    test1_scores: pd.DataFrame,
    test2_scores: pd.DataFrame,
    output_path: str = "results/summary_report.md"
) -> str:
    """
    Generate a markdown summary report comparing models.

    Returns:
        Markdown string
    """
    import json

    lines = ["# LLM Evaluation Summary Report\n"]

    # Test 1 Summary
    lines.append("## Test 1: Structured Data Extraction\n")

    if not test1_scores.empty:
        # Load raw outputs to get actual success/failure counts
        raw_outputs_dir = Path("results/test1/raw_outputs")

        # Calculate success rate and averages per model
        model_stats = []
        for model in test1_scores['model'].unique():
            model_data = test1_scores[test1_scores['model'] == model]
            successful = len(model_data[model_data['overall'] > 0])

            # Try to load raw outputs to get total attempted
            total_cases = successful  # Default to successful count
            raw_file = raw_outputs_dir / f"{model}_outputs.json"
            if raw_file.exists():
                try:
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        raw_results = json.load(f)
                    total_cases = len(raw_results)
                except:
                    pass

            success_rate = successful / total_cases if total_cases > 0 else 0

            # Calculate averages only for successful cases
            successful_data = model_data[model_data['overall'] > 0]
            if len(successful_data) > 0:
                model_stats.append({
                    'model': model,
                    'overall': successful_data['overall'].mean(),
                    'ticker': successful_data['ticker'].mean(),
                    'plaintiffs_f1': successful_data['plaintiffs_f1'].mean(),
                    'defendants_f1': successful_data['defendants_f1'].mean(),
                    'causes_f1': successful_data['causes_f1'].mean(),
                    'success_rate': success_rate,
                    'successful': successful,
                    'total': total_cases
                })

        if model_stats:
            # Sort by overall score descending
            model_stats = sorted(model_stats, key=lambda x: x['overall'], reverse=True)

            lines.append("| Model | Overall | Ticker | Plaintiffs | Defendants | Causes | Success Rate |")
            lines.append("|-------|---------|--------|------------|------------|--------|--------------|")
            for stat in model_stats:
                success_str = f"{stat['successful']}/{stat['total']} ({stat['success_rate']*100:.0f}%)"
                lines.append(f"| {stat['model']} | {stat['overall']:.3f} | {stat['ticker']:.3f} | {stat['plaintiffs_f1']:.3f} | {stat['defendants_f1']:.3f} | {stat['causes_f1']:.3f} | {success_str} |")

            lines.append("")
            lines.append("*Scores calculated on successful cases only*\n")
    else:
        lines.append("No Test 1 results available.\n")

    lines.append("")

    # Test 2 Summary
    lines.append("## Test 2: Judicial Summary + Claim Rulings vs Order Text\n")

    if not test2_scores.empty:
        raw_outputs_dir = Path("results/test2/raw_outputs")

        model_stats = []
        for model in test2_scores['model'].unique():
            model_data = test2_scores[test2_scores['model'] == model]
            successful = len(model_data[model_data['overall'] > 0])

            # Try to load raw outputs to get total attempted
            total_cases = successful
            raw_file = raw_outputs_dir / f"{model}_outputs.json"
            if raw_file.exists():
                try:
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        raw_results = json.load(f)
                    total_cases = len(raw_results)
                except:
                    pass

            success_rate = successful / total_cases if total_cases > 0 else 0

            successful_data = model_data[model_data['overall'] > 0]
            if len(successful_data) > 0:
                stat = {
                    'model': model,
                    'overall': successful_data['overall'].mean(),
                    'rouge1': successful_data['rouge1'].mean(),
                    'rougeL': successful_data['rougeL'].mean(),
                    'bleu': successful_data['bleu'].mean() if 'bleu' in successful_data.columns else 0.0,
                    'meteor': successful_data['meteor'].mean() if 'meteor' in successful_data.columns else 0.0,
                    'bertscore_f1': successful_data['bertscore_f1'].mean(),
                    'summac': successful_data['summac'].mean() if 'summac' in successful_data.columns else 0.0,
                    'ruling_f1': successful_data['ruling_f1'].mean() if 'ruling_f1' in successful_data.columns else 0.0,
                    'success_rate': success_rate,
                    'successful': successful,
                    'total': total_cases
                }
                model_stats.append(stat)

        if model_stats:
            model_stats = sorted(model_stats, key=lambda x: x['overall'], reverse=True)

            lines.append("### Surface & Semantic Metrics\n")
            lines.append("| Model | Overall | ROUGE-1 | ROUGE-L | BLEU | METEOR | BERTScore |")
            lines.append("|-------|---------|---------|---------|------|--------|-----------|")
            for stat in model_stats:
                lines.append(f"| {stat['model']} | {stat['overall']:.3f} | {stat['rouge1']:.3f} | {stat['rougeL']:.3f} | {stat['bleu']:.3f} | {stat['meteor']:.3f} | {stat['bertscore_f1']:.3f} |")

            lines.append("")
            lines.append("### Factual Consistency & Ruling Metrics\n")
            lines.append("| Model | SummaC | Ruling F1 | Success Rate |")
            lines.append("|-------|--------|-----------|--------------|")
            for stat in model_stats:
                success_str = f"{stat['successful']}/{stat['total']} ({stat['success_rate']*100:.0f}%)"
                ruling_f1 = stat.get('ruling_f1', 0.0)
                lines.append(f"| {stat['model']} | {stat['summac']:.3f} | {ruling_f1:.3f} | {success_str} |")

            lines.append("")
            lines.append("*Scores calculated on successful cases only*\n")
            lines.append("*Overall = 30% surface + 20% semantic + 20% factual + 30% ruling F1*\n")
    else:
        lines.append("No Test 2 results available.\n")

    report = "\n".join(lines)

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


def generate_macro_f1_table(
    test1_scores: pd.DataFrame,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Generate a Macro F1 results table per model and save to results folder.

    Macro F1 is the unweighted mean of F1 scores across extraction fields:
    - plaintiffs_f1
    - defendants_f1
    - causes_f1

    Also includes individual field scores and overall metrics.

    Args:
        test1_scores: DataFrame with per-case scores from evaluate_all_test1
        output_dir: Directory to save results

    Returns:
        DataFrame with Macro F1 and detailed scores per model
    """
    if test1_scores.empty:
        print("No Test 1 scores available for Macro F1 calculation.")
        return pd.DataFrame()

    # F1 fields for Macro F1 calculation
    f1_fields = ['plaintiffs_f1', 'defendants_f1', 'causes_f1']

    # All metric fields
    metric_fields = ['ticker', 'plaintiffs_f1', 'defendants_f1',
                     'class_period_start', 'class_period_end', 'causes_f1', 'overall']

    results = []

    for model in sorted(test1_scores['model'].unique()):
        model_data = test1_scores[test1_scores['model'] == model]

        # Calculate means for all metrics
        row = {'model': model}

        for field in metric_fields:
            if field in model_data.columns:
                row[field] = model_data[field].mean()

        # Calculate Macro F1 (mean of the three F1 scores)
        f1_values = [row.get(f, 0) for f in f1_fields]
        row['macro_f1'] = np.mean(f1_values)

        # Calculate class period average
        row['class_period_avg'] = (row.get('class_period_start', 0) + row.get('class_period_end', 0)) / 2

        # Count cases
        row['n_cases'] = len(model_data)
        row['n_successful'] = len(model_data[model_data['overall'] > 0])

        results.append(row)

    # Create DataFrame and sort by Macro F1
    df = pd.DataFrame(results)
    df = df.sort_values('macro_f1', ascending=False).reset_index(drop=True)

    # Add rank column
    df.insert(0, 'rank', range(1, len(df) + 1))

    # Round numeric columns
    numeric_cols = ['ticker', 'plaintiffs_f1', 'defendants_f1', 'causes_f1',
                    'class_period_avg', 'macro_f1', 'overall']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(3)

    # Reorder columns for clarity
    column_order = ['rank', 'model', 'macro_f1', 'plaintiffs_f1', 'defendants_f1',
                    'causes_f1', 'ticker', 'class_period_avg', 'overall', 'n_cases', 'n_successful']
    df = df[[c for c in column_order if c in df.columns]]

    # Save to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_path / "test1_macro_f1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved Macro F1 results to {csv_path}")

    # Save Excel with formatting
    excel_path = output_path / "test1_macro_f1_results.xlsx"
    df.to_excel(excel_path, index=False, sheet_name='Macro F1 Results')
    print(f"Saved Macro F1 results to {excel_path}")

    # Print table to console
    print("\n" + "=" * 100)
    print("TEST 1 RESULTS: MACRO F1 SCORES BY MODEL")
    print("=" * 100)
    print(f"\nMacro F1 = mean(plaintiffs_f1, defendants_f1, causes_f1)\n")
    print(df.to_string(index=False))
    print("\n" + "=" * 100)

    return df


def generate_test2_results_table(
    test2_scores: pd.DataFrame,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Generate a comprehensive Test 2 results table per model and save to results folder.

    Includes:
    - Ruling Macro F1 (primary metric for claim prediction accuracy)
    - Surface metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR
    - Semantic metric: BERTScore
    - Factual metrics: SummaC
    - Overall composite score

    Args:
        test2_scores: DataFrame with per-case scores from evaluate_all_test2
        output_dir: Directory to save results

    Returns:
        DataFrame with comprehensive scores per model
    """
    if test2_scores.empty:
        print("No Test 2 scores available for results table.")
        return pd.DataFrame()

    # All metric fields
    metric_fields = [
        'ruling_f1', 'ruling_accuracy',
        'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor',
        'bertscore_f1',
        'summac',
        'overall'
    ]

    results = []

    for model in sorted(test2_scores['model'].unique()):
        model_data = test2_scores[test2_scores['model'] == model]

        row = {'model': model}

        # Calculate means for all metrics
        for field in metric_fields:
            if field in model_data.columns:
                row[field] = model_data[field].mean()
            else:
                row[field] = 0.0

        # Calculate category averages
        # Surface metrics average
        surface_fields = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']
        row['surface_avg'] = np.mean([row.get(f, 0) for f in surface_fields])

        # Factual metrics average
        factual_fields = ['summac']
        row['factual_avg'] = np.mean([row.get(f, 0) for f in factual_fields])

        # Count cases
        row['n_cases'] = len(model_data)
        row['n_successful'] = len(model_data[model_data['overall'] > 0])

        results.append(row)

    # Create DataFrame and sort by ruling_f1 (primary metric)
    df = pd.DataFrame(results)
    df = df.sort_values('ruling_f1', ascending=False).reset_index(drop=True)

    # Add rank column
    df.insert(0, 'rank', range(1, len(df) + 1))

    # Round numeric columns
    numeric_cols = [
        'ruling_f1', 'ruling_accuracy',
        'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor',
        'bertscore_f1', 'summac',
        'surface_avg', 'factual_avg', 'overall'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(3)

    # Reorder columns for clarity
    column_order = [
        'rank', 'model',
        'ruling_f1', 'ruling_accuracy',
        'surface_avg', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor',
        'bertscore_f1',
        'factual_avg', 'summac',
        'overall', 'n_cases', 'n_successful'
    ]
    df = df[[c for c in column_order if c in df.columns]]

    # Save to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_path / "test2_results_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved Test 2 results to {csv_path}")

    # Save Excel with formatting
    excel_path = output_path / "test2_results_comparison.xlsx"
    df.to_excel(excel_path, index=False, sheet_name='Test 2 Results')
    print(f"Saved Test 2 results to {excel_path}")

    # Print comprehensive table to console
    print("\n" + "=" * 120)
    print("TEST 2 RESULTS: MODEL COMPARISON")
    print("=" * 120)

    # Print Ruling F1 table (primary metric)
    print("\n### RULING PREDICTION (Macro F1) ###")
    print("-" * 60)
    ruling_df = df[['rank', 'model', 'ruling_f1', 'ruling_accuracy', 'n_successful']].copy()
    print(ruling_df.to_string(index=False))

    # Print Surface Metrics table
    print("\n### SURFACE METRICS (Text Overlap with Order) ###")
    print("-" * 80)
    surface_df = df[['rank', 'model', 'surface_avg', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']].copy()
    print(surface_df.to_string(index=False))

    # Print Semantic & Factual Metrics table
    print("\n### SEMANTIC & FACTUAL METRICS ###")
    print("-" * 70)
    semantic_df = df[['rank', 'model', 'bertscore_f1', 'factual_avg', 'summac']].copy()
    print(semantic_df.to_string(index=False))

    # Print Overall Summary
    print("\n### OVERALL SUMMARY ###")
    print("-" * 80)
    print("Overall = 30% surface + 20% semantic + 20% factual + 30% ruling F1")
    print("-" * 80)
    summary_df = df[['rank', 'model', 'overall', 'ruling_f1', 'surface_avg', 'bertscore_f1', 'factual_avg']].copy()
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 120)

    return df


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")

    # Test list F1
    pred = ["Apple Inc.", "Tim Cook", "John Doe"]
    actual = ["Apple", "Tim Cook", "Jane Doe"]
    result = list_f1_score(pred, actual)
    print(f"\nList F1 test: {result}")

    # Test date match
    print(f"\nDate match test:")
    print(f"  2022-03-01 vs 2022-03-01: {date_match('2022-03-01', '2022-03-01')}")
    print(f"  2022-03-01 vs 03/01/2022: {date_match('2022-03-01', '03/01/2022')}")
    print(f"  2022-03-01 vs 2022-03-15: {date_match('2022-03-01', '2022-03-15')}")

    # Test MTD accuracy
    pred_mtd = ["granted", "denied", "granted-in-part"]
    actual_mtd = ["granted", "denied", "denied"]
    mtd_result = mtd_accuracy(pred_mtd, actual_mtd)
    print(f"\nMTD accuracy test: {mtd_result}")

    print("\nEvaluation functions ready!")
