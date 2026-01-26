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
    # Remove common legal suffixes
    for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", corp.", " corp.", ", co.", " co."]:
        name = name.replace(suffix, "")
    return name.strip()


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
    import re

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


def evaluate_test1_case(predicted: dict, actual: dict) -> dict:
    """
    Evaluate Test 1 extraction for a single case.

    Args:
        predicted: LLM output dict with plaintiffs, defendants, ticker, class_period, causes_of_action
        actual: Ground truth dict with same structure

    Returns:
        dict with scores for each field
    """
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

    # Causes of action (compare number and names)
    pred_causes = predicted.get('causes_of_action', []) or []
    actual_causes = actual.get('causes_of_action', []) or []

    # Extract just the claim names
    pred_claims = [c.get('claim', '') for c in pred_causes if isinstance(c, dict)]
    actual_claims = [c.get('claim', '') for c in actual_causes if isinstance(c, dict)]

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


def mtd_accuracy(predicted_outcomes: list, actual_outcomes: list) -> dict:
    """
    Calculate accuracy for MTD outcome predictions.

    Args:
        predicted_outcomes: List of predictions ('granted', 'denied', 'granted-in-part')
        actual_outcomes: List of actual outcomes

    Returns:
        dict with accuracy and per-class metrics
    """
    if not predicted_outcomes or not actual_outcomes:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    # Normalize values
    def normalize_outcome(o):
        if not o:
            return None
        o = str(o).lower().strip()
        if 'grant' in o and 'part' in o:
            return 'granted-in-part'
        elif 'grant' in o:
            return 'granted'
        elif 'denied' in o or 'deny' in o:
            return 'denied'
        return o

    pred_norm = [normalize_outcome(p) for p in predicted_outcomes]
    actual_norm = [normalize_outcome(a) for a in actual_outcomes]

    # Count matches
    min_len = min(len(pred_norm), len(actual_norm))
    correct = sum(1 for p, a in zip(pred_norm[:min_len], actual_norm[:min_len]) if p == a)

    return {
        "accuracy": correct / min_len if min_len > 0 else 0.0,
        "correct": correct,
        "total": min_len
    }


def evaluate_test2_case(
    predicted: dict,
    actual: dict,
    source_text: Optional[str] = None,
    compute_summac: bool = False
) -> dict:
    """
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

            # Convert ground truth row to dict
            actual = {
                'plaintiffs': str(gt_row['plaintiffs'].iloc[0]).split(';') if pd.notna(gt_row['plaintiffs'].iloc[0]) else [],
                'defendants': str(gt_row['defendants'].iloc[0]).split(';') if pd.notna(gt_row['defendants'].iloc[0]) else [],
                'ticker': gt_row['ticker'].iloc[0] if pd.notna(gt_row['ticker'].iloc[0]) else None,
                'class_period': {
                    'start': gt_row['class_period_start'].iloc[0] if pd.notna(gt_row['class_period_start'].iloc[0]) else None,
                    'end': gt_row['class_period_end'].iloc[0] if pd.notna(gt_row['class_period_end'].iloc[0]) else None,
                },
                'causes_of_action': []
            }

            # Add causes of action
            for i in range(1, 5):
                cause = gt_row[f'cause_{i}'].iloc[0] if pd.notna(gt_row.get(f'cause_{i}', pd.Series([None])).iloc[0]) else None
                if cause:
                    actual['causes_of_action'].append({'claim': cause})

            # Evaluate
            scores = evaluate_test1_case(predicted, actual)
            scores['model'] = model_name
            scores['case_id'] = case_id
            all_scores.append(scores)

    return pd.DataFrame(all_scores)


def evaluate_all_test2(
    results: dict,
    ground_truth_df: pd.DataFrame,
    complaints_df: Optional[pd.DataFrame] = None,
    compute_summac: bool = False
) -> pd.DataFrame:
    """
    Evaluate all Test 2 results for all models.

    Args:
        results: dict mapping model_name -> list of result dicts
        ground_truth_df: DataFrame with ground truth (from ground_truth_test2.xlsx)
        complaints_df: DataFrame with complaint texts (for SummaC)
        compute_summac: Whether to compute SummaC scores

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

            # Convert to dict
            actual = gt_row.iloc[0].to_dict()

            # Get source text if needed
            source_text = None
            if compute_summac and complaints_df is not None:
                complaint_row = complaints_df[complaints_df['case_id'] == case_id]
                if not complaint_row.empty:
                    source_text = complaint_row['complaint_text'].iloc[0]

            # Evaluate
            scores = evaluate_test2_case(predicted, actual, source_text, compute_summac)
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
    lines = ["# LLM Evaluation Summary Report\n"]

    # Test 1 Summary
    lines.append("## Test 1: Structured Data Extraction\n")

    if not test1_scores.empty:
        summary = test1_scores.groupby('model').agg({
            'overall': 'mean',
            'ticker': 'mean',
            'plaintiffs_f1': 'mean',
            'defendants_f1': 'mean',
            'causes_f1': 'mean'
        }).round(3)

        lines.append("| Model | Overall | Ticker | Plaintiffs F1 | Defendants F1 | Causes F1 |")
        lines.append("|-------|---------|--------|---------------|---------------|-----------|")
        for model in summary.index:
            row = summary.loc[model]
            lines.append(f"| {model} | {row['overall']:.3f} | {row['ticker']:.3f} | {row['plaintiffs_f1']:.3f} | {row['defendants_f1']:.3f} | {row['causes_f1']:.3f} |")
    else:
        lines.append("No Test 1 results available.\n")

    lines.append("")

    # Test 2 Summary
    lines.append("## Test 2: Summarization + MTD Prediction\n")

    if not test2_scores.empty:
        summary = test2_scores.groupby('model').agg({
            'overall': 'mean',
            'rouge1': 'mean',
            'rougeL': 'mean',
            'bertscore_f1': 'mean',
            'mtd_accuracy': 'mean'
        }).round(3)

        lines.append("| Model | Overall | ROUGE-1 | ROUGE-L | BERTScore F1 | MTD Accuracy |")
        lines.append("|-------|---------|---------|---------|--------------|--------------|")
        for model in summary.index:
            row = summary.loc[model]
            lines.append(f"| {model} | {row['overall']:.3f} | {row['rouge1']:.3f} | {row['rougeL']:.3f} | {row['bertscore_f1']:.3f} | {row['mtd_accuracy']:.3f} |")
    else:
        lines.append("No Test 2 results available.\n")

    report = "\n".join(lines)

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


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
