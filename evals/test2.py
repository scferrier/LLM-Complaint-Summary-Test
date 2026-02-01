from typing import Optional
import pandas as pd
from .metrics import rouge_scores, bleu_score, meteor_score, bert_score, faithfulness_score, ruling_f1_score

def evaluate_test2_case(predicted: dict, ground_truth_summary: str, source_text: Optional[str] = None,
                        ground_truth_rulings: Optional[list] = None, compute_factual: bool = True) -> dict:
    pred = predicted.get('summary', '')
    if not pred or not ground_truth_summary:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu': 0.0, 'meteor': 0.0, 'bertscore_f1': 0.0, 'faithfulness': 0.0, 'ruling_f1': 0.0, 'ruling_accuracy': 0.0, 'overall': 0.0}
    r = rouge_scores(pred, ground_truth_summary)
    s = {'rouge1': r['rouge1'], 'rouge2': r['rouge2'], 'rougeL': r['rougeL'], 'bleu': bleu_score(pred, ground_truth_summary), 'meteor': meteor_score(pred, ground_truth_summary), 'bertscore_f1': bert_score(pred, ground_truth_summary)['f1']}
    s['faithfulness'] = faithfulness_score(source_text, pred) if compute_factual and source_text else 0.0
    if ground_truth_rulings:
        rulings = predicted.get('claim_rulings', []) or predicted.get('mtd_predictions', [])
        pred_rulings = [ru.get('ruling', ru.get('predicted_outcome', '')) for ru in rulings if isinstance(ru, dict)]
        rm = ruling_f1_score(pred_rulings, ground_truth_rulings)
        s['ruling_f1'], s['ruling_accuracy'], s['ruling_correct'], s['ruling_total'] = rm['macro_f1'], rm['accuracy'], rm['correct'], rm['total']
    else: s['ruling_f1'], s['ruling_accuracy'] = 0.0, 0.0
    surf, sem, fact, rul = (s['rouge1'] + s['rouge2'] + s['rougeL'] + s['bleu'] + s['meteor']) / 5, s['bertscore_f1'], s['faithfulness'] if compute_factual else 0.0, s['ruling_f1']
    if compute_factual and ground_truth_rulings: s['overall'] = 0.3 * surf + 0.2 * sem + 0.2 * fact + 0.3 * rul
    elif ground_truth_rulings: s['overall'] = 0.4 * surf + 0.3 * sem + 0.3 * rul
    elif compute_factual: s['overall'] = 0.4 * surf + 0.3 * sem + 0.3 * fact
    else: s['overall'] = 0.6 * surf + 0.4 * sem
    return s

def _find_row(df, case_id, col='case_id'):
    row = df[df[col] == case_id]
    if row.empty:
        norm = case_id.replace('-', '_').replace(' ', '_')
        row = df[df[col].str.replace('-', '_').str.replace(' ', '_') == norm]
    return row

def evaluate_all_test2(results: dict, ground_truth_df: pd.DataFrame, complaints_df: Optional[pd.DataFrame] = None,
                       rulings_df: Optional[pd.DataFrame] = None, compute_factual: bool = True) -> pd.DataFrame:
    scores = []
    for model, model_results in results.items():
        for r in model_results:
            if not r.get('success'): continue
            case_id = r['case_id']
            gt = _find_row(ground_truth_df, case_id)
            if gt.empty: continue
            summary = gt.iloc[0].get('summary', '')
            if not summary: continue
            src = None
            if compute_factual and complaints_df is not None:
                cr = _find_row(complaints_df, case_id)
                if not cr.empty: src = cr['complaint_text'].iloc[0]
            rulings = None
            if rulings_df is not None:
                rr = _find_row(rulings_df, case_id)
                if not rr.empty:
                    rulings = [str(rr.iloc[0][f'cause_{i}_mtd_outcome']) for i in range(1, 16) if f'cause_{i}_mtd_outcome' in rr.iloc[0].index and pd.notna(rr.iloc[0][f'cause_{i}_mtd_outcome'])]
            s = evaluate_test2_case(r.get('response', {}), summary, src, rulings, compute_factual)
            s['model'], s['case_id'] = model, case_id
            scores.append(s)
    return pd.DataFrame(scores)
