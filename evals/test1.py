import pandas as pd
from .cleaning import clean_llm_response, clean_ground_truth_row
from .metrics import exact_match, list_f1_score, date_match

def evaluate_test1_case(predicted: dict, actual: dict, apply_cleaning: bool = True) -> dict:
    if apply_cleaning:
        predicted = clean_llm_response(predicted)
        if not all(k in actual for k in ['plaintiffs', 'defendants', 'ticker', 'class_period', 'causes_of_action']):
            actual = clean_ground_truth_row(actual)
    s = {'ticker': exact_match(predicted.get('ticker'), actual.get('ticker'))}
    for field in ['plaintiffs', 'defendants']:
        r = list_f1_score(predicted.get(field, []), actual.get(field, []))
        s[f'{field}_precision'], s[f'{field}_recall'], s[f'{field}_f1'] = r['precision'], r['recall'], r['f1']
    pp, ap = predicted.get('class_period', {}) or {}, actual.get('class_period', {}) or {}
    s['class_period_start'] = date_match(pp.get('start'), ap.get('start'))
    s['class_period_end'] = date_match(pp.get('end'), ap.get('end'))
    extract = lambda causes: [c if isinstance(c, str) else c.get('claim', '') for c in causes]
    cr = list_f1_score(extract(predicted.get('causes_of_action', []) or []), extract(actual.get('causes_of_action', []) or []))
    s['causes_precision'], s['causes_recall'], s['causes_f1'] = cr['precision'], cr['recall'], cr['f1']
    s['overall'] = sum([s['ticker'], s['plaintiffs_f1'], s['defendants_f1'], (s['class_period_start'] + s['class_period_end']) / 2, s['causes_f1']]) / 5
    return s

def evaluate_all_test1(results: dict, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for model, model_results in results.items():
        for r in model_results:
            if not r.get('success'): continue
            gt = ground_truth_df[ground_truth_df['case_id'] == r['case_id']]
            if gt.empty: continue
            s = evaluate_test1_case(r.get('response', {}), clean_ground_truth_row(gt.iloc[0].to_dict()))
            s['model'], s['case_id'] = model, r['case_id']
            scores.append(s)
    return pd.DataFrame(scores)
