import re
import pandas as pd

def clean_text_for_comparison(text: str) -> str:
    if not text or not isinstance(text, str): return ""
    text = text.replace('§', 'Section ').replace('�', 'Section ').replace('\u00a7', 'Section ')
    return re.sub(r'\s+', ' ', re.sub(r'\s*&\s*', ' and ', text)).strip()

def normalize_name(name: str) -> str:
    if not name: return ""
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name.lower().strip())
    for s in [", inc.", ", inc", " inc.", " inc", ", llc", " llc", ", corp.", " corp.", ", co.", " co."]:
        name = name.replace(s, "")
    return re.sub(r'\s+', ' ', name.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"').replace("�", "'")).strip()

def normalize_cause_of_action(cause: str) -> str:
    if not cause: return ""
    cause = clean_text_for_comparison(cause)
    for p in [r'^violations?\s+of\s+', r'^claims?\s+under\s+', r'^claims?\s+for\s+']:
        cause = re.sub(p, '', cause, flags=re.IGNORECASE)
    return cause.split('\n')[0].strip()

def normalize_ruling(outcome: str) -> str:
    if not outcome: return None
    o = str(outcome).lower().strip()
    if 'dismiss' in o and 'part' in o: return 'dismissed_in_part'
    if 'grant' in o and 'part' in o: return 'dismissed_in_part'
    if 'dismiss' in o or 'grant' in o: return 'dismissed'
    if 'sustain' in o or 'denied' in o or 'deny' in o or 'survive' in o: return 'sustained'
    return o

def split_list_field(text: str) -> list:
    if not text or not isinstance(text, str): return []
    text = clean_text_for_comparison(text)
    if ';' in text: items = [i.strip() for i in text.split(';')]
    elif ',' in text: items = [i.strip() for i in re.sub(r',\s+and\s+', ', ', text, flags=re.IGNORECASE).split(',')]
    elif ' and ' in text.lower(): items = [i.strip() for i in re.split(r'\s+and\s+', text, flags=re.IGNORECASE)]
    else: items = [text.strip()]
    seen, result = set(), []
    for item in items:
        n = normalize_name(item)
        if item.strip() and n and n not in seen: seen.add(n); result.append(item.strip())
    return result

def clean_ground_truth_row(gt_row: dict) -> dict:
    c = {'plaintiffs': [], 'defendants': [], 'ticker': None, 'class_period': {'start': None, 'end': None}, 'causes_of_action': []}
    if 'plaintiffs' in gt_row and pd.notna(gt_row.get('plaintiffs')): c['plaintiffs'] = split_list_field(str(gt_row['plaintiffs']))
    if 'defendants' in gt_row and pd.notna(gt_row.get('defendants')): c['defendants'] = split_list_field(str(gt_row['defendants']))
    if 'ticker' in gt_row and pd.notna(gt_row.get('ticker')): c['ticker'] = str(gt_row['ticker']).strip().upper()
    if 'class_period_start' in gt_row and pd.notna(gt_row.get('class_period_start')): c['class_period']['start'] = str(gt_row['class_period_start'])[:10]
    if 'class_period_end' in gt_row and pd.notna(gt_row.get('class_period_end')): c['class_period']['end'] = str(gt_row['class_period_end'])[:10]
    for i in range(1, 16):
        if f'cause_{i}' in gt_row and pd.notna(gt_row.get(f'cause_{i}')):
            cause = normalize_cause_of_action(str(gt_row[f'cause_{i}']))
            if cause: c['causes_of_action'].append(cause)
    return c

def clean_llm_response(response: dict) -> dict:
    c = {'plaintiffs': [], 'defendants': [], 'ticker': None, 'class_period': {'start': None, 'end': None}, 'causes_of_action': []}
    if isinstance(response.get('plaintiffs', []), list): c['plaintiffs'] = [clean_text_for_comparison(p) for p in response['plaintiffs'] if p]
    if isinstance(response.get('defendants', []), list): c['defendants'] = [clean_text_for_comparison(d) for d in response['defendants'] if d]
    if response.get('ticker'): c['ticker'] = str(response['ticker']).strip().upper()
    period = response.get('class_period', {}) or {}
    if period.get('start'): c['class_period']['start'] = str(period['start'])[:10]
    if period.get('end'): c['class_period']['end'] = str(period['end'])[:10]
    for cause in response.get('causes_of_action', []) or []:
        if isinstance(cause, str): c['causes_of_action'].append(normalize_cause_of_action(cause))
        elif isinstance(cause, dict): c['causes_of_action'].append(normalize_cause_of_action(cause.get('claim', '')))
    return c
