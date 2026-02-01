import re
from pathlib import Path
from typing import Optional
import pandas as pd
from config import PROCESSED_COMPLAINTS_DIR, PROCESSED_ORDERS_DIR, GROUND_TRUTH_TEST1_PATH, GROUND_TRUTH_TEST2_PATH, COURT_SUMMARIES_PATH

def parse_case_id(filename: str) -> dict:
    
    case_id = filename.replace('.txt', '').replace('.pdf', '')
    m = re.match(r'^([a-z]+)[-_]?(\d)?[-_]?(\d{2})[-_]?cv[-_]?(\d+)[-_]?\d?$', case_id, re.IGNORECASE)
    
    if m: 
        return {'case_id': case_id, 'court': m.group(1), 'division': m.group(2), 'year': f"20{m.group(3)}", 'case_number': m.group(4)}
    
    return {'case_id': case_id, 'court': None, 'division': None, 'year': None, 'case_number': None}

def load_complaint_texts(d: str = PROCESSED_COMPLAINTS_DIR) -> dict:
    p = Path(d)
    if not p.exists(): 
        raise FileNotFoundError(f"Directory not found: {d}")
    
    return {f.stem: open(f, 'r', encoding='utf-8').read() for f in p.glob("*.txt")}

def load_order_texts(d: str = PROCESSED_ORDERS_DIR) -> dict:
    p = Path(d)
    return {f.stem: open(f, 'r', encoding='utf-8').read() for f in p.glob("*.txt")} if p.exists() else {}

def load_ground_truth_test1(path: str = GROUND_TRUTH_TEST1_PATH) -> pd.DataFrame:
    
    if not Path(path).exists(): 
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_excel(path)

def load_ground_truth_test2(path: str = GROUND_TRUTH_TEST2_PATH) -> pd.DataFrame:
    
    if not Path(path).exists(): 
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_excel(path); df.columns = df.columns.str.strip()
    for c in df.columns:
        if c.lower() == 'summary' and 'summary' not in df.columns: df = df.rename(columns={c: 'summary'}); break
    return df

def load_ground_truth_test2_orders(d: str = PROCESSED_ORDERS_DIR) -> pd.DataFrame:
    texts = load_order_texts(d)
    if not texts: 
        raise FileNotFoundError(f"No orders in: {d}")
    return pd.DataFrame([{'case_id': k, 'order_text': v, 'order_text_length': len(v)} for k, v in texts.items()]).sort_values('case_id').reset_index(drop=True)

def load_court_summaries(path: str = COURT_SUMMARIES_PATH) -> pd.DataFrame:
    if not Path(path).exists(): 
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_excel(path).rename(columns={'Name': 'case_id', 'Summaries': 'order_summary'})
    return df[[c for c in df.columns if not c.startswith('Unnamed')]]

def build_complaints_df(complaints_dir: str = PROCESSED_COMPLAINTS_DIR, ground_truth_path: Optional[str] = GROUND_TRUTH_TEST1_PATH) -> pd.DataFrame:
    
    texts = load_complaint_texts(complaints_dir)
    rows = [{**parse_case_id(k), 'complaint_text': v, 'text_length': len(v)} for k, v in texts.items()]
    df = pd.DataFrame(rows)
    
    if ground_truth_path and Path(ground_truth_path).exists(): 
        df = df.merge(load_ground_truth_test1(ground_truth_path), on='case_id', how='left')
    
    return df.sort_values('case_id').reset_index(drop=True)

def build_orders_df(orders_dir: str = PROCESSED_ORDERS_DIR, court_summaries_path: str = COURT_SUMMARIES_PATH, ground_truth_path: Optional[str] = GROUND_TRUTH_TEST2_PATH) -> pd.DataFrame:
    
    texts = load_order_texts(orders_dir)
    rows = [{**parse_case_id(k), 'order_text': v, 'order_text_length': len(v)} for k, v in texts.items()]
    df = pd.DataFrame(rows)
    
    if Path(court_summaries_path).exists(): 
        df = df.merge(load_court_summaries(court_summaries_path), on='case_id', how='left')
    
    if ground_truth_path and Path(ground_truth_path).exists():
        gt = load_ground_truth_test2(ground_truth_path)
        df = df.merge(gt.drop(columns=['case_id'], errors='ignore'), left_on='case_id', right_on=gt['case_id'] if 'case_id' in gt.columns else None, how='left')
    
    return df.sort_values('case_id').reset_index(drop=True)

def get_case_ids() -> list: return sorted(load_complaint_texts().keys())
