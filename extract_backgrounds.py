import os, pandas as pd
from typing import Optional
import argparse

from clean_pdf import process_pdf
from prompts import format_background_extraction_prompt
from llm_inference import call_llm
from config import MODELS

def extract_background_from_order(order_text: str, model: str = "perplexity", max_tokens: int = 8000) -> Optional[str]:
    resp = call_llm(format_background_extraction_prompt(order_text), MODELS.get(model, model), model_name=model, max_tokens=max_tokens, temperature=0.1)
    if resp and resp.get('success'):
        data = resp.get('response', {})
        if isinstance(data, dict): 
            return data.get('background') or data.get('raw_response')
    return None

def process_all_orders(orders_dir: str = "Selected Cases/Orders/PDFs", output_path: str = "data/extracted_backgrounds.csv", model: str = "perplexity") -> pd.DataFrame:
    
    pdfs = sorted([f for f in os.listdir(orders_dir) if f.endswith('.pdf')])
    results = []
    
    for f in pdfs:
        cid = f.replace('.pdf', '')
        try:
            text, _ = process_pdf(os.path.join(orders_dir, f))
            bg = extract_background_from_order(text, model=model)
            results.append({'case_id': cid, 'background': bg, 'background_length': len(bg) if bg else 0})
        except:
            results.append({'case_id': cid, 'background': None, 'background_length': 0})
    
    df = pd.DataFrame(results); df.to_csv(output_path, index=False)
    return df

def update_ground_truth(extracted_path: str = "data/extracted_backgrounds.csv", ground_truth_path: str = "data/ground_truth_test2.xlsx"):
    ext, gt = pd.read_csv(extracted_path), pd.read_excel(ground_truth_path)
    
    gt.columns = gt.columns.str.strip()
    
    for _, r in ext.iterrows():
        if pd.notna(r['background']): gt.loc[gt['case_id'] == r['case_id'], 'summary'] = r['background']
    
    gt.to_excel(ground_truth_path, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="perplexity"); p.add_argument("--orders-dir", default="Selected Cases/Orders/PDFs")
    p.add_argument("--output", default="data/extracted_backgrounds.csv"); p.add_argument("--update-gt", action="store_true"); p.add_argument("--single", type=str)
    args = p.parse_args()
    if args.single:
        text, _ = process_pdf(os.path.join(args.orders_dir, f"{args.single}.pdf"))
        print(extract_background_from_order(text, model=args.model))
    else:
        df = process_all_orders(args.orders_dir, args.output, args.model)
        if args.update_gt: update_ground_truth(args.output)
