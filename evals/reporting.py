import json
from pathlib import Path
import numpy as np
import pandas as pd

def _get_model_stats(scores_df, test_name):
    stats = []
    for model in scores_df['model'].unique():
        data = scores_df[scores_df['model'] == model]
        successful = len(data[data['overall'] > 0])
        total = successful
        raw = Path(f"results/{test_name}/raw_outputs/{model}_outputs.json")
        if raw.exists(): total = len(json.load(open(raw, 'r', encoding='utf-8')))
        if successful > 0:
            d = data[data['overall'] > 0]
            stats.append({'model': model, 'data': d, 'successful': successful, 'total': total, 'rate': successful / total if total else 0})
    return stats

def generate_summary_report(test1_scores: pd.DataFrame, test2_scores: pd.DataFrame, test3_scores: pd.DataFrame = None, output_path: str = "results/summary_report.md") -> str:
    if test3_scores is None: test3_scores = pd.DataFrame()
    lines = ["# LLM Evaluation Summary Report\n", "## Test 1: Structured Data Extraction\n"]
    if not test1_scores.empty:
        stats = sorted(_get_model_stats(test1_scores, "test1"), key=lambda x: x['data']['overall'].mean(), reverse=True)
        if stats:
            lines += ["| Model | Overall | Ticker | Plaintiffs | Defendants | Causes | Success |", "|-------|---------|--------|------------|------------|--------|---------|"]
            for s in stats:
                d = s['data']
                lines.append(f"| {s['model']} | {d['overall'].mean():.3f} | {d['ticker'].mean():.3f} | {d['plaintiffs_f1'].mean():.3f} | {d['defendants_f1'].mean():.3f} | {d['causes_f1'].mean():.3f} | {s['successful']}/{s['total']} |")
    else: lines.append("No Test 1 results.\n")
    lines.append("\n## Test 2: Summary + Rulings\n")
    if not test2_scores.empty:
        stats = sorted(_get_model_stats(test2_scores, "test2"), key=lambda x: x['data']['overall'].mean(), reverse=True)
        if stats:
            lines += ["| Model | Overall | ROUGE-1 | BERTScore | Faithfulness | Ruling F1 |", "|-------|---------|---------|-----------|--------------|-----------|"]
            for s in stats:
                d = s['data']
                lines.append(f"| {s['model']} | {d['overall'].mean():.3f} | {d['rouge1'].mean():.3f} | {d['bertscore_f1'].mean():.3f} | {d.get('faithfulness', pd.Series([0])).mean():.3f} | {d.get('ruling_f1', pd.Series([0])).mean():.3f} |")
    else: lines.append("No Test 2 results.\n")

    lines.append("\n## Test 3: Scienter-Focused Ruling Predictions\n")
    if not test3_scores.empty:
        stats = []
        for model in test3_scores['model'].unique():
            d = test3_scores[test3_scores['model'] == model]
            if len(d) > 0:
                stats.append({'model': model, 'data': d, 'n_cases': len(d)})
        stats = sorted(stats, key=lambda x: x['data']['ruling_f1'].mean(), reverse=True)
        if stats:
            lines += ["| Model | Ruling F1 | Ruling Accuracy | Cases |", "|-------|-----------|-----------------|-------|"]
            for s in stats:
                d = s['data']
                lines.append(f"| {s['model']} | {d['ruling_f1'].mean():.3f} | {d['ruling_accuracy'].mean():.3f} | {s['n_cases']} |")
    else: lines.append("No Test 3 results.\n")

    # Ruling comparison section
    if not test2_scores.empty or not test3_scores.empty:
        lines.append("\n## Ruling Prediction Comparison: Test 2 vs Test 3\n")
        lines.append("Test 2 uses standard prompts. Test 3 uses scienter-focused prompts.\n")
        models = set()
        if not test2_scores.empty: models.update(test2_scores['model'].unique())
        if not test3_scores.empty: models.update(test3_scores['model'].unique())
        lines += ["| Model | Test2 F1 | Test3 F1 | Improvement |", "|-------|----------|----------|-------------|"]
        for model in sorted(models):
            t2_f1 = test2_scores[test2_scores['model'] == model]['ruling_f1'].mean() if not test2_scores.empty and model in test2_scores['model'].values else 0.0
            t3_f1 = test3_scores[test3_scores['model'] == model]['ruling_f1'].mean() if not test3_scores.empty and model in test3_scores['model'].values else 0.0
            imp = t3_f1 - t2_f1
            imp_str = f"+{imp:.3f}" if imp >= 0 else f"{imp:.3f}"
            lines.append(f"| {model} | {t2_f1:.3f} | {t3_f1:.3f} | {imp_str} |")

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    open(output_path, 'w', encoding='utf-8').write(report)
    return report

def generate_macro_f1_table(test1_scores: pd.DataFrame, output_dir: str = "results") -> pd.DataFrame:
    if test1_scores.empty: return pd.DataFrame()
    rows = []
    for model in sorted(test1_scores['model'].unique()):
        d = test1_scores[test1_scores['model'] == model]
        row = {'model': model, 'macro_f1': np.mean([d['plaintiffs_f1'].mean(), d['defendants_f1'].mean(), d['causes_f1'].mean()])}
        for f in ['ticker', 'plaintiffs_f1', 'defendants_f1', 'causes_f1', 'overall']: row[f] = d[f].mean() if f in d.columns else 0
        row['class_period_avg'] = (d['class_period_start'].mean() + d['class_period_end'].mean()) / 2 if 'class_period_start' in d.columns else 0
        row['n_cases'], row['n_successful'] = len(d), len(d[d['overall'] > 0])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('macro_f1', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    for c in ['ticker', 'plaintiffs_f1', 'defendants_f1', 'causes_f1', 'class_period_avg', 'macro_f1', 'overall']:
        if c in df.columns: df[c] = df[c].round(3)
    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "test1_macro_f1_results.csv", index=False)
    df.to_excel(p / "test1_macro_f1_results.xlsx", index=False)
    return df

def generate_test2_results_table(test2_scores: pd.DataFrame, output_dir: str = "results") -> pd.DataFrame:
    if test2_scores.empty: return pd.DataFrame()
    rows = []
    for model in sorted(test2_scores['model'].unique()):
        d = test2_scores[test2_scores['model'] == model]
        row = {'model': model}
        for f in ['ruling_f1', 'ruling_accuracy', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'bertscore_f1', 'faithfulness', 'overall']:
            row[f] = d[f].mean() if f in d.columns else 0.0
        row['surface_avg'] = np.mean([row['rouge1'], row['rouge2'], row['rougeL'], row['bleu'], row['meteor']])
        row['n_cases'], row['n_successful'] = len(d), len(d[d['overall'] > 0])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('ruling_f1', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    for c in ['ruling_f1', 'ruling_accuracy', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'bertscore_f1', 'faithfulness', 'surface_avg', 'overall']:
        if c in df.columns: df[c] = df[c].round(3)
    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "test2_results_comparison.csv", index=False)
    df.to_excel(p / "test2_results_comparison.xlsx", index=False)
    return df

def generate_ruling_comparison_table(test2_scores: pd.DataFrame, test3_scores: pd.DataFrame, output_dir: str = "results") -> pd.DataFrame:
    """Generate comparison table of ruling predictions between test2 (standard) and test3 (scienter-focused)."""
    if test2_scores.empty and test3_scores.empty: return pd.DataFrame()

    models = set()
    if not test2_scores.empty: models.update(test2_scores['model'].unique())
    if not test3_scores.empty: models.update(test3_scores['model'].unique())

    rows = []
    for model in sorted(models):
        row = {'model': model}

        # Test2 ruling metrics
        if not test2_scores.empty:
            d2 = test2_scores[test2_scores['model'] == model]
            if not d2.empty:
                row['test2_ruling_f1'] = d2['ruling_f1'].mean() if 'ruling_f1' in d2.columns else 0.0
                row['test2_ruling_accuracy'] = d2['ruling_accuracy'].mean() if 'ruling_accuracy' in d2.columns else 0.0
                row['test2_n_cases'] = len(d2)
            else:
                row['test2_ruling_f1'], row['test2_ruling_accuracy'], row['test2_n_cases'] = 0.0, 0.0, 0
        else:
            row['test2_ruling_f1'], row['test2_ruling_accuracy'], row['test2_n_cases'] = 0.0, 0.0, 0

        # Test3 ruling metrics
        if not test3_scores.empty:
            d3 = test3_scores[test3_scores['model'] == model]
            if not d3.empty:
                row['test3_ruling_f1'] = d3['ruling_f1'].mean() if 'ruling_f1' in d3.columns else 0.0
                row['test3_ruling_accuracy'] = d3['ruling_accuracy'].mean() if 'ruling_accuracy' in d3.columns else 0.0
                row['test3_n_cases'] = len(d3)
            else:
                row['test3_ruling_f1'], row['test3_ruling_accuracy'], row['test3_n_cases'] = 0.0, 0.0, 0
        else:
            row['test3_ruling_f1'], row['test3_ruling_accuracy'], row['test3_n_cases'] = 0.0, 0.0, 0

        # Calculate improvement (test3 - test2)
        row['f1_improvement'] = row['test3_ruling_f1'] - row['test2_ruling_f1']
        row['accuracy_improvement'] = row['test3_ruling_accuracy'] - row['test2_ruling_accuracy']

        rows.append(row)

    df = pd.DataFrame(rows).sort_values('test3_ruling_f1', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))

    for c in ['test2_ruling_f1', 'test2_ruling_accuracy', 'test3_ruling_f1', 'test3_ruling_accuracy', 'f1_improvement', 'accuracy_improvement']:
        if c in df.columns: df[c] = df[c].round(3)

    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "ruling_comparison_test2_vs_test3.csv", index=False)
    df.to_excel(p / "ruling_comparison_test2_vs_test3.xlsx", index=False)
    return df
