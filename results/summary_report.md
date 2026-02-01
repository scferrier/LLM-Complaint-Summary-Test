# LLM Evaluation Summary Report

## Test 1: Structured Data Extraction

| Model | Overall | Ticker | Plaintiffs | Defendants | Causes | Success Rate |
|-------|---------|--------|------------|------------|--------|--------------|
| gpt-5.2 | 0.810 | 1.000 | 0.663 | 0.812 | 0.674 | 25/25 (100%) |
| grok | 0.760 | 1.000 | 0.663 | 0.780 | 0.455 | 25/25 (100%) |
| gemini | 0.746 | 0.917 | 0.656 | 0.745 | 0.538 | 24/25 (96%) |
| claude-opus | 0.730 | 0.960 | 0.595 | 0.776 | 0.420 | 25/25 (100%) |

*Scores calculated on successful cases only*


## Test 2: Judicial Summary + Claim Rulings vs Order Text

### Surface & Semantic Metrics

| Model | Overall | ROUGE-1 | ROUGE-L | BLEU | METEOR | BERTScore |
|-------|---------|---------|---------|------|--------|-----------|
| gpt-5.2 | 0.283 | 0.448 | 0.172 | 0.071 | 0.252 | -0.004 |
| perplexity | 0.266 | 0.287 | 0.134 | 0.025 | 0.130 | -0.060 |
| claude-opus | 0.260 | 0.330 | 0.153 | 0.033 | 0.152 | -0.023 |
| gemini | 0.259 | 0.255 | 0.130 | 0.012 | 0.104 | -0.003 |
| grok | 0.256 | 0.306 | 0.134 | 0.040 | 0.147 | -0.075 |

### Factual Consistency & Ruling Metrics

| Model | RAGAS Faithfulness | Ruling F1 | Success Rate |
|-------|--------|-----------|--------------|
| gpt-5.2 | 0.550 | 0.360 | 22/25 (88%) |
| perplexity | 0.532 | 0.436 | 24/25 (96%) |
| claude-opus | 0.544 | 0.361 | 23/25 (92%) |
| gemini | 0.580 | 0.358 | 25/25 (100%) |
| grok | 0.528 | 0.408 | 25/25 (100%) |

*Scores calculated on successful cases only*

*Overall = 30% surface + 20% semantic + 20% factual + 30% ruling F1*
