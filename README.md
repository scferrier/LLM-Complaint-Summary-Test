# LLM-Complaint-Summary-Test
## Introduction

This readme and code are intended to define and evaluate benchmarks for assessing the performance of large language models (LLMs) on legal-document summarization tasks. In particular, the focus is on complex civil complaints, which present challenges that are not well captured by existing general-purpose summarization benchmarks.

Prior scholarly work has shown that LLM performance degrades as context length increases and as tasks move beyond surface-level or literal matching. See, e.g., *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (https://arxiv.org/abs/2502.05167). Legal complaints exacerbate these issues: they are lengthy, fact-dense, and often rely on layered narratives to support multiple (and sometimes contradictory) theories of liability.

Accordingly, this work concentrates on complaints that are likely to stress current LLM capabilities due to both their length and structural complexity. The goal is not to demonstrate isolated successes, but to provide a realistic assessment of current LLMs capabilities. To that end, the models are evaluated through three distinct tests. The first test assesses whether an LLM can reliably extract simple, objective information from a complaint. The second test evaluates whether an LLM can objectively summarize the complaint and assess the likelihood that each asserted cause of action would survive a motion to dismiss. Then a third test that provides a more detailed prompt can help improve predictions of a claim's likelihood of success.

## The Tests

### Test 1: Structured Data Extraction

This test evaluates whether LLMs can reliably extract simple, objective information from a complaint. The model is asked to identify:

- **Plaintiffs**: All named plaintiffs in the case
- **Defendants**: All named defendants
- **Ticker Symbol**: The stock ticker if mentioned
- **Class Period**: Start and end dates of the alleged fraud
- **Causes of Action**: Each legal claim asserted

**Evaluation Approach:**
- **Ticker**: Exact match (case-insensitive)
- **Plaintiffs/Defendants**: F1 score using normalized name matching (removes corporate suffixes like "Inc.", "LLC", handles punctuation variants)
- **Class Period**: Date matching with partial credit (full credit for exact match, 0.8 for same year/month)
- **Causes of Action**: F1 score with text normalization (strips prefixes like "Violation of", "Claim under")
- **Overall Score**: Weighted average of all component scores

### Test 2: Judicial Summary & Motion-to-Dismiss Prediction

This test evaluates whether LLMs can objectively summarize a complaint and assess the likelihood that each asserted cause of action would survive a motion to dismiss. The model acts as a federal district court judge and provides:

- A comprehensive summary covering parties, alleged misconduct, key facts, legal claims, and class period
- A ruling ("dismissed" or "sustained") with reasoning for each cause of action

**Evaluation Approach:**

*Summary Quality:*
- **ROUGE-1/2/L**: N-gram overlap with reference summaries (using stemming)
- **BLEU**: Precision-based n-gram matching
- **METEOR**: Flexible matching with synonyms and stemming
- **BERTScore**: Semantic similarity using contextual embeddings (rescaled with baseline)
- **Faithfulness**: NLI-based scoring using BART-MNLI - splits summary into sentences and scores each for entailment against the source complaint

*Ruling Prediction:*
- **Ruling Accuracy**: Percentage of correctly predicted outcomes
- **Ruling F1**: Macro F1 score across ruling categories (dismissed/sustained/dismissed_in_part)
- Rulings are normalized before comparison (e.g., "granted" → "dismissed", "denied" → "sustained")

### Test 3: Scienter-Focused Prompt Engineering

This test addresses the systematic prediction bias observed in Test 2 by providing a more detailed prompt that applies the heightened pleading standards of the Private Securities Litigation Reform Act (PSLRA). The prompt instructs models to:

- **Scrutinize Scienter Allegations**: Require particularized facts giving rise to a strong inference of fraudulent intent
- **Dismiss Conclusory Allegations**: Reject claims based on "fraud by hindsight" or speculation about what defendants "must have known"
- **Apply Derivative Claim Logic**: If Section 10(b)/Rule 10b-5 claims are dismissed, Section 20(a) control person claims must also be dismissed

**Evaluation Approach:**
- Uses the same ruling prediction metrics as Test 2 (Accuracy, Macro F1)
- Compares prediction distribution against ground truth to measure bias correction
- Tracks improvement in accuracy relative to Test 2 baseline

This test demonstrates whether explicit legal framework guidance in the prompt can improve prediction accuracy.

## Metrics Used to Evaluate Performance

To evaluate the accuracy and faithfulness of LLM-generated outputs, this work employs a combination of established automated summarization metrics. These metrics capture complementary aspects of similarity, semantic alignment, and factual consistency between LLM outputs and reference texts.

### Surface-Level Metrics
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation): Measures n-gram overlap between generated and reference text (ROUGE-1, ROUGE-2, ROUGE-L)
- **BLEU** (Bilingual Evaluation Understudy): Precision-based metric measuring n-gram matches
- **METEOR**: Considers synonyms and stemming for more flexible matching

### Semantic Metrics
- **BERTScore**: Uses contextual embeddings to measure semantic similarity beyond surface-level matching

### Factual Consistency Metrics

Initially, we planned to use **SummaC** and **QAFactEval** to assess factual consistency. However, these metrics proved computationally expensive and had compatibility issues with our evaluation pipeline.

We instead adopted a **BART-MNLI based faithfulness metric** inspired by the RAGAS framework. This approach:
- Uses the `facebook/bart-large-mnli` model for Natural Language Inference (NLI)
- Splits the generated summary into sentences
- Scores each sentence for entailment against the source document
- Returns the average entailment score (0-1 scale)

This metric is faster than SummaC while still providing meaningful factual consistency scores. It measures whether claims in the LLM-generated summary can be inferred from the source complaint text.

### Ruling Prediction Metrics
- **Ruling F1**: Macro F1 score comparing predicted motion-to-dismiss outcomes against actual court rulings
- **Ruling Accuracy**: Simple accuracy of ruling predictions

For an overview of summarization metrics and their respective strengths and limitations, see Shannon Gallagher, Swati Rallapalli, and Tyler Brooks, *Evaluating LLMs for Text Summarization: An Introduction*, SEI Blog (Apr. 7, 2025) (https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction).

The amount of tokens used and cost will also be tracked and factored in to evaluate performance.

## Data Collection & Cleaning

Collection 

- Used Court Listener to identify all Securities and Commodities Cases filed in the last 5-years.
- Used the delta between filing and termination to identify cases that likely had motion to dismiss briefing and decisions. 
- Identify cases with 25 orders that are publicly available and that had a background section that described the compliant that was greater than 3,000 words 
- Pulled relevant complaints
- For data to fine-tune used the list from Court Listener to pull order from GovInfo & Justia
- For orders pulled then went back to Court Listener and looked for the operative compliant and pulled that if available 
- Gathered 25 Complaints & Orders for evaluation
- Gathered 100 Complaints & Orders for finetuning

Cleaning

- Removed headers and footers 
- Removed case captions and sig blocks
- If not OCR'ed made text searchable
- For Orders - pulled background section for summary evaluation
## Models Tested

The following models were evaluated:
- **GPT-5.2** (OpenAI)
- **Claude Opus 4.5** (Anthropic)
- **Gemini 3 Flash** (Google)
- **Grok 4.1** (xAI)
- **Sonar Pro** (Perplexity)

## Model Results

### Test 1: Structured Data Extraction

Test 1 evaluated each model's ability to extract objective information from complaints: plaintiffs, defendants, ticker symbols, class periods, and causes of action.

| Rank | Model | Overall | Macro F1 | Ticker | Plaintiffs F1 | Defendants F1 | Causes F1 |
|------|-------|---------|----------|--------|---------------|---------------|-----------|
| 1 | **GPT-5.2** | **0.810** | **0.716** | 1.00 | 0.663 | **0.812** | **0.674** |
| 2 | Grok | 0.760 | 0.633 | 1.00 | 0.663 | 0.780 | 0.455 |
| 3 | Gemini | 0.716 | 0.620 | 0.88 | 0.629 | 0.715 | 0.516 |
| 4 | Claude Opus | 0.730 | 0.597 | 0.96 | 0.595 | 0.776 | 0.420 |

**Key Findings (Test 1):**
- **GPT-5.2** performed best overall, with the highest Macro F1 (0.716) and strongest performance on defendant and cause of action extraction
- All models achieved near-perfect ticker symbol extraction (88-100%)
- Cause of action extraction was the most challenging task across all models

### Test 2: Judicial Summary & Motion-to-Dismiss Prediction

Test 2 evaluated each model's ability to (1) summarize the complaint and (2) predict motion-to-dismiss outcomes for each cause of action.

#### Ruling Prediction Performance

| Rank | Model | Ruling F1 | Ruling Accuracy | Success Rate |
|------|-------|-----------|-----------------|--------------|
| 1 | **Perplexity** | **0.436** | **0.465** | 96% |
| 2 | Grok | 0.408 | 0.472 | 100% |
| 3 | Claude Opus | 0.361 | 0.358 | 92% |
| 4 | GPT-5.2 | 0.360 | 0.357 | 88% |
| 5 | Gemini | 0.358 | 0.408 | 100% |

#### Summary Quality Metrics

| Model | ROUGE-1 | ROUGE-L | BLEU | METEOR | BERTScore | Faithfulness |
|-------|---------|---------|------|--------|-----------|--------------|
| GPT-5.2 | **0.448** | **0.172** | **0.071** | **0.252** | -0.004 | 0.550 |
| Claude Opus | 0.330 | 0.153 | 0.033 | 0.152 | -0.023 | 0.544 |
| Grok | 0.306 | 0.134 | 0.040 | 0.147 | -0.075 | 0.528 |
| Perplexity | 0.287 | 0.134 | 0.025 | 0.130 | -0.060 | 0.532 |
| Gemini | 0.255 | 0.130 | 0.012 | 0.104 | -0.003 | **0.580** |

**Key Findings (Test 2):**
- **Perplexity** achieved the best ruling prediction performance (F1: 0.436), making it the top choice for predicting motion-to-dismiss outcomes
- **GPT-5.2** produced summaries with the highest surface-level similarity to reference summaries (ROUGE-1: 0.448)
- **Gemini** achieved the highest faithfulness score (0.580), indicating strong factual consistency with source documents
- Negative BERTScores across models suggest generated summaries use different vocabulary than reference summaries, though content may still be accurate

### Test 2 Analysis: Prediction Bias

Analysis of Test 2 revealed a significant bias across all models: every model heavily over-predicted "sustained" outcomes. While ground truth showed a roughly 50/50 split between dismissed and sustained rulings, models predicted sustained 82-99% of the time.

| Model | Predicted Dismissed | Predicted Sustained |
|-------|---------------------|---------------------|
| Ground Truth | 51% | 49% |
| Gemini | 1% | 99% |
| Grok | 10% | 90% |
| Claude Opus | 10% | 90% |
| GPT-5.2 | 11% | 89% |
| Perplexity | 18% | 82% |

This bias likely stems from LLMs being influenced by the plaintiff-favorable framing inherent in complaints, rather than applying the rigorous pleading standards courts actually use.

### Test 3: Scienter-Focused Prompt Engineering

To address the sustained-prediction bias, Test 3 introduced a prompt specifically designed to apply the heightened pleading standards of the Private Securities Litigation Reform Act (PSLRA).

#### The Scienter-Focused Approach

The Test 3 prompt instructs models to:

1. **Scrutinize Scienter Allegations**: The PSLRA requires plaintiffs to plead with *particularity* facts giving rise to a *strong inference* of fraudulent intent. The prompt directs models to dismiss claims where:
   - Scienter allegations are conclusory (e.g., "defendants knew or should have known")
   - There are no specific facts showing defendant's actual knowledge of falsity
   - The complaint relies on "fraud by hindsight" rather than contemporaneous knowledge
   - Innocent explanations are equally or more plausible

2. **Apply Derivative Claim Logic**: If Section 10(b)/Rule 10b-5 claims are dismissed, Section 20(a) control person liability claims must also be dismissed as they are derivative of the underlying violation.

#### Test 3 Results

| Rank | Model | Test 2 Accuracy | Test 3 Accuracy | Improvement |
|------|-------|-----------------|-----------------|-------------|
| 1 | **GPT-5.2** | 35.7% | **66.0%** | **+30.3%** |
| 2 | **Grok** | 47.2% | **65.5%** | **+18.3%** |
| 3 | Perplexity | 46.5% | 60.4% | +13.9% |
| 4 | Claude Opus | 35.8% | 57.4% | +21.6% |
| 5 | Gemini | 40.8% | 41.9% | +1.1% |

#### Prediction Bias Correction

| Model | Test 2 Dismissed % | Test 3 Dismissed % | Ground Truth: 51% |
|-------|--------------------|--------------------|-------------------|
| GPT-5.2 | 11% | 32% | Improved |
| Grok | 10% | 32% | Improved |
| Perplexity | 18% | 37% | Improved |
| Claude Opus | 10% | 18% | Slightly improved |
| Gemini | 1% | 10% | Marginally improved |

**Key Findings (Test 3):**
- **Prompt engineering dramatically improved ruling accuracy** - GPT-5.2 improved by 30 percentage points
- **GPT-5.2** became the top performer at 66% ruling accuracy (up from 35.7%)
- **Grok** maintained strong performance at 65.5%
- The scienter-focused prompt partially corrected the sustained-prediction bias
- **Gemini** showed minimal response to the prompt modification (+1.1%)

### Overall Assessment

For the primary goal of predicting motion-to-dismiss outcomes:
1. **GPT-5.2** with Test 3 prompt achieves the best ruling accuracy (66.0%)
2. **Grok** with Test 3 prompt is a close second (65.5%)
3. **Perplexity** with Test 3 prompt performs well (60.4%)
4. **Prompt engineering proved more impactful than fine-tuning** for this task

The results demonstrate that instructing models to apply specific legal standards (PSLRA scienter requirements) significantly improves their ability to predict judicial outcomes. The simple addition of legal framework guidance in the prompt outperformed more complex approaches like few-shot examples or multi-stage reasoning.

## Finetuning

We fine-tuned GPT-4.1 using 50 complaint-order pairs with full complaint texts with background summaries and rulings as output. Two versions were trained:
- **GPT-4.1-finetuned**: Initial fine-tuning run
- **GPT-4.1-finetuned-v2**: Second iteration with refined training data

### Finetuned Model Ruling Prediction Results

| Model | Test 2 Accuracy | Test 3 Accuracy | Improvement |
|-------|-----------------|-----------------|-------------|
| GPT-4.1-finetuned | 7.0% | 63.5% | +56.5% |
| GPT-4.1-finetuned-v2 | 21.3% | 60.6% | +39.3% |

### Finetuned Model Summary Quality Metrics (Test 2)

| Model | ROUGE-1 | ROUGE-L | BLEU | METEOR | BERTScore | Success Rate |
|-------|---------|---------|------|--------|-----------|--------------|
| GPT-4.1-finetuned | 0.358 | 0.148 | 0.053 | 0.220 | 0.019 | 50% (13/26) |
| GPT-4.1-finetuned-v2 | 0.457 | 0.199 | 0.090 | 0.336 | 0.044 | 38% (10/26) |
| *GPT-5.2 (baseline)* | *0.448* | *0.172* | *0.071* | *0.252* | *-0.004* | *88% (22/25)* |

### Finetuned Model Summary Quality Metrics (Test 3)

| Model | ROUGE-1 | ROUGE-L | BLEU | METEOR | BERTScore | Success Rate |
|-------|---------|---------|------|--------|-----------|--------------|
| GPT-4.1-finetuned-v2 | 0.282 | 0.131 | 0.050 | 0.192 | -0.044 | 92% (24/26) |
| *GPT-5.2 (baseline)* | *0.394* | *0.151* | *0.063* | *0.222* | *-0.003* | *88% (22/25)* |

**Key Findings (Fine-tuning):**
- **High failure rate on Test 2**: Fine-tuned models failed to produce valid outputs for 50-62% of cases, compared to only 12% for GPT-5.2
- **Test 3 dramatically improved reliability**: GPT-4.1-finetuned-v2 success rate jumped from 38% to 92% with the scienter-focused prompt
- **Competitive summary quality when successful**: On successful cases, finetuned-v2 achieved higher ROUGE-1 (0.457) and METEOR (0.336) than GPT-5.2, suggesting the model learned effective summarization patterns
- **Fine-tuning alone is insufficient**: Without explicit legal framework guidance (Test 3 prompt), fine-tuned models achieved only 7-21% ruling accuracy
- **Prompt engineering remains critical**: The Test 3 prompt improved fine-tuned model accuracy by 40-57 percentage points, demonstrating that proper prompting is essential even for fine-tuned models
- The combination of fine-tuning + Test 3 prompt achieved similar accuracy (60-63%) to base GPT-5.2 + Test 3 (66%), but with lower reliability

## Conclusion & Future Work

### Conclusions

1. **LLMs exhibit systematic bias** in legal prediction tasks, heavily favoring plaintiff-side outcomes when reviewing complaints
2. **Prompt engineering is highly effective** for legal reasoning tasks - incorporating specific legal standards (PSLRA scienter requirements) dramatically improved accuracy from ~40% to ~66%
3. **Fine-tuning has limited benefit** when prompts don't include proper legal framework guidance
4. **GPT-5.2 and Grok** emerged as the top performers for motion-to-dismiss prediction when using the scienter-focused prompt
5. **Surface metrics (ROUGE, BLEU) don't correlate** with ruling prediction accuracy - models can produce similar-looking summaries while reaching different legal conclusions

### Future Work

- Test if using a RAG approach using a legal based transformer to generate embeddings helps improve performance
- Test additional prompt variations incorporating other PSLRA elements (loss causation, materiality, forward-looking statement safe harbor)
- Investigate whether fine-tuning on the Test 3 prompt format improves results further
- Test multi-stage reasoning approaches (chain-of-thought prompting)

