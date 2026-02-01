# LLM-Complaint-Summary-Test
## Introduction

This readme and code are intended to define and evaluate benchmarks for assessing the performance of large language models (LLMs) on legal-document summarization tasks. In particular, the focus is on complex civil complaints, which present challenges that are not well captured by existing general-purpose summarization benchmarks.

Prior scholarly work has shown that LLM performance degrades as context length increases and as tasks move beyond surface-level or literal matching. See, e.g., *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (https://arxiv.org/abs/2502.05167). Legal complaints exacerbate these issues: they are lengthy, fact-dense, and often rely on layered narratives to support multiple (and sometimes contradictory) theories of liability.`

Accordingly, this work concentrates on complaints that are likely to stress current LLM capabilities due to both their length and structural complexity. The goal is not to demonstrate isolated successes, but to provide a realistic assessment of current LLMs capabilities. To that end, the models are evaluated through two distinct tests. The first test assesses whether an LLM can reliably extract simple, objective information from a complaint. The second test evaluates whether an LLM can objectively summarize the complaint and assess the likelihood that each asserted cause of action would survive a motion to dismiss.  Finally, given the nature of these tests, we will see if finetuning models will meaningfully improve performance of the LLM. 
## The Tests

To evaluate large language models’ ability to summarize long and complex documents, we employ two complementary approaches: (1) direct long-context summarization and (2) retrieval-augmented generation (“RAG”)–based summarization. These approaches are tested separately to isolate model capabilities from retrieval and embedding effects.

### 1. Direct Long-Context Summarization (Baseline)

In the first test, each model is evaluated on its ability to summarize long documents without retrieval or chunk-based augmentation.

Source PDFs are collected and converted to text.The extracted text is cleaned and normalized to remove artifacts such as headers, footers, page numbers, and filing stamps, while preserving logical section boundaries The same cleaned text is provided directly to each model as input, subject only to the model’s maximum context window.Each model generates a summary according to a fixed prompt and output specification.

The resulting summaries are evaluated against predefined benchmarks. This test isolates the model’s native long-context reasoning and summarization ability, independent of embeddings, chunking, or retrieval strategies, and serves as the baseline for all subsequent comparisons.

### 2. Retrieval-Augmented Summarization (RAG)

In the second test, models are evaluated using a RAG pipeline to assess summarization performance when document context is selectively retrieved rather than fully provided.

Cleaned document text is segmented into fixed-size chunks using a consistent chunking strategy. Two embedding strategies are evaluated:

 1. general-purpose embedding model (e.g., OpenAI embeddings), and

 2. a transformer-based embedding model trained on legal-domain text.

Chunk embeddings and associated metadata are stored in a vector database For each document, targeted retrieval queries are issued to retrieve relevant chunks from the vector database. Retrieved text chunks are provided to the model along with a fixed summarization prompt. The model generates a summary based solely on the retrieved context.

Summaries are evaluated using the same scoring framework as the direct summarization test. This test measures the combined performance of retrieval quality and downstream summarization, enabling comparison of embedding strategies and their impact on summary accuracy, coverage, and faithfulness.

### 3. Comparative Evaluation and Fine-Tuning

Results from the direct summarization and RAG-based tests are compared to distinguish intrinsic model summarization capability from retrieval-assisted performance, and assess whether domain-specific embeddings materially improve summarization outcomes.

Where appropriate, selected models may be fine-tuned using a separate training corpus, and the above tests are repeated using a strict train/dev/test split to evaluate the impact of fine-tuning on both direct and RAG-based summarization performance.

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

### Overall Assessment

For the primary goal of predicting motion-to-dismiss outcomes:
1. **Perplexity (Sonar Pro)** is the best performer with 0.436 Ruling F1
2. **Grok** is a close second with balanced performance across metrics
3. **GPT-5.2** excels at extraction tasks but underperforms on ruling prediction

## Finetuning

Based on the evaluation results, **Perplexity (Sonar Pro)** was selected for fine-tuning due to its superior performance on ruling prediction tasks. The model will be fine-tuned using the 100 complaint-order pairs collected for training, with the goal of improving both summary quality and ruling prediction accuracy.

## Finetuned Model Results
#TODO - Results will be added after fine-tuning is complete

## Conclusion & Potential Future Work
#TODO

