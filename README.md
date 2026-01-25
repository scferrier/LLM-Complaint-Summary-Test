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

## Metrics  Used to Evaluate Performance

To evaluate the accuracy and faithfulness of LLM-generated outputs for the non-objective outputs, this work employs a combination of established automated summarization metrics: ROUGE, BLEU, METEOR, BERTScore, SummaC, and QAFactEval. These metrics are selected to capture complementary aspects of similarity, semantic alignment, and factual consistency between LLM outputs and reference texts. For an overview of these metrics and their respective strengths and limitations. See Shannon Gallagher, Swati Rallapalli, and Tyler Brooks, *Evaluating LLMs for Text Summarization: An Introduction*, SEI Blog (Apr. 7, 2025) (https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction); Guide to Evaluating Large Language Models (LLMs), Arockia Liborious, (June 12, 2024) https://arockialiborious.com/f/guide-to-evaluating-large-language-models-llms

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
## Models, Libraries, & Code
#TODO
## Model Results
#TODO
## Finetuning
#TODO
## Finetuned Model Results
#TODO
## Conclusion & Potential Future 
#TODO

