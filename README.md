# LLM-Complaint-Summary-Test
## Introduction

This readme and code are intended to define and evaluate benchmarks for assessing the performance of large language models (LLMs) on legal-document summarization tasks. In particular, the focus is on complex civil complaints, which present challenges that are not well captured by existing general-purpose summarization benchmarks.

Prior scholarly work has shown that LLM performance degrades as context length increases and as tasks move beyond surface-level or literal matching. See, e.g., *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (https://arxiv.org/abs/2502.05167). Legal complaints exacerbate these issues: they are lengthy, fact-dense, and often rely on layered narratives to support multiple (and sometimes contradictory) theories of liability.`

Accordingly, this work concentrates on complaints that are likely to stress current LLM capabilities due to both their length and structural complexity. The goal is not to demonstrate isolated successes, but to provide a realistic assessment of current LLMs capabilities. To that end, the models are evaluated through two distinct tests. The first test assesses whether an LLM can reliably extract simple, objective information from a complaint. The second test evaluates whether an LLM can objectively summarize the complaint and assess the likelihood that each asserted cause of action would survive a motion to dismiss.  Finally, given the nature of these tests, we will see if finetuning models will meaningfully improve performance of the LLM. 
## The Tests

The evaluation consists of two tests designed to assess distinct capabilities of large language models in the context of complex legal complaints.
### Test One: Objective Information Extraction

The first test evaluates an LLM’s ability to extract basic, objective data points from complaints filed in securities and commodities litigation. Specifically, the models are evaluated on whether they can correctly identify:

1. The plaintiff(s)
2. The defendant(s)
3. Whether any defendant is a publicly traded company and, if so, the corresponding ticker symbol
4. The causes of action asserted against each defendant
5. The key factual allegations supporting the asserted causes of action

Items (1) through (4) are objectively scored. Each correctly identified plaintiff or defendant receives one point, and each correctly identified cause of action receives one point, regardless of whether the cause of action is stated verbatim or paraphrased.

Item (5) is evaluated by comparing the LLM’s output to the source complaint using automated metrics designed to assess accuracy and factual faithfulness.
### Test Two: Complaint Summarization and Dismissal Assessment

The second test evaluates an LLM’s ability to objectively summarize a complaint and assess the likelihood that each asserted cause of action would survive a motion to dismiss. For this test, LLM-generated summaries are compared to the court’s characterization of the complaint as reflected in written decisions resolving motions to dismiss.

As in the first test, model outputs are evaluated using the same set of metrics to assess accuracy and faithfulness.

### Fine-Tuning and Re-Evaluation

Based on performance across both tests, the top-performing models are selected for further fine-tuning. Fine-tuning is performed using domain-specific embeddings derived from Legal-BERT. The fine-tuned models are then re-evaluated using the same testing framework to assess whether fine-tuning produces measurable improvements in extraction accuracy and summarization faithfulness.

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

