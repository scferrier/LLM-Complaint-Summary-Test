import evaluate

from summac.model_summac import SummaCConv

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
summac = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

qfact = LLMTestCase(
    input="...",
    actual_output="Model’s generated summary",
    expected_output="Reference summary"
)


#blleu usage results = bleu.compute(predictions=predictions, references=references)
reference = """Your ground-truth / correct summary here..."""
prediction = """The LLM's generated summary here..."""

"""
# ROUGE (overlap-based)
rouge_result = rouge.compute(
    predictions=[prediction],
    references=[reference],
    use_stemmer=True,   # helps a bit with inflections
)
print("ROUGE:", rouge_result)
"""
"""
# METEOR (alignment-based, tends to correlate better than ROUGE in some cases)
meteor_result = meteor.compute(
    predictions=[prediction],
    references=[reference],
)
print("METEOR:", meteor_result)
"""

""" BERTScore (semantic similarity using contextual embeddings)
bertscore_result = bertscore.compute(
    predictions=[prediction],
    references=[reference],
    lang="en",          # set language
    rescale_with_baseline=True,  # makes numbers more interpretable across runs/models
)
# bertscore returns lists (one per example)
print("BERTScore (P/R/F1):",
      bertscore_result["precision"][0],
      bertscore_result["recall"][0],
      bertscore_result["f1"][0])
"""


""" Example usage of evaluation metrics and SummaC models  
summac_score = summac.score([document], [summary1])
print("[Summary 1] SummacConv score: %.3f" % (score_conv1["scores"][0]))
"""


""" use of GEval to measure correctness metric 
correctness = GEval(
    name="Correctness",
    criteria="Correctness - determine if actual output matches expected output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    strict_mode=True
)

correctness.measure(qfact)
print(correctness.score, correctness.reason)
 """