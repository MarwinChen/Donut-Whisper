import datasets
from jiwer import compute_measures
import evaluate

_CITATION = """\
@inproceedings{inproceedings,
    author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
    year = {2004},
    month = {01},
    pages = {},
    title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
"""

_DESCRIPTION = """\
Character error rate (CER) is a common metric for evaluating automatic speech recognition systems for languages without clear word boundaries, such as Chinese.

CER is derived from the Levenshtein distance, working at the character level instead of the word level. CER is particularly useful for Chinese text evaluation as it doesn't require word segmentation.

Character error rate can be computed as:

CER = (S + D + I) / N = (S + D + I) / (S + D + C)

where

S is the number of character substitutions,
D is the number of character deletions, 
I is the number of character insertions,
C is the number of correct characters,
N is the number of characters in the reference (N=S+D+C).

This value indicates the average number of errors per reference character. The lower the value, the better the
performance of the ASR system with a CER of 0 being a perfect score.
"""

_KWARGS_DESCRIPTION = """
Compute CER score of transcribed segments against references.

Args:
    references: List of references for each speech input.
    predictions: List of transcriptions to score.
    concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute CER iteratively.

Returns:
    (float): the character error rate

Examples:

    >>> predictions = ["这是预测结果", "另一个样本"]
    >>> references = ["这是参考文本", "另一个例子"] 
    >>> cer = CER()
    >>> cer_score = cer.compute(predictions=predictions, references=references)
    >>> print(cer_score)
    0.25
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CER(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
            ],
        )

    def _compute(self, predictions=None, references=None, concatenate_texts=False):
        if concatenate_texts:
            # Convert to character level for jiwer
            ref_chars = [" ".join(list(ref)) for ref in references]
            pred_chars = [" ".join(list(pred)) for pred in predictions]
            return compute_measures(ref_chars, pred_chars)["wer"]
        else:
            total = 0
            subs = 0
            dels = 0
            ins = 0
            for prediction, reference in zip(predictions, references):
                # Convert to character level
                ref_chars = " ".join(list(reference))
                pred_chars = " ".join(list(prediction))
                if ref_chars == "":
                    continue
                measures = compute_measures(ref_chars, pred_chars)
                
                subs += measures["substitutions"]
                dels += measures["deletions"] 
                ins += measures["insertions"]
                total += measures["substitutions"] + measures["deletions"] + measures["hits"]
            return subs, dels, ins, total 