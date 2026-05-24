import argparse
import json
from pathlib import Path

try:
    from .wer import WER
    from .whisper_normalizer import EnglishTextNormalizer
    from .cer import CER
    from .chinese_normalizer import ChineseTextNormalizer
except ImportError:
    from wer import WER
    from whisper_normalizer import EnglishTextNormalizer
    from cer import CER
    from chinese_normalizer import ChineseTextNormalizer


def strip_special_tokens(text):
    if text is None:
        return ""
    return str(text).split("</s>")[0].replace("<s>", "").strip()


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def build_normalizer(language):
    if language == "zh":
        return ChineseTextNormalizer()
    if language == "en":
        return EnglishTextNormalizer()
    raise ValueError(f"Unsupported language: {language}")


def build_metric(metric):
    if metric == "cer":
        return CER(), "characters"
    if metric == "wer":
        return WER(), "words"
    raise ValueError(f"Unsupported metric: {metric}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR JSON output with CER or WER.")
    parser.add_argument("--json_file", required=True, help="Path to results_final.json.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--metric", choices=["cer", "wer"], default="cer")
    return parser.parse_args()


def main():
    args = parse_args()
    json_file = Path(args.json_file)
    data = load_predictions(json_file)

    normalizer = build_normalizer(args.language)
    criterion, unit_name = build_metric(args.metric)

    refs = [normalizer(strip_special_tokens(item.get("ref", ""))) for item in data]
    preds = [normalizer(strip_special_tokens(item.get("pred", ""))) for item in data]

    subs, dels, ins, total = criterion._compute(predictions=preds, references=refs)
    error_rate = (subs + dels + ins) / total * 100 if total else 0.0

    print(f"{len(preds)} samples")
    print(f"Substitutions: {subs}, deletions: {dels}, insertions: {ins}, total {unit_name}: {total}")
    print(f"{args.metric.upper()}: {error_rate:.2f}%")


if __name__ == "__main__":
    main()
