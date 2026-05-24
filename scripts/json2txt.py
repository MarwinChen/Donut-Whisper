import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert results JSON to SCTK ref/pred txt files.")
    parser.add_argument("--json_path", required=True, help="Path to results_final.json.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to the JSON file directory.")
    parser.add_argument("--char_level", action="store_true", help="Write characters separated by spaces.")
    return parser.parse_args()


def format_text(text, char_level):
    text = "" if text is None else str(text)
    text = text.split("</s>")[0].replace("<s>", "").strip()
    return " ".join(text) if char_level else text


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sctk").mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref_txt_path = output_dir / "refs.txt"
    pred_txt_path = output_dir / "preds.txt"
    with open(ref_txt_path, "w", encoding="utf-8") as ref_f, open(pred_txt_path, "w", encoding="utf-8") as pred_f:
        for idx, item in enumerate(data, 1):
            ref_f.write(f"{format_text(item.get('ref', ''), args.char_level)} ({idx})\n")
            pred_f.write(f"{format_text(item.get('pred', ''), args.char_level)} ({idx})\n")

    print(f"Wrote {ref_txt_path}")
    print(f"Wrote {pred_txt_path}")


if __name__ == "__main__":
    main()
