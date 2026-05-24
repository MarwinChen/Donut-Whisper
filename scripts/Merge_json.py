import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple JSON list files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files. Each file must contain a list.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    merged_data = []
    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError(f"{input_path} must contain a JSON list.")
        merged_data.extend(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(merged_data, fp, ensure_ascii=False, indent=4)
    print(f"Wrote {len(merged_data)} items to {output_path}")


if __name__ == "__main__":
    main()
