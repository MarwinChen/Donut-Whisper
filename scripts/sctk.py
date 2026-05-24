import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run sclite on ref/pred text files.")
    parser.add_argument("--pred_txt", required=True, help="Prediction txt path.")
    parser.add_argument("--ref_txt", required=True, help="Reference txt path.")
    parser.add_argument("--out_dir", required=True, help="SCTK output directory.")
    parser.add_argument("--sclite", default="sclite", help="Path to sclite executable.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.sclite,
        "-i",
        "wsj",
        "-e",
        "utf-8",
        "-h",
        args.pred_txt,
        "-r",
        args.ref_txt,
        "-o",
        "all",
        "-O",
        str(out_dir),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
