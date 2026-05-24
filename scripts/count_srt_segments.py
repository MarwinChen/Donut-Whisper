#!/usr/bin/env python3
import os
import sys
import argparse


def count_srt_segments(file_path: str) -> int:
    """Count SRT subtitle segments by detecting block starts.

    A robust heuristic: a block starts when a line is only digits and the next
    non-empty line contains the arrow '-->'. This avoids counting noise lines.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        return 0

    count = 0
    total = len(lines)
    i = 0
    while i < total:
        line = lines[i].strip()
        if line.isdigit():
            # find next non-empty line to check for timecode
            j = i + 1
            while j < total and lines[j].strip() == "":
                j += 1
            if j < total and "-->" in lines[j]:
                count += 1
                # fast-forward to the end of this block (optional)
                # move until blank line separating blocks
                k = j + 1
                while k < total and lines[k].strip() != "":
                    k += 1
                i = k
                continue
        i += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="统计目录内所有 .srt 字幕段数量之和（递归）")
    parser.add_argument("directory", type=str, help="待统计的目录")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)
    if not os.path.isdir(root_dir):
        print(f"目录不存在: {root_dir}", file=sys.stderr)
        sys.exit(1)

    total_segments = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith('.srt'):
                file_path = os.path.join(dirpath, name)
                segs = count_srt_segments(file_path)
                total_segments += segs
                file_count += 1

    print(f"目录: {root_dir}")
    print(f"SRT 文件数: {file_count}")
    print(f"字幕段总数: {total_segments}")


if __name__ == "__main__":
    main()


