#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess


def get_duration_seconds_ffprobe(file_path: str) -> float:
    """Return duration in seconds using ffprobe. Raise RuntimeError if fails."""
    try:
        # ffprobe prints duration in seconds as a float
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        out = result.stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"ffprobe 获取时长失败: {file_path} ({exc})")


def format_hms(total_seconds: float) -> str:
    seconds = int(round(total_seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="统计文件夹内所有 .mp4 的总时长")
    parser.add_argument(
        "directory",
        type=str,
        help="待统计的目录（将递归搜索）",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)
    if not os.path.isdir(root_dir):
        print(f"目录不存在: {root_dir}", file=sys.stderr)
        sys.exit(1)

    total_seconds = 0.0
    file_count = 0
    failed_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".mkv"):
                file_path = os.path.join(dirpath, name)
                try:
                    dur = get_duration_seconds_ffprobe(file_path)
                    total_seconds += dur
                    file_count += 1
                except RuntimeError as exc:
                    failed_files.append((file_path, str(exc)))

    print(f"目录: {root_dir}")
    print(f"视频数量: {file_count}")
    print(f"总时长(秒): {total_seconds:.3f}")
    print(f"总时长(HH:MM:SS): {format_hms(total_seconds)}")

    if failed_files:
        print("\n以下文件获取时长失败(已跳过):")
        for path, reason in failed_files:
            print(f"- {path}: {reason}")


if __name__ == "__main__":
    main()


