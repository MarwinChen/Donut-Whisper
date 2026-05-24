import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether local audio/video files in a dataset JSON can be opened.")
    parser.add_argument("--data_path", required=True, help="Dataset JSON path.")
    parser.add_argument("--bad_files", default="bad_files.txt", help="Output path for failed samples.")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    return parser.parse_args()


def check_sample(args):
    idx, sample = args
    errors = []

    audio_file = sample.get("audio")
    if audio_file:
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"{audio_file} not found")
            sf.read(audio_file)
        except Exception as exc:
            errors.append((idx, "audio", audio_file, str(exc)))

    video_file = sample.get("video")
    if video_file:
        try:
            if not os.path.exists(video_file):
                raise FileNotFoundError(f"{video_file} not found")
            cap = cv2.VideoCapture(video_file)
            ret, _ = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Cannot read first frame")
        except Exception as exc:
            errors.append((idx, "video", video_file, str(exc)))

    return errors


def main():
    args = parse_args()
    with open(args.data_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    error_list = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_sample, (idx, sample)) for idx, sample in enumerate(data)]
        for future in as_completed(futures):
            errs = future.result()
            if errs:
                error_list.extend(errs)
                for err in errs:
                    print(f"[{err[1].upper()}] Error at idx {err[0]}: {err[2]} - {err[3]}")

    print(f"\nFound {len(error_list)} unreadable audio/video files.")
    if error_list:
        bad_files = Path(args.bad_files)
        bad_files.parent.mkdir(parents=True, exist_ok=True)
        with open(bad_files, "w", encoding="utf-8") as fp:
            for idx, file_type, path, err in error_list:
                fp.write(f"{idx}\t{file_type}\t{path}\t{err}\n")
        print(f"Details written to {bad_files}")


if __name__ == "__main__":
    main()
