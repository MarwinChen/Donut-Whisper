import argparse
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path

from tqdm import tqdm


def time_to_float(time_str):
    parts = re.split(r"[:,. ]", time_str)
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    milliseconds = float(parts[3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Cut video/audio clips from SRT timestamps and build dataset JSON.")
    parser.add_argument("--srt_root", required=True, help="Directory containing .srt files.")
    parser.add_argument("--video_root", required=True, help="Directory containing source videos.")
    parser.add_argument("--output_dir", required=True, help="Directory for generated video/audio clips.")
    parser.add_argument("--output_json", required=True, help="Output dataset JSON path.")
    parser.add_argument("--video_ext", default=".mkv", help="Source video extension.")
    parser.add_argument("--srt_filter", default=None, help="Only process SRT files whose names contain this string.")
    parser.add_argument(
        "--subtitle_regex",
        default=r'<font color="#eba862">(.*?)</font>',
        help="Regex used to extract subtitle text. Use an empty string to keep raw subtitle text.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run_ffmpeg", action="store_true", help="Actually write clipped media files.")
    return parser.parse_args()


def extract_subtitle(text, subtitle_regex):
    text = text.strip()
    if subtitle_regex:
        match = re.search(subtitle_regex, text)
        if not match:
            return None
        return match.group(1).strip()
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_srt_file(srt_path, subtitle_regex):
    with open(srt_path, "r", encoding="utf-8-sig", errors="ignore") as file:
        content = file.read()

    entries = re.split(r"\n\s*\n", content.strip())
    result = []
    for entry in entries:
        lines = [line.strip() for line in entry.splitlines() if line.strip()]
        time_idx = next((idx for idx, line in enumerate(lines) if " --> " in line), None)
        if time_idx is None or time_idx + 1 >= len(lines):
            continue

        time_range = lines[time_idx].split(" --> ")
        start_time = time_to_float(time_range[0])
        end_time = time_to_float(time_range[1])
        subtitle = extract_subtitle("".join(lines[time_idx + 1:]), subtitle_regex)
        if subtitle:
            result.append((start_time, end_time, subtitle))
    return result


def merge_segments(segments):
    if not segments:
        return []

    merged = []
    start, end, text = segments[0]
    count = 1
    for next_start, next_end, next_text in segments[1:]:
        if not text.endswith((".", "?", "!")):
            text += " " + next_text
            end = next_end
            count += 1
        else:
            merged.append((start, end, text, count))
            start, end, text = next_start, next_end, next_text
            count = 1
    merged.append((start, end, text, count))
    return merged


def build_tasks(args):
    tasks = []
    srt_root = Path(args.srt_root)
    video_root = Path(args.video_root)

    for srt_path in sorted(srt_root.glob("*.srt")):
        if args.srt_filter and args.srt_filter not in srt_path.name:
            continue
        segments = merge_segments(parse_srt_file(srt_path, args.subtitle_regex))
        video_path = video_root / f"{srt_path.stem}{args.video_ext}"
        for start_time, end_time, text, count in segments:
            tasks.append((video_path, start_time, end_time, text, count))
    return tasks


def run_command(command):
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def process_task(task, output_dir, run_ffmpeg):
    video_path, start_time, end_time, text, count = task
    video_name = video_path.stem
    video_out_dir = Path(output_dir) / "video" / video_name
    audio_out_dir = Path(output_dir) / "audio" / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = video_out_dir / f"{video_name}-{round(start_time)}-{round(end_time)}.mp4"
    output_audio_path = audio_out_dir / f"{video_name}-{round(start_time)}-{round(end_time)}.wav"

    if run_ffmpeg:
        video_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(output_video_path),
        ]
        audio_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-i",
            str(video_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_audio_path),
        ]
        run_command(video_cmd)
        run_command(audio_cmd)

    return {
        "video": str(output_video_path),
        "audio": str(output_audio_path),
        "text": text,
        "image_cnt": count,
    }


def main():
    args = parse_args()
    tasks = build_tasks(args)
    print(f"Found {len(tasks)} segments.")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_task, task, args.output_dir, args.run_ffmpeg)
            for task in tasks
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)
    print(f"Wrote {len(results)} items to {output_json}")


if __name__ == "__main__":
    main()
