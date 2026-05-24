import os
import subprocess
import re
from tqdm import tqdm

# 目标文件夹
folder = r"/hdd/srt11/data_8844/Video_with_TGW/S02-p"
destination_folder = r"/hdd/srt11/data_8844/S02_srt"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历所有 mkv 文件
for filename in os.listdir(folder):
    if filename.endswith(".mkv"):
        # 匹配 SxxExx
        match = re.search(r"S(\d{2})E(\d{2})", filename, re.IGNORECASE)
        if not match:
            continue
        season = match.group(1)
        episode = match.group(2)
        base_name = f"S{season}E{episode}"

        mkv_path = os.path.join(folder, filename)

        # 先用 ffprobe 查找字幕流编号
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=index", "-of", "csv=p=0",
            mkv_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        subtitle_streams = result.stdout.strip().split('\n')

        # 提取每个字幕流
        for idx, stream_index in enumerate(tqdm(subtitle_streams, desc=f"提取字幕流 {base_name}")):
            if not stream_index.strip():
                continue
            out_path = os.path.join(destination_folder, f"{base_name}.srt")  # 输出为srt格式
            extract_cmd = [
                "ffmpeg", "-y", "-i", mkv_path, "-map", f"0:s:{idx}", "-c:s", "srt", out_path
            ]
            subprocess.run(extract_cmd)

print("字幕提取完成！")