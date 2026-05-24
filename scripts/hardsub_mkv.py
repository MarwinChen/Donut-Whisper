import os
import subprocess
from tqdm import tqdm
import concurrent.futures

mkv_dir = '/hdd/srt11/data_8844/YouTube/mkv'
ass_dir = '/hdd/srt11/data_8844/YouTube/ass'
output_dir = '/hdd/srt11/data_8844/YouTube/mkv_hardsub'
os.makedirs(output_dir, exist_ok=True)

mkv_files = [f for f in os.listdir(mkv_dir) if f.endswith('.mkv')]
ass_files = [f for f in os.listdir(ass_dir) if f.endswith('.ass')]

# 建立ass文件的前五位索引
ass_dict = {f[:5]: f for f in ass_files}

def process(i):
    mkv = mkv_files[i]
    key = mkv[:5]
    if key in ass_dict:
        mkv_path = os.path.join(mkv_dir, mkv)
        ass_path = os.path.join(ass_dir, ass_dict[key])
        output_path = os.path.join(output_dir, mkv)
        cmd = [
            'ffmpeg', '-y',
            '-i', mkv_path,
            '-vf', f"subtitles='{ass_path}'",
            '-c:a', 'copy',
            output_path
        ]
        print(f"正在烧录: {mkv} + {ass_dict[key]}")
        if not os.path.exists(output_path):
            subprocess.run(cmd)
    else:
        print(f"未找到对应ASS字幕: {mkv}")

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    responses = list(tqdm(executor.map(process, range(len(mkv_files)))))

print("全部视频已烧录硬字幕！")