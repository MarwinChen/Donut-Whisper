import os
from tqdm import tqdm
import subprocess
import concurrent.futures

vtt_dir = '/hdd/srt11/data_8844/YouTube/vtt_original'
ass_dir = '/hdd/srt11/data_8844/YouTube/ass'
os.makedirs(ass_dir, exist_ok=True)

vtt_files = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')]

def process(i):
    vtt_file = vtt_files[i]
    vtt_path = os.path.join(vtt_dir, vtt_file)
    ass_file = os.path.splitext(vtt_file)[0] + '.ass'
    ass_path = os.path.join(ass_dir, ass_file)
    cmd = [
        'ffmpeg', '-y',
        '-i', vtt_path,
        ass_path
    ]
    print(f"正在转换: {vtt_file} -> {ass_file}")
    subprocess.run(cmd)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    responses = list(tqdm(executor.map(process, range(len(vtt_files)))))

print("全部VTT字幕已转换为ASS格式！")