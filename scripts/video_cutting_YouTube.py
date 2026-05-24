import json
import os
import re
from tqdm import tqdm
import subprocess
import concurrent.futures
from concurrent.futures import as_completed

def time_to_float(time_str):
    parts = re.split(r'[:,. ]', time_str)
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    milliseconds = float(parts[3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

# srt_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/SRT_kurz"
# video_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Video_kurz"
# output_dir = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Cut2"

srt_root = "/hdd/srt11/data_8844/S02_srt/"
video_root = "/hdd/srt11/data_8844/Video_with_TGW/S02-p/"
output_dir = "/hdd/srt11/data_8844/CutS02"

output_json = "/hdd/srt11/data_8844/TrainS02.json"

all_srt = {}
srt_files = os.listdir(srt_root)
video_files = os.listdir(video_root)
for srt_file in srt_files:
    if int(srt_file[4:6]) <= 14:
        with open(os.path.join(srt_root, srt_file), 'r') as file:
            content = file.read()
            entries = re.split(r'\n{2,}', content.strip())[1:]
            result = []
            for entry in entries:
                lines = entry.split('\n')
                if ' --> ' in lines[0]:
                    time_range = lines[0].split(' --> ')
                    start_time = time_to_float(time_range[0])
                    end_time = time_to_float(time_range[1])
                    match = ' '.join(lines[1:])
                else:
                    time_range = lines[1].split(' --> ')
                    start_time = time_to_float(time_range[0])
                    end_time = time_to_float(time_range[1])
                    match = ' '.join(lines[2:])
                if match:
                    subtitle = match
                    result.append((start_time, end_time, subtitle))
        all_srt[srt_file] = result

for k in all_srt.keys():
    merge_v = []
    v = all_srt[k]
    start, end, text = v[0][0], v[0][1], v[0][2]
    cnt = 1
    for i in range(1, len(v)):
        if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
            text += " " + v[i][2]
            end = v[i][1]
            cnt += 1
        else:
            merge_v.append((start, end, text, cnt))
            start, end, text = v[i][0], v[i][1], v[i][2]
            cnt = 1
    merge_v.append((start, end, text, cnt))
    all_srt[k] = merge_v

video_lst = []
time_lst = []
for k, v in all_srt.items():
    video_lst += [os.path.join(video_root, f'{k.split(".")[0]}.mkv')] * len(v)
    time_lst += v

res = []

def process(i):
    start_time, end_time, text, cnt = time_lst[i]
    video_path = video_lst[i]
    video_name = os.path.basename(video_path).split('.')[0]
    os.makedirs(os.path.join(output_dir, "video", video_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio", video_name), exist_ok=True)

    output_video_path = os.path.join(output_dir, "video", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.mp4")
    output_audio_path = os.path.join(output_dir, "audio", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.wav")

    video_cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', video_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_video_path
    ]
    # 重定向标准输出和标准错误输出到空设备，避免显示信息
    # if not os.path.exists(output_video_path):
    # subprocess.run(video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    # 提取音频并转换为 16kHz 单声道
    audio_cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', video_path,
        '-ar', '16000',
        '-ac', '1',
        output_audio_path
    ]
    # 重定向标准输出和标准错误输出到空设备，避免显示信息
    # if not os.path.exists(output_audio_path):
    # subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    res.append({
        "video": output_video_path,
        "audio": output_audio_path,
        "text": text,
        "image_cnt": cnt
    })

    return output_video_path, output_audio_path, text

print(len(time_lst))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process, i) for i in range(len(time_lst))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            resi = future.result()
        except Exception as e:
            print(f"有任务出错: {e}")
print("Done!")

# res = []
# for i in range(len(time_lst)):
#     res.append({
#         "video": output_video_path,
#         "audio": output_audio_path,
#         "text": text,
#         "image_cnt": cnt
#     })

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

# print(len(res))
# print(output_json)