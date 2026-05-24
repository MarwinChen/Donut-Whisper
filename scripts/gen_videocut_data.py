import json
import os
import re

def time_to_float(time_str):
    parts = re.split(r'[:,]', time_str)
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    milliseconds = float(parts[3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

srt_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/SRT_kurz"
video_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Video_kurz"
output_dir = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Cut2"
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/video_film_train.json"

select_idx = range(1, 21) # [21] # [22, 23]

all_srt = {}
srt_files = os.listdir(srt_root)
for srt_file in srt_files:
    if int(srt_file.split(".srt")[0][-2:]) not in select_idx:
        continue
    with open(os.path.join(srt_root, srt_file), 'r') as file:
        content = file.read()
        entries = content.strip().split('\n\n')
        result = []
        for entry in entries:
            lines = entry.split('\n')
            if len(lines) >= 3:
                time_range = lines[1].split(' --> ')
                start_time = time_to_float(time_range[0])
                end_time = time_to_float(time_range[1])
                pattern = r'<font color="#eba862">(.*?)</font>'
                match = re.search(pattern, ''.join(lines[2:]))
                if match:
                    subtitle = match.group(1)
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
    video_lst += [os.path.join(video_root, k.replace(".srt", ".mkv"))] * len(v)
    time_lst += v

res = []
for i in range(len(time_lst)):
    start_time, end_time, text, cnt = time_lst[i]
    video = video_lst[i]
    video_name = os.path.basename(video).replace(".mkv", "")
    output_video_path = os.path.join(output_dir, "video", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.mp4")
    output_audio_path = os.path.join(output_dir, "audio", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.wav")

    res.append({
        "video": output_video_path,
        "audio": output_audio_path,
        "text": text,
        "image_cnt": cnt
    })

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

print(len(res))
print(output_json)