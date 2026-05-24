import json
from datasets import load_dataset
from tqdm import tqdm

cache_directory = "/mnt/bn/tiktok-mm-4/aiic/public/data/finevideo"
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/finevideo.json"

ds = load_dataset("HuggingFaceFV/finevideo", cache_dir=cache_directory)

def time_string_to_float(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds) + float(milliseconds) / 1000
    return total_seconds

res = []
for i in tqdm(range(len(ds['train']))):
    for item in ds["train"][i]["json"]["timecoded_text_to_speech"]:
        start, end = time_string_to_float(item["start"]), time_string_to_float(item["end"])
        res.append({
            "tos_video": f"video/finevideo/{i}.mp4",
            "timestamps": [start, end],
            "text": item["text"],
        })

with open(output_json, 'w') as fp:
    json.dump(res, fp)

print(len(res))
print(output_json)