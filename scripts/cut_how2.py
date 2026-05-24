import soundfile as sf
import json
import os
import concurrent.futures
import threading
from tqdm import tqdm

json_file = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val.json"
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val_cut.json"
output_root = "/mnt/bn/tiktok-mm-4/aiic/public/data/how2_split_wav"

os.makedirs(output_root, exist_ok=True)

with open(json_file, 'r') as fp:
    data = json.load(fp)

def process(item):
    audio_file = item["audio"]
    audio, sr = sf.read(audio_file)
    if len(audio.shape) >= 2:
        audio = audio[:, 0]

    start, end = item["timestamps"]
    audio = audio[int(start * sr): int(end * sr)]

    output_file = os.path.join(output_root, os.path.basename(audio_file).replace(".wav", f"-{round(start)}-{round(end)}.wav"))
    sf.write(output_file, audio, sr)
    return output_file

res = []
for item in data:
    audio_file = item["audio"]
    start, end = item["timestamps"]
    res.append({
        "audio": os.path.join(output_root, os.path.basename(audio_file).replace(".wav", f"-{round(start)}-{round(end)}.wav")),
        "text": item["text"],
    })

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

print(len(res))
print(output_json)

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    responses = list(tqdm(executor.map(process, data)))

