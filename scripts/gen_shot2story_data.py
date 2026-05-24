import json
import os
import soundfile as sf
from tqdm import tqdm
import concurrent.futures

json_file = "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/shot2story/Shot2Story-20K/20k_train.json" # "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/shot2story/134k_full_train.json" # 
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/shot2story_20k_train.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/shot2story_134k_train.json" # 

audio_root = "/mnt/bn/tiktok-mm-2/aiic/public/data/audio/training/shot2story" # "/mnt/bn/tiktok-mm-2/aiic/public/data/audio/training/shot2story-134k" # 
video_root = "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/shot2story/data/collation_final_videos_20k" # "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/shot2story/data/release_134k_videos" # 

afile = os.listdir(audio_root)
adic = dict(zip(afile, [1] * len(afile)))

vfile = os.listdir(video_root)
vdic = dict(zip(vfile, [1] * len(vfile)))

with open(json_file, 'r') as fp:
    data = json.load(fp)

res = []
for item in tqdm(data):
    if item["video"] in vdic and item["video"].replace(".mp4", ".wav") in adic:
        res.append({
            "audio": os.path.join(audio_root, item["video"].replace(".mp4", ".wav")),
            "video": os.path.join(video_root, item["video"]),
            "text": item["whole_ASR"],
        })

def check_duration(item):
    audio, sr = sf.read(item["audio"])
    duration = len(audio) / sr
    if duration > 30:
        return None
    item["duration"] = duration
    return item

with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
    new_res = list(tqdm(executor.map(check_duration, res)))
new_res = [it for it in new_res if it is not None]

with open(output_json, 'w') as fp:
    json.dump(new_res, fp, indent=4)

print(len(new_res))
print(output_json)
