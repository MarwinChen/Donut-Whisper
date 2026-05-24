import json

json_file = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/shot2story_20k_val.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/shot2story_134k_train.json" # 
output_json = json_file

with open(json_file, 'r') as fp:
    data = json.load(fp)

res = []
cnt = 0
for item in data:
    # if len(item["text"].split()) > 150:
    if len(item["text"]) > 800:
        print(item["text"])
        breakpoint()
        cnt += 1
        continue

    res.append(item)

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

print(len(res), cnt)
print(output_json)