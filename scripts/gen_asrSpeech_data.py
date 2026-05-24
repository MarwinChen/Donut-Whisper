import json

json_file_lst = [
    # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/commonvoice20_train.json",
#     "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/librispeech_asr_train-clean-100.json",
#     "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/librispeech_asr_train-clean-360.json",
#     "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/librispeech_asr_train-other-500.json",
    # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/m3av_asr_test.json",
    "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/librispeech_asr_test-clean.json",
]

output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/LS_test_clean.json"

data = []
for json_file in json_file_lst:
    with open(json_file, 'r') as fp:
        data += json.load(fp)

res = []
cnt = 0
for item in data:
    # if len(item["conversations"][1]["value"]) > 300:
    #     # print(len(item["conversations"][1]["value"]))
    #     cnt += 1
    #     continue

    res.append({
        # "video": item["video"],
        "audio": item["audio"],
        "text": item["conversations"][1]["value"],
    })

    # res.append({
    #     "tos_video": item["tos_key"],
    #     "tos_audio": item["tos_audio"],
    #     "text": item["conversations"][1]["value"],
    # })

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

print(len(res), cnt)
print(output_json)