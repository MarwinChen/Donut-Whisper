import re
import numpy as np
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
import json
import cv2
from tqdm import tqdm
import os

processor = DonutProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2")

device = "cuda"
model.to(device)

json_file = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val_cut.json"
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/donut-base-how2val/results_final.json"
os.makedirs(os.path.dirname(output_json), exist_ok=True)

with open(json_file, 'r') as fp:
    data = json.load(fp)

res = []
for item in tqdm(data):
    text = item["text"]
    bottom_part = np.ones((60, 1920, 3), dtype=np.float32) * -1.0
    cv2.putText(bottom_part, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    image = np.transpose(bottom_part, (2, 0, 1))
    image = (image + 1) / 2 * 255
    image = image.astype(np.uint8)

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").replace("<s_cord-v2><s_menu><s_nm> ", "")
    
    res.append({
        "id": item["audio"],
        "ref": item["text"],
        "pred": sequence,
    })

with open(output_json, 'w') as fp:
    json.dump(res, fp, indent=4)

print(len(res))
print(output_json)