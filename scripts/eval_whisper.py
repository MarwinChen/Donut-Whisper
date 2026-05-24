import os
import json

import argparse
from tqdm import tqdm
import soundfile as sf
from transformers import AutoProcessor, WhisperForConditionalGeneration


def whisper_decode(data_path, out_path, start, end, b, lang="en", task="transcribe"):
    w_p = '/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base'

    processor = AutoProcessor.from_pretrained(w_p)
    model = WhisperForConditionalGeneration.from_pretrained(w_p).cuda().eval()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=task)
    
    with open(data_path, 'r') as f:
        item_list = json.loads(f.read())

    if end == -1:
        end = len(item_list)

    rets = []

    for item in tqdm(item_list[start: end]):
        path = item["audio"]
        data, sr = sf.read(path)
        if len(data.shape) == 2:
            data = data[:, 0]
        ground_truth = item['text']

        inputs = processor(data, return_tensors="pt", sampling_rate=sr)
        input_features = inputs.input_features.cuda()

        generated_ids = model.generate(inputs=input_features, return_dict_in_generate=True, num_beams=b, num_return_sequences=b, output_scores=True)
        text = processor.batch_decode(generated_ids["sequences"], skip_special_tokens=True)[0]

        ret = {
            "id": path,
            "ref": ground_truth,
            "pred": text,
        }

        rets.append(ret)


    with open(out_path, 'w') as fp:
        json.dump(rets, fp, indent=4)
    print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("-b", type=int, default=1)
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)

    whisper_decode(
        args.data_path, args.out_path, args.start, args.end, args.b, lang=args.lang,
    )