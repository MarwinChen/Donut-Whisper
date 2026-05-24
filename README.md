# Donut Whisper

Donut Whisper is a multimodal ASR research codebase that combines Whisper audio features with Donut visual features. The main training entry is `train.py`, and `run.sh` wraps the common single-node/multi-node `torchrun` arguments.

## Environment

```bash
pip install -r requirements.txt
```

`ffmpeg` is also required for the video/audio preprocessing scripts. If you use DeepSpeed, install it separately in a Linux/CUDA environment and keep `--deepspeed scripts/zero0.json`; otherwise pass `--deepspeed None`.

Default model IDs:

- Whisper: `openai/whisper-base`
- Whisper large feature extractor: `openai/whisper-large-v3`
- Donut: `naver-clova-ix/donut-base-finetuned-cord-v2`

Local Hugging Face cache/model paths can still be passed through `--whisper_path`, `--whisper_large_path`, and `--image_model_path`.

## Data Format

The dataset is a JSON list. Each item should contain local paths:

```json
[
  {
    "audio": "/path/to/audio.wav",
    "video": "/path/to/video.mp4",
    "text": "transcription text",
    "timestamps": [0.0, 3.2],
    "image_cnt": 3
  }
]
```

Fields:

- `audio`: required local audio path.
- `video`: optional local video path. If omitted, the loader renders text-only frames.
- `text`: reference transcription.
- `timestamps`: optional `[start, end]` seconds used to sample frames from the video.
- `image_cnt`: optional number of sampled frames. It is cast to `int`.

Remote object-storage fields were removed from the public loader. Convert those samples to local `audio`/`video` paths before publishing or training.

## Training

```bash
bash run.sh \
  --train_data data/train.json \
  --eval_data data/val.json \
  --output_name donut_whisper_debug \
  --model_type donut_whisper \
  --epochs 10 \
  --train_bs 16 \
  --eval_bs 4 \
  --nproc_per_node 1 \
  --deepspeed None
```

Useful options:

- `--model_type`: `donut_whisper`, `donut_whisper_attn`, `whisper_only`, `donut_only`, `distillation_whisper_only`, or `donut_whisper_ed`.
- `--tokenizer_language`: defaults to `zh`; set `en` for English experiments.
- `--use_lora`: defaults to `True`.
- `--fusion_window_size`, `--fusion_stride`, `--fusion_num_heads`, `--fusion_dropout`: sliding-window fusion parameters.
- `--cuda_visible_devices`: optional GPU list, for example `0,1`.

## Testing

```bash
bash run.sh \
  --do_test \
  --test_data data/test.json \
  --ckpt output/donut_whisper_debug/checkpoint-1000/pytorch_model.bin \
  --output_name donut_whisper_test \
  --model_type donut_whisper \
  --eval_bs 4 \
  --nproc_per_node 1 \
  --deepspeed None
```

Merged predictions are written to `output/test/<output_name>/results_final.json`.

## Evaluation

```bash
python eval/eval_asr.py \
  --json_file output/test/donut_whisper_test/results_final.json \
  --language zh \
  --metric cer
```

For SCTK-style files:

```bash
python scripts/json2txt.py --json_path output/test/donut_whisper_test/results_final.json --char_level
python scripts/sctk.py --pred_txt output/test/donut_whisper_test/preds.txt --ref_txt output/test/donut_whisper_test/refs.txt --out_dir output/test/donut_whisper_test/sctk
```

## Preprocessing Helpers

Merge JSON lists:

```bash
python scripts/Merge_json.py --inputs data/train_a.json data/train_b.json --output data/train_merged.json
```

Check whether local media files can be opened:

```bash
python dataset/Datei_test.py --data_path data/train.json --bad_files bad_files.txt
```

Build a dataset JSON from SRT/video files:

```bash
python scripts/video_cutting.py \
  --srt_root data/srt \
  --video_root data/videos \
  --output_dir data/cuts \
  --output_json data/cuts.json \
  --run_ffmpeg
```
