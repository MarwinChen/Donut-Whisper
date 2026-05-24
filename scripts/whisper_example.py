import torch
from model.modeling_whisper import WhisperModel
from transformers import AutoFeatureExtractor # , WhisperModel
from datasets import load_dataset

model = WhisperModel.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base").cuda()
feature_extractor = AutoFeatureExtractor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features.cuda().squeeze()
input_features = torch.stack([input_features] * 2)

decoder_input_ids = torch.tensor([[50258, 13], [50258, 13]])
decoder_input_ids = decoder_input_ids.cuda()

last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
print(last_hidden_state)