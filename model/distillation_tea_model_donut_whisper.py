import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.distributed as dist
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from transformers import VisionEncoderDecoderModel

class DonutWhisperConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DisDonutWhisper(PreTrainedModel):
    def __init__(self, whisper_model, image_model_path):
        config = DonutWhisperConfig()
        super().__init__(config)
        self.model_type = "donut_whisper"

        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

        self.image_encoder = donut_model.encoder
        self.audio_encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder

        self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.config.d_model)
        self.mix_linear = nn.Linear(whisper_model.config.d_model, whisper_model.config.d_model)

    def forward(self, input_ids=None, spectrograms=None, images=None, labels=None, images_len=None, audio_zero=False, **kwargs):
        image_feat = self.image_encoder(pixel_values=images).last_hidden_state
        image_feat = F.gelu(self.image_linear(image_feat))

        video_feats = torch.split(image_feat, images_len, dim=0)
        # print(f"RANK {dist.get_rank()}: {images_len}")
        
        nt_list = [ni * video_feat.shape[1] for ni, video_feat in zip(images_len, video_feats)]
        max_nt = max(nt_list)

        flatten_feats = [vf.view(-1, image_feat.shape[2]) for vf in video_feats]
        padded_feats = nn.utils.rnn.pad_sequence(flatten_feats, batch_first=True, padding_value=0)
        
        audio_feat = self.audio_encoder(spectrograms).last_hidden_state

        encoder_output = torch.cat((audio_feat, padded_feats), dim=1)
        encoder_output = F.gelu(self.mix_linear(encoder_output))

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output, output_hidden_states=True)
        else:
            decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output,  output_hidden_states=True)
        
        output = decoder_output.last_hidden_state
        logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output, decoder_hidden_states=decoder_output.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "spectrograms": kwargs.get("spectrograms"),
            "images": kwargs.get("images"),
            "images_len": kwargs.get("images_len"),
            "audio_zero": kwargs.get("audio_zero", False),
        }
