import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import VisionEncoderDecoderModel
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
import torch.distributed as dist

class DonutOnlyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DonutOnly(PreTrainedModel):
    def __init__(self, image_model_path):
        config = DonutOnlyConfig()
        super().__init__(config)
        self.model_type = "donut_only"

        # self.donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)
        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

        self.encoder = donut_model.encoder
        self.decoder = donut_model.decoder

    def forward(self, input_ids=None, images=None, labels=None, images_len=None, audio_zero=False, **kwargs):
        image_feat = self.encoder(pixel_values=images).last_hidden_state

        video_feats = torch.split(image_feat, images_len, dim=0)
        # print(f"RANK {dist.get_rank()}: {images_len}")
        
        nt_list = [ni * video_feat.shape[1] for ni, video_feat in zip(images_len, video_feats)]
        max_nt = max(nt_list)

        flatten_feats = [vf.view(-1, image_feat.shape[2]) for vf in video_feats]
        padded_feats = nn.utils.rnn.pad_sequence(flatten_feats, batch_first=True, padding_value=0)

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 2
        else:
            new_labels = input_ids

        outputs = self.decoder(input_ids=new_labels, encoder_hidden_states=padded_feats)
        # outputs = self.donut_model(pixel_values=images, decoder_input_ids=new_labels)

        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "images": kwargs.get("images"),
            "images_len": kwargs.get("images_len"),
        }