import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
# from model.modeling_whisper import WhisperModel
import copy
from transformers import VisionEncoderDecoderModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.attn(query.transpose(0, 1), key.transpose(0, 1),
                                   value.transpose(0, 1), key_padding_mask=key_padding_mask)
        return attn_output.transpose(0, 1)

class QFormer(nn.Module):
    def __init__(self, num_queries, hidden_size, num_heads, num_layers):
        super().__init__()
        self.num_queries = num_queries
        self.query_embeddings = nn.Embedding(num_queries, hidden_size)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, input_features, input_mask=None):
        batch_size = input_features.size(0)
        query_embeds = self.query_embeddings(torch.arange(self.num_queries, device=input_features.device))
        query_embeds = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        for cross_attn, norm in zip(self.cross_attn_layers, self.norm_layers):
            attn_output = cross_attn(query_embeds, input_features, input_features, key_padding_mask=input_mask)
            query_embeds = norm(query_embeds + attn_output)

        return query_embeds

class DonutWhisperAttnConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DonutWhisperAttn(PreTrainedModel):
    def __init__(self, whisper_model, image_model_path):
        config = DonutWhisperAttnConfig()
        super().__init__(config)

        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

        self.image_encoder = donut_model.encoder
        self.audio_encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder

        self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.config.d_model)
        self.mix_linear = nn.Linear(whisper_model.config.d_model, whisper_model.config.d_model)

        self.qformer = QFormer(64, donut_model.config.encoder.hidden_size, 8, 2)

    def forward(self, input_ids=None, spectrograms=None, images=None, labels=None, audio_zero=False, **kwargs):

        if isinstance(images, list):
            raise NotImplementedError
        else:
            image_feat = self.image_encoder(pixel_values=images).last_hidden_state
            image_feat = self.qformer(image_feat)
            image_feat = F.gelu(self.image_linear(image_feat))

        audio_feat = self.audio_encoder(spectrograms).last_hidden_state

        encoder_output = torch.cat((audio_feat, image_feat), dim=1)
        encoder_output = F.gelu(self.mix_linear(encoder_output))

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output)
        else:
            decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)
        
        output = decoder_output.last_hidden_state
        logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "audios": kwargs.get("audios"),
            "spectrograms": kwargs.get("spectrograms"),
            "images": kwargs.get("images"),
            "audio_zero": kwargs.get("audio_zero", False),
        }
