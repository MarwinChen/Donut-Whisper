import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput


class DisWhisperOnlyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DisWhisperOnly(PreTrainedModel):
    def __init__(self, whisper_model, model_type):
        config = DisWhisperOnlyConfig()
        super().__init__(config)
        self.model_type = model_type

        self.audio_encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder
        

    def forward(self, input_ids=None, spectrograms=None, spectrograms_large=None, labels=None, audio_zero=False, **kwargs):
        # spectrograms = spectrograms.to(torch.float32)
        # spectrograms_large = spectrograms_large.to(torch.float32)
        if self.model_type == "distillation_whisper_only":
            encoder_output = self.audio_encoder(spectrograms).last_hidden_state
        else:
            encoder_output = self.audio_encoder(spectrograms_large).last_hidden_state

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output, output_hidden_states=True)
        else:
            decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output, output_hidden_states=True)
        
        output = decoder_output.last_hidden_state
        logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output, decoder_hidden_states=decoder_output.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "spectrograms": kwargs.get("spectrograms"),
            "spectrograms_large": kwargs.get("spectrograms_large"),
            "audio_zero": kwargs.get("audio_zero", False),
        }


