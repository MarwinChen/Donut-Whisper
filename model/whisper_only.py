import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

class MultiModalFusionLayer(nn.Module):
    def __init__(self, hidden_size, fusion_type="sliding_window_q_former", num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.fusion_type = fusion_type
        self.dropout = dropout
        self.hidden_size = hidden_size
        if fusion_type == "sliding_window_q_former":
            # 滑窗Q-Former风格融合
            self.query = nn.Parameter(torch.randn(1, 256, hidden_size))  # 256个查询向量
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.audio_cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.image_cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.norm3 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            self.output_proj = nn.Linear(hidden_size, hidden_size)
            # 滑窗参数
            self.window_size = kwargs.get('window_size', 128)  # 默认窗口大小
            self.stride = kwargs.get('stride', 64)  # 默认步长
            self.overlap = kwargs.get('overlap', True)  # 是否重叠
    def forward(self, audio_feat, audio_mask=None):
        if self.fusion_type == "sliding_window_q_former":
            # 滑窗Q-Former风格：使用查询向量与多模态特征交互
            batch_size = audio_feat.size(0)
            queries = self.query.expand(batch_size, -1, -1)
            
            # 自注意力
            queries = self.norm1(queries + self.self_attention(queries, queries, queries)[0])
            
            # 对音频特征进行滑窗处理
            audio_windows = []
            audio_masks = []
            
            # 计算滑窗位置
            for start_idx in range(0, audio_feat.size(1) - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                audio_windows.append(audio_feat[:, start_idx:end_idx])
                if audio_mask is not None:
                    audio_masks.append(audio_mask[:, start_idx:end_idx])
            
            # 如果没有足够的音频长度，使用整个音频
            if len(audio_windows) == 0:
                audio_windows = [audio_feat]
                if audio_mask is not None:
                    audio_masks = [audio_mask]
            
            # 处理每个音频窗口
            audio_enhanced_queries = []
            for i, audio_win in enumerate(audio_windows):
                audio_mask_win = audio_masks[i] if audio_mask is not None else None
                
                # 与音频窗口的交叉注意力
                audio_attn_output, _ = self.audio_cross_attention(
                    query=queries,
                    key=audio_win,
                    value=audio_win,
                    key_padding_mask=audio_mask_win
                )
                audio_enhanced_queries.append(audio_attn_output)
            
            # 合并所有音频窗口的增强查询向量
            if len(audio_enhanced_queries) > 1:
                # 使用平均池化合并多个窗口的结果
                audio_enhanced_queries = torch.stack(audio_enhanced_queries, dim=0).mean(dim=0)
            else:
                audio_enhanced_queries = audio_enhanced_queries[0]
            
            # 应用音频增强
            queries = self.norm2(queries + audio_enhanced_queries)
            
            # FFN
            queries = self.norm3(queries + self.ffn(queries))
            
            return self.output_proj(queries)

class WhisperOnlyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class WhisperOnly(PreTrainedModel):
    def __init__(self, whisper_model, fusion_type="sliding_window_q_former", fusion_config=None):
        config = WhisperOnlyConfig()
        super().__init__(config)
        self.model_type = "whisper_only"

        self.encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder
        if fusion_config is None:
            fusion_config = {}
        self.fusion_layer = MultiModalFusionLayer(
            hidden_size=whisper_model.config.d_model,
            fusion_type=fusion_type,
            **fusion_config
        )

    def forward(self, input_ids=None, spectrograms=None, labels=None, **kwargs):
        encoder_output = self.encoder(spectrograms).last_hidden_state
        encoder_output = self.fusion_layer(encoder_output)

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
            "spectrograms": kwargs.get("spectrograms"),
        }


