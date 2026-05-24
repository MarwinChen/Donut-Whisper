import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.distributed as dist
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.generic import ModelOutput
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timing_tracer import TimingContext

class DonutWhisperConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class MultiModalFusionLayer(nn.Module):
    """多模态融合层，支持多种融合策略"""
    
    def __init__(self, hidden_size, fusion_type="linear", num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_type = fusion_type
        self.dropout = dropout
        
        if fusion_type == "linear":
            # 简单线性融合
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )
            
        elif fusion_type == "attention":
            # 自注意力融合
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            
        elif fusion_type == "cross_attention":
            # 交叉注意力融合
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            
        elif fusion_type == "q_former":
            # Q-Former风格融合
            self.query = nn.Parameter(torch.randn(1, 256, hidden_size))  # 256个查询向量
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attention = nn.MultiheadAttention(
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
            
        elif fusion_type == "sliding_window_q_former":
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
            self.norm4 = nn.LayerNorm(hidden_size)
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
            
    def forward(self, audio_feat, image_feat, audio_mask=None, image_mask=None):
        if self.fusion_type == "linear":
            # 简单拼接后线性变换
            combined = torch.cat([audio_feat, image_feat], dim=1)
            return self.fusion_layer(combined)
            
        elif self.fusion_type == "attention":
            # 自注意力融合
            combined = torch.cat([audio_feat, image_feat], dim=1)
            attn_output, _ = self.self_attention(combined, combined, combined)
            combined = self.norm1(combined + attn_output)
            ffn_output = self.ffn(combined)
            return self.norm2(combined + ffn_output)
            
        elif self.fusion_type == "cross_attention":
            # 交叉注意力：音频作为查询，图像作为键值
            attn_output, _ = self.cross_attention(
                query=audio_feat,
                key=image_feat,
                value=image_feat,
                key_padding_mask=image_mask
            )
            audio_feat = self.norm1(audio_feat + attn_output)
            ffn_output = self.ffn(audio_feat)
            return self.norm2(audio_feat + ffn_output)
            
        elif self.fusion_type == "q_former":
            # Q-Former风格：使用查询向量与多模态特征交互
            batch_size = audio_feat.size(0)
            queries = self.query.expand(batch_size, -1, -1)
            
            # 自注意力
            queries = self.norm1(queries + self.self_attention(queries, queries, queries)[0])
            
            # 与音频特征的交叉注意力
            queries = self.norm2(queries + self.cross_attention(
                query=queries,
                key=audio_feat,
                value=audio_feat,
                key_padding_mask=audio_mask
            )[0])
            
            # 与图像特征的交叉注意力
            queries = self.norm3(queries + self.cross_attention(
                query=queries,
                key=image_feat,
                value=image_feat,
                key_padding_mask=image_mask
            )[0])
            
            # FFN
            queries = queries + self.ffn(queries)
            
            return self.output_proj(queries)
            
        elif self.fusion_type == "sliding_window_q_former":
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
            
            # cross attention
            image_attn_output, _ = self.image_cross_attention(
                query=queries,
                key=image_feat,
                value=image_feat,
                key_padding_mask=image_mask
            )
            queries = self.norm3(queries + image_attn_output)
            
            # FFN
            queries = self.norm4(queries + self.ffn(queries))
            
            return self.output_proj(queries)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

class DonutWhisper(PreTrainedModel):
    def __init__(self, whisper_model, image_model_path, fusion_type="linear", fusion_config=None):
        config = DonutWhisperConfig()
        super().__init__(config)
        self.model_type = "donut_whisper"

        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

        self.image_encoder = donut_model.encoder
        self.audio_encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder

        self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.config.d_model)
        
        # 使用新的融合层替代简单的线性层
        if fusion_config is None:
            fusion_config = {}
        self.fusion_layer = MultiModalFusionLayer(
            hidden_size=whisper_model.config.d_model,
            fusion_type=fusion_type,
            **fusion_config
        )
        
        # 时序记录器
        self.timing_tracer = None
    
    def set_timing_tracer(self, tracer):
        """设置时序记录器"""
        self.timing_tracer = tracer

    def forward(self, input_ids=None, spectrograms=None, images=None, labels=None, images_len=None, **kwargs):
        # 记录图像编码时间
        if self.timing_tracer:
            with TimingContext(self.timing_tracer, "image_encoder"):
                image_feat = self.image_encoder(pixel_values=images).last_hidden_state
            with TimingContext(self.timing_tracer, "image_linear"):
                image_feat = F.gelu(self.image_linear(image_feat))
        else:
            image_feat = self.image_encoder(pixel_values=images).last_hidden_state
            image_feat = F.gelu(self.image_linear(image_feat))

        video_feats = torch.split(image_feat, images_len, dim=0)
        
        nt_list = [ni * video_feat.shape[1] for ni, video_feat in zip(images_len, video_feats)]
        max_nt = max(nt_list)

        flatten_feats = [vf.view(-1, image_feat.shape[2]) for vf in video_feats]
        padded_feats = nn.utils.rnn.pad_sequence(flatten_feats, batch_first=True, padding_value=0)
        # padded_feats = padded_feats * 0
        
        # 记录音频编码时间
        if self.timing_tracer:
            with TimingContext(self.timing_tracer, "audio_encoder"):
                audio_feat = self.audio_encoder(spectrograms).last_hidden_state
        else:
            audio_feat = self.audio_encoder(spectrograms).last_hidden_state
        # if audio_zero:
        #     audio_feat = torch.zeros_like(audio_feat)

        # 记录融合层时间
        if self.timing_tracer:
            with TimingContext(self.timing_tracer, "fusion_layer"):
                encoder_output = self.fusion_layer(audio_feat, padded_feats)
        else:
            encoder_output = self.fusion_layer(audio_feat, padded_feats)

        # 记录解码器时间
        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            if self.timing_tracer:
                with TimingContext(self.timing_tracer, "decoder"):
                    decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output)
            else:
                decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output)
        else:
            if self.timing_tracer:
                with TimingContext(self.timing_tracer, "decoder"):
                    decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)
            else:
                decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)
        
        output = decoder_output.last_hidden_state
        
        # 记录logits计算时间
        if self.timing_tracer:
            with TimingContext(self.timing_tracer, "logits_computation"):
                logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())
        else:
            logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "spectrograms": kwargs.get("spectrograms"),
            "images": kwargs.get("images"),
            "images_len": kwargs.get("images_len"),
        }
