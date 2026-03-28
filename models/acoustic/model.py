import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, dim)

    def forward(self, x):
        # x: (batch, time, dim)
        return x + self.pe[:, :x.size(1), :]

class LiteConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim), # 深度可分离卷积
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention 分支 (全局上下文)
        norm_x = self.ln(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # Conv 分支 (局部连贯性)
        res = x
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x + res

class ResidualConv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.ln = nn.GroupNorm(8, dim) # 使用 GroupNorm 对 M1 更友好，且不受 Batch 大小影响
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.ln2 = nn.GroupNorm(8, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.activation(self.ln(self.conv1(x)))
        out = self.dropout(out)
        out = self.ln2(self.conv2(out))
        out = self.dropout(out)
        return out + residual

class DecoderResNet(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=256, out_dim=80, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualConv1DBlock(hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x):
        # x: (batch, time, in_dim) -> (batch, in_dim, time)
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x) 
        return x.transpose(1, 2).contiguous()


class StablePostNet(nn.Module):
    def __init__(self, mel_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, mel_dim, kernel_size=5, padding=2),
        )

    def forward(self, mel):
        residual = self.net(mel.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return mel + residual

class MyBuddyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = getattr(config, 'MODEL_HIDDEN_DIM', 192)
        max_frames = getattr(config, 'MAX_FRAME_LEN', 10000)

        # 语义投影与情感融入
        self.input_proj = nn.Linear(config.OUTPUT_DIM, hidden_dim)
        self.prosody_proj = nn.Linear(getattr(config, 'ACOUSTIC_PROSODY_DIM', 20), hidden_dim)
        self.emotion_embed = nn.Embedding(config.NUM_EMOTIONS, hidden_dim)
        
        # Conformer 编码层（文本层面建模）
        self.encoder_blocks = nn.ModuleList([
            LiteConformerBlock(hidden_dim) for _ in range(2)
        ])
        
        # 位置编码与解码器
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, max_len=max_frames)
        self.decoder = DecoderResNet(
            in_dim=hidden_dim, 
            hidden_dim=256, 
            out_dim=config.MEL_BINS,
            num_layers=4
        )
        self._init_weights()
    
    # 对所有线性层和卷积层进行 Xavier 初始化
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def forward(self, bert_feat, prosody_feat, emotion_ids, durations):
        """
        bert_feat: (batch, T_token, 256)
        emotion_ids: (batch,)
        durations: (batch, T_token) 每个音素的帧数
        """
  
        x = self.input_proj(bert_feat)
        x = x + self.prosody_proj(prosody_feat)
        if emotion_ids.ndim > 1:
            emotion_ids = emotion_ids.view(-1)
        e = self.emotion_embed(emotion_ids)
        e = e.view(x.size(0), 1, -1)
        e = e.expand(-1, x.size(1), -1)
        x = x + e

        x = self.pos_embed(x)
        
        for block in self.encoder_blocks:
            x = block(x)
            
        # duration expand,把每个字按照 “时长” 拉长！如：“今” 读 3 帧 → 复制 3 次
        # 直接使用高效的 interleave 逻辑并处理 padding
        expanded_list = []
        for i in range(x.size(0)):
            rep = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            expanded_list.append(rep)
        
        # 自动对齐 Batch 长度
        x_expanded = pad_sequence(expanded_list, batch_first=True)
        
        # 解码生成频谱
        mel_output = self.decoder(x_expanded)
        
        return mel_output


class StableAcousticModel(nn.Module):
    """
    A simpler and more stable acoustic model for small-data TTS:
    token conditioning -> duration expansion -> BiLSTM frame modeling -> mel + postnet
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = getattr(config, 'ACOUSTIC_HIDDEN_DIM', 256)
        prosody_dim = getattr(config, 'ACOUSTIC_PROSODY_DIM', 20)
        postnet_dim = getattr(config, 'ACOUSTIC_POSTNET_DIM', 256)
        dropout = getattr(config, 'ACOUSTIC_DROPOUT', 0.0)

        self.input_proj = nn.Linear(config.OUTPUT_DIM, hidden_dim)
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.emotion_embed = nn.Embedding(config.NUM_EMOTIONS, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.token_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.frame_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.frame_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.frame_norm = nn.LayerNorm(hidden_dim)
        self.skip_mel_proj = nn.Linear(hidden_dim, config.MEL_BINS)
        self.mel_proj = nn.Linear(hidden_dim, config.MEL_BINS)
        self.postnet = StablePostNet(config.MEL_BINS, hidden_dim=postnet_dim, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, bert_feat, prosody_feat, emotion_ids, durations):
        x = self.input_proj(bert_feat) + self.prosody_proj(prosody_feat)
        if emotion_ids.ndim > 1:
            emotion_ids = emotion_ids.view(-1)
        emo = self.emotion_embed(emotion_ids).unsqueeze(1)
        x = x + emo
        x = self.input_norm(x)
        x = self.token_encoder(x)

        expanded_list = []
        for i in range(x.size(0)):
            rep = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            expanded_list.append(rep)

        x_expanded = pad_sequence(expanded_list, batch_first=True)
        frame_hidden, _ = self.frame_encoder(x_expanded)
        frame_hidden = self.frame_proj(frame_hidden)
        frame_hidden = self.frame_norm(frame_hidden)
        mel = self.mel_proj(frame_hidden) + self.skip_mel_proj(x_expanded)
        mel = self.postnet(mel)
        return mel


class AttentionDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory, memory_key_padding_mask=None, attn_bias=None):
        h = self.self_ln(x)
        self_attn_out, _ = self.self_attn(h, h, h)
        x = x + self_attn_out

        h = self.cross_ln(x)
        attn_mask = None
        if attn_bias is not None:
            attn_mask = attn_bias.repeat_interleave(self.num_heads, dim=0)
        cross_out, cross_weights = self.cross_attn(
            h,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
        )
        x = x + cross_out

        x = x + self.ffn(self.ffn_ln(x))
        return x, cross_weights


class AttentionAcousticModel(nn.Module):
    """
    Attention-based acoustic model:
    token conditioning -> token encoder -> frame queries -> cross attention -> mel + postnet
    Durations are only used to estimate output frame length, not to hard-expand tokens.
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = getattr(config, 'ACOUSTIC_HIDDEN_DIM', 256)
        prosody_dim = getattr(config, 'ACOUSTIC_PROSODY_DIM', 20)
        postnet_dim = getattr(config, 'ACOUSTIC_POSTNET_DIM', 256)
        dropout = getattr(config, 'ACOUSTIC_DROPOUT', 0.0)
        num_heads = getattr(config, 'ACOUSTIC_ATTN_HEADS', 4)
        num_layers = getattr(config, 'ACOUSTIC_DECODER_LAYERS', 3)
        max_frames = getattr(config, 'MAX_FRAME_LEN', 10000)
        self.duration_prior_strength = getattr(config, 'ACOUSTIC_DURATION_PRIOR_STRENGTH', 0.0)
        self.duration_prior_sigma = getattr(config, 'ACOUSTIC_DURATION_PRIOR_SIGMA', 0.2)

        self.input_proj = nn.Linear(config.OUTPUT_DIM, hidden_dim)
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.emotion_embed = nn.Embedding(config.NUM_EMOTIONS, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_pos = SinusoidalPositionalEmbedding(hidden_dim, max_len=config.MAX_LENGTH)
        self.token_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.frame_queries = SinusoidalPositionalEmbedding(hidden_dim, max_len=max_frames)
        self.decoder_blocks = nn.ModuleList([
            AttentionDecoderBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pre_mel = nn.Linear(hidden_dim, config.MEL_BINS)
        self.postnet = StablePostNet(config.MEL_BINS, hidden_dim=postnet_dim, dropout=dropout)
        self.latest_attn = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def build_duration_prior(self, durations, max_frames, device):
        if self.duration_prior_strength <= 0.0:
            return None

        batch_size, token_len = durations.shape
        total_frames = durations.sum(dim=1).clamp_min(1).float()
        frame_pos = (torch.arange(max_frames, device=device).float() + 0.5).unsqueeze(0)
        frame_pos = frame_pos / total_frames.unsqueeze(1)

        cumulative = durations.cumsum(dim=1).float()
        token_centers = cumulative - durations.float() * 0.5
        token_centers = token_centers / total_frames.unsqueeze(1)

        diff = frame_pos.unsqueeze(-1) - token_centers.unsqueeze(1)
        bias = -self.duration_prior_strength * (diff ** 2) / (2 * (self.duration_prior_sigma ** 2))
        return bias

    def forward(self, bert_feat, prosody_feat, emotion_ids, durations):
        x = self.input_proj(bert_feat) + self.prosody_proj(prosody_feat)
        if emotion_ids.ndim > 1:
            emotion_ids = emotion_ids.view(-1)
        x = x + self.emotion_embed(emotion_ids).unsqueeze(1)
        x = self.token_pos(x)
        x = self.token_norm(x)
        memory, _ = self.token_encoder(x)
        memory = self.memory_proj(memory)
        token_padding_mask = durations <= 0

        frame_lengths = durations.long().sum(dim=1)
        max_frames = int(frame_lengths.max().item())
        queries = torch.zeros(x.size(0), max_frames, memory.size(-1), device=x.device)
        queries = self.frame_queries(queries)
        attn_bias = self.build_duration_prior(durations, max_frames, x.device)

        attn_maps = []
        for block in self.decoder_blocks:
            queries, cross_weights = block(
                queries,
                memory,
                memory_key_padding_mask=token_padding_mask,
                attn_bias=attn_bias,
            )
            attn_maps.append(cross_weights)

        mel = self.pre_mel(queries)
        mel = self.postnet(mel)
        if attn_maps:
            self.latest_attn = torch.stack(attn_maps, dim=0).mean(dim=0)
        else:
            self.latest_attn = None
        return mel


def build_acoustic_model(cfg):
    model_type = getattr(cfg, 'ACOUSTIC_MODEL_TYPE', 'stable_upsample_lstm')
    if model_type == 'attention_frame_decoder':
        return AttentionAcousticModel(cfg)
    if model_type == 'stable_upsample_lstm':
        return StableAcousticModel(cfg)
    if model_type == 'lite_conformer':
        return MyBuddyModel(cfg)
    raise ValueError(f"未知声学模型类型: {model_type}")



# test
if __name__ == "__main__":
    # 测试模型前向传播
    batch_size = 2
    token_len = 10
    mel_bins = 128
    hidden_dim = 192

    # 创建 dummy config 对象（或直接使用 config）
    class DummyConfig:
        OUTPUT_DIM = 256
        NUM_EMOTIONS = 6
        MEL_BINS = mel_bins
        MODEL_HIDDEN_DIM = hidden_dim
        MAX_FRAME_LEN = 5000
        ACOUSTIC_PROSODY_DIM = 20
        ACOUSTIC_MODEL_TYPE = 'stable_upsample_lstm'
        ACOUSTIC_HIDDEN_DIM = 256
        ACOUSTIC_POSTNET_DIM = 256

    dummy_config = DummyConfig()

    model = build_acoustic_model(dummy_config)
    bert_feat = torch.randn(batch_size, token_len, 256)
    prosody_feat = torch.randn(batch_size, token_len, dummy_config.ACOUSTIC_PROSODY_DIM)
    emotion_ids = torch.randint(0, 6, (batch_size,))
    durations = torch.randint(1, 5, (batch_size, token_len))  # 每个音素帧数

    output = model(bert_feat, prosody_feat, emotion_ids, durations)
    print("Model output shape:", output.shape)  # 应为 (batch, T_mel, mel_bins)
    print("测试通过！")
