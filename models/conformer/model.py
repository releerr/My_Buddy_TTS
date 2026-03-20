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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
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
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x) 
        return x.transpose(1, 2) # (batch, time, out_dim)

class MyBuddyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = getattr(config, 'MODEL_HIDDEN_DIM', 192)
        max_frames = getattr(config, 'MAX_FRAME_LEN', 10000)

        # 语义投影与情感融入
        self.input_proj = nn.Linear(config.OUTPUT_DIM, hidden_dim)
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


    def forward(self, bert_feat, emotion_ids, durations):
        """
        bert_feat: (batch, T_token, 256)
        emotion_ids: (batch,)
        durations: (batch, T_token) 每个音素的帧数
        """
  
        x = self.input_proj(bert_feat)
        e = self.emotion_embed(emotion_ids).unsqueeze(1)
        x = x + e # 融合情感信息
        
        for block in self.encoder_blocks:
            x = block(x)
            
        # duration expand,把每个字按照 “时长” 拉长！如：“今” 读 3 帧 → 复制 3 次
        # 直接使用高效的 interleave 逻辑并处理 padding
        expanded_list = []
        for i in range(x.size(0)):
            # 根据 durations[i] 复制 x[i]
            rep = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            expanded_list.append(rep)
        
        # 自动对齐 Batch 长度
        x_expanded = pad_sequence(expanded_list, batch_first=True)
        
        # 展开后的位置编码 (让声码器知道音频的先后)
        x_expanded = self.pos_embed(x_expanded)

        # 解码生成频谱
        mel_output = self.decoder(x_expanded)
        
        return mel_output
    



# test
if __name__ == "__main__":
    # 测试模型前向传播
    batch_size = 2
    token_len = 10
    mel_bins = 80
    hidden_dim = 192

    # 创建 dummy config 对象（或直接使用 config）
    class DummyConfig:
        OUTPUT_DIM = 256
        NUM_EMOTIONS = 6
        MEL_BINS = mel_bins
        MODEL_HIDDEN_DIM = hidden_dim
        MAX_FRAME_LEN = 5000

    dummy_config = DummyConfig()

    model = MyBuddyModel(dummy_config)
    bert_feat = torch.randn(batch_size, token_len, 256)
    emotion_ids = torch.randint(0, 6, (batch_size,))
    durations = torch.randint(1, 5, (batch_size, token_len))  # 每个音素帧数

    output = model(bert_feat, emotion_ids, durations)
    print("Model output shape:", output.shape)  # 应为 (batch, T_mel, mel_bins)
    print("测试通过！")