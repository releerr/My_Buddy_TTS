import json
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class AudioEncoder(nn.Module):
    def __init__(self, input_dim=config.N_MELS, hidden_dim=256, output_dim=config.OUTPUT_DIM):
        super().__init__()
        # 1D cnn
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(config.MAX_LENGTH)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim) # 强制特征归一化

    def forward(self, mel, target_len=None):
        # mel: (batch, time, mel_bins)
        x = mel.transpose(1, 2)  # (batch, mel_bins, time)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        # 动态决定输出长度：优先对齐文本 Token 数
        out_size = target_len if target_len is not None else config.MAX_LENGTH
        # x: (batch, hidden_dim, time)
        # Adaptive pooling: MPS backend currently doesn't support non-divisible input->output sizes.
        # Workaround: if running on MPS and input_time % MAX_LENGTH != 0, move tensor to CPU for pooling then back.
        # input_time = x.size(-1)
        # out_size = config.MAX_LENGTH
        orig_device = x.device
        if orig_device.type == 'mps' and (x.size(-1) % out_size != 0):
            x = x.to('cpu')
            x = F.adaptive_avg_pool1d(x, out_size)
            x = x.to(orig_device)
        else:
            x = F.adaptive_avg_pool1d(x, out_size)

        x = x.transpose(1, 2)      # (batch, out_size, hidden_dim)
        x = self.proj(x)           # (batch, out_size, output_dim)
        x = self.layer_norm(x)
        return x
    


if __name__ == "__main__":
    # 模拟输入：Batch=2, Time=800帧, Mel=128
    dummy_mel = torch.randn(2, 800, 128)
    # 模拟文本长度：假设当前 Batch 的文本是 15 个 Token
    target_len = 15
    
    model = AudioEncoder(input_dim=128, output_dim=256)
    output = model(dummy_mel, target_len=target_len)
    
    print(f"输入形状: {dummy_mel.shape}")
    print(f"输出形状: {output.shape}") # 预期应该是 [2, 15, 256]
    
    if output.shape == (2, target_len, 256):
        print(" AudioEncoder 形状对齐测试通过！")
    else:
        print(" 形状不对，请检查代码。")