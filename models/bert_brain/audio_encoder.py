import json
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn as nn
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
        self.adaptive_pool = nn.AdaptiveAvgPool1d(config.MAX_LENGTH)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, mel):
        # mel: (batch, time, mel_bins)
        x = mel.transpose(1, 2)  # (batch, mel_bins, time)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        # x: (batch, hidden_dim, time)
        # Adaptive pooling: MPS backend currently doesn't support non-divisible input->output sizes.
        # Workaround: if running on MPS and input_time % MAX_LENGTH != 0, move tensor to CPU for pooling then back.
        input_time = x.size(-1)
        out_size = config.MAX_LENGTH
        orig_device = x.device
        if orig_device.type == 'mps' and (input_time % out_size != 0):
            x = x.to('cpu')
            x = self.adaptive_pool(x)
            x = x.to(orig_device)
        else:
            x = self.adaptive_pool(x)  # (batch, hidden_dim, MAX_LENGTH)
        x = x.transpose(1, 2)      # (batch, MAX_LENGTH, hidden_dim)
        x = self.proj(x)           # (batch, MAX_LENGTH, output_dim)
        return x