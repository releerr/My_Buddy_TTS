import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# 路径对齐：确保能导入 config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import config
from models.acoustic.model import build_acoustic_model
from models.acoustic.data_loader import create_dataloader, denormalize_mel


def load_model(checkpoint_path, device):
    """加载训练好的 Conformer 模型"""
    print(f"--- 正在加载模型: {os.path.basename(checkpoint_path)} ---")
    model = build_acoustic_model(config)
    # 允许不完全匹配，以便加载 latest_model.pth (包含 optimizer 状态)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # 如果是 latest_model.pth 或周期性备份
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果是 acoustic_best.pth (仅包含 model 状态)
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("模型加载完成，已进入评估模式。")
    return model


def plot_mel_comparison(mel_pred, mel_target, save_path, title_suffix=""):
    """
    绘制预测梅尔谱与目标梅尔谱的对比图。
    mel_pred: (T, mel_bins) numpy array
    mel_target: (T, mel_bins) numpy array
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 设置 Seaborn 样式让图表更好看
    sns.set_style("white")
    
    # 绘制目标梅尔谱 (原始录音)
    im1 = axes.imshow(mel_target.T, aspect='auto', origin='lower', cmap='inferno')
    axes.set_title(f"Target Mel-Spectrogram (Ground Truth) {title_suffix}", fontsize=14)
    axes.set_ylabel("Mel Bin", fontsize=12)
    fig.colorbar(im1, ax=axes, format='%+2.1f dB')
    
    # 绘制预测梅尔谱 (模型生成)
    im2 = axes.imshow(mel_pred.T, aspect='auto', origin='lower', cmap='inferno')
    axes.set_title(f"Generated Mel-Spectrogram (Prediction) {title_suffix}", fontsize=14)
    axes.set_xlabel("Frame (Time)", fontsize=12)
    axes.set_ylabel("Mel Bin", fontsize=12)
    fig.colorbar(im2, ax=axes, format='%+2.1f dB')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"梅尔谱对比图已保存至: {save_path}")


def inference_and_visualize(checkpoint_name='acoustic_best.pth', num_samples=3):
    """运行推理并生成可视化结果"""
    device = torch.device(config.DEVICE)
    print(f"梅尔谱推理启动! 设备: {device}")

    # 1. 确定路径
    # 这里假设脚本在项目根目录，如果不是，需要调整
    val_json = config.VAL_JSON
    mel_dir = config.ACOUSTIC_MEL_DIR
    bert_feat_dir = config.ACOUSTIC_BERT_FEAT_DIR
    prosody_feat_dir = config.ACOUSTIC_PROSODY_FEAT_DIR
    run_name = config.ACOUSTIC_RUN_NAME
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, run_name, checkpoint_name)
    output_dir = os.path.join(config.BASE_DIR, 'inference_results')
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载模型
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}。请先运行训练或确认路径。")
        return
    model = load_model(checkpoint_path, device)

    # 3. 加载验证集数据（用作对比）
    # 如果没有 val_json，就用 train_json
    data_json = val_json if os.path.exists(val_json) else config.TRAIN_JSON
    if not os.path.exists(data_json):
        print(f"错误: 找不到数据 JSON 文件 {data_json}。")
        return
        
    # 我们只需要取几个样本，所以 batch_size 设小一点，且不 shuffle
    data_loader = create_dataloader(
        json_path=data_json,
        mel_dir=mel_dir,
        bert_feat_dir=bert_feat_dir,
        prosody_feat_dir=prosody_feat_dir,
        batch_size=1,  # 逐条推理
        shuffle=False, # 顺序读取，方便对应
        num_workers=0
    )

    print(f"--- 开始为前 {num_samples} 个样本生成梅尔谱 ---")
    
    processed_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if processed_count >= num_samples:
                break
                
            # 搬运到设备
            bert_feat = batch['bert_feat'].to(device)
            prosody_feat = batch['prosody_feat'].to(device)
            emotion_id = batch['emotion_id'].to(device)
            durations = batch['durations'].to(device)
            mel_target = batch['mel'].to(device) # (1, T_mel, mel_bins)
            mel_mask = batch['mel_mask'].to(device)
            
            # 模型推理
            mel_pred = model(bert_feat, prosody_feat, emotion_id, durations) # (1, T_pred, mel_bins)
            
            # 提取数据到 numpy，去掉 batch 维度
            # 注意：MyBuddyLoss 里的自动对齐逻辑是针对计算 Loss 的。
            # 这里我们直接取各自的长度，通过 plt 的 aspect='auto' 自动在视觉上对齐。
            mel_pred_np = denormalize_mel(mel_pred.squeeze(0)).cpu().numpy() # (T_pred, 128)
            mel_target_np = denormalize_mel(mel_target.squeeze(0)).cpu().numpy() # (T_mel, 128)
            
            # 还原之前的 Log 缩放：(x+4)/2 -> x
            # 如果你在 prepare_data 里做了这个归一化，这里可以反向还原，让 dB 数值更真实
            # 如果 Loss 计算用的是归一化的值，可视化的值其实不影响相对对比。
            # 建议不还原，直接用归一化的值看相对结构更安全。
            # mel_pred_np = mel_pred_np * 2.0 - 4.0
            # mel_target_np = mel_target_np * 2.0 - 4.0

            # 生成保存文件名
            sample_id = f"sample_{processed_count+1}"
            save_name = f"acoustic_{checkpoint_name.replace('.pth', '')}_{sample_id}.png"
            save_path = os.path.join(output_dir, save_name)
            
            # 绘图
            plot_mel_comparison(
                mel_pred_np, 
                mel_target_np, 
                save_path, 
                title_suffix=f"({sample_id})"
            )
            
            processed_count += 1

    print(f"--- 所有 {num_samples} 个样本的可视化已完成。---")
    print(f"请查看文件夹: {output_dir}")


if __name__ == '__main__':
    # 你可以通过修改checkpoint_name来测试不同的模型节点
    # 默认加载 best 模型，也可以改为 'latest_model.pth' 或 'epoch_10.pth'
    inference_and_visualize(checkpoint_name='acoustic_best.pth', num_samples=3)
