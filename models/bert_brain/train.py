import json
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from utils.data_loader import create_dataloader
from model import TextEncoder
from prosody_predictor import ProsodyPredictor
from loss import ProsodyLoss
from audio_encoder import AudioEncoder

def train():
    device = config.DEVICE
    print(f"--- My Buddy 训练开始！设备: {device} ---")
    
    # TODO
    checkpoints_dir = os.path.join(config.CHECKPOINT_DIR, 'bert_brain_checkpoint_v4_5000')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 加载数据
    # TODO
    train_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_train_data.json') 
    val_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_val_data.json')
    
    train_loader = create_dataloader(train_path, batch_size=config.BATCH_SIZE, shuffle=True, load_audio_flag=True, compute_mel_flag=True)
    val_loader = create_dataloader(val_path, batch_size=config.BATCH_SIZE, shuffle=False, load_audio_flag=True, compute_mel_flag=True)

    # 初始化模型
    text_encoder = TextEncoder().to(device)       
    audio_encoder = AudioEncoder().to(device) 
    prosody_predictor = ProsodyPredictor().to(device) 
    criterion = ProsodyLoss().to(device)
    mse_align = nn.MSELoss()

    optimizer = optim.AdamW([
        {'params': text_encoder.parameters(), 'lr': config.LEARNING_RATE_BERT},
        {'params': audio_encoder.parameters(), 'lr': config.LEARNING_RATE_HEAD},
        {'params': prosody_predictor.parameters(), 'lr': config.LEARNING_RATE_HEAD}
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    #  --- 断点续练逻辑 ---
    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    
    # 尝试加载历史记录
    history_path = os.path.join(checkpoints_dir, 'train_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            if history:
                start_epoch = history[-1]['epoch']
                print(f"检测到历史记录，将从第 {start_epoch + 1} 轮继续...")

    # 尝试加载最近的一个模型（可以是 best，也可以是定期保存的）
    resume_path = os.path.join(checkpoints_dir, 'latest_model.pth') # 我们增加一个最新模型记录
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        prosody_predictor.load_state_dict(checkpoint['prosody_predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"成功加载断点：Epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        text_encoder.train()
        audio_encoder.train()
        prosody_predictor.train()

        train_metrics = {'total': 0, 'pause': 0, 'duration': 0, 'breath': 0, 'event': 0, 'emotion': 0, 'align': 0}
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Train")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mel = batch['mel'].to(device) if batch['mel'] is not None else None
            target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            text_features = text_encoder(input_ids, attention_mask)
            
            align_loss = 0
            if mel is not None:
                audio_features = audio_encoder(mel, target_len=text_features.size(1))
                align_loss = mse_align(text_features, audio_features)
            
            predictions = prosody_predictor(text_features)
            loss_dict = criterion(predictions, target, mask=attention_mask)
            total_loss = loss_dict['total_loss'] + 0.1 * align_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(text_encoder.parameters()) + list(audio_encoder.parameters()) + list(prosody_predictor.parameters()), 1.0)
            optimizer.step()

            # 统计
            nb = len(train_loader)
            for k in train_metrics:
                if k == 'total': train_metrics[k] += total_loss.item() / nb
                elif k == 'align': train_metrics[k] += (align_loss.item() if mel is not None else 0) / nb
                else: train_metrics[k] += loss_dict[f'{k}_loss'].item() / nb
            pbar.set_postfix({'L': f"{total_loss.item():.3f}"})

        # 验证
        text_encoder.eval()
        audio_encoder.eval()
        prosody_predictor.eval()
        val_metrics = {'total': 0, 'pause': 0, 'duration': 0, 'breath': 0, 'event': 0, 'emotion': 0, 'align': 0}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Val"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                mel = batch['mel'].to(device) if batch['mel'] is not None else None
                target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                text_features = text_encoder(input_ids, attention_mask)
                align_loss = 0
                if mel is not None:
                    audio_features = audio_encoder(mel, target_len=text_features.size(1))
                    if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
                        print("警告：发现异常音频特征，跳过本次对齐计算")
                        align_loss = 0
                    else:
                        align_loss = mse_align(text_features, audio_features)
                
                predictions = prosody_predictor(text_features)
                loss_dict = criterion(predictions, target, mask=attention_mask)
                total_loss = loss_dict['total_loss'] + 0.1 * align_loss

                nv = len(val_loader)
                for k in val_metrics:
                    if k == 'total': val_metrics[k] += total_loss.item() / nv
                    elif k == 'align': val_metrics[k] += (align_loss.item() if mel is not None else 0) / nv
                    else: val_metrics[k] += loss_dict[f'{k}_loss'].item() / nv

        # 保存逻辑
        epoch_log = {'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics}
        history.append(epoch_log)
        scheduler.step(val_metrics['total'])
        
        # 获取当前的学习率（用于记录，方便观察它降了没）
        current_lr = optimizer.param_groups[0]['lr']
        epoch_log['lr'] = current_lr
        print(f" > 当前 BERT 学习率: {current_lr:.6f}")

        # 立即更新 JSON 日志文件
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"\n[Epoch {epoch+1}] Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")

        # 构建通用保存字典
        checkpoint_data = {
            'epoch': epoch + 1,
            'text_encoder_state_dict': text_encoder.state_dict(),
            'audio_encoder_state_dict': audio_encoder.state_dict(),
            'prosody_predictor_state_dict': prosody_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics': val_metrics
        }

        # 保存最新模型（用于断点恢复）
        torch.save(checkpoint_data, os.path.join(checkpoints_dir, 'latest_model.pth'))

        # 保存最佳模型
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save(checkpoint_data, os.path.join(checkpoints_dir, 'best_brain_model.pth'))
            print(f"发现更优模型! Val Loss: {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint_data, os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth'))
            print(f"已进行周期性备份: epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()