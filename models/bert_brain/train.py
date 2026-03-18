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


# train
def train():
    device = config.DEVICE
    print(f"--- My Buddy 训练开始！设备: {device} ---")
    # TODO
    train_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_train_data.json') 
    val_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_val_data.json')
    
    train_loader = create_dataloader(
        train_path, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        load_audio_flag=True,
        compute_mel_flag=True
    )
    val_loader = create_dataloader(
        val_path, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, # 验证集不需要打乱
        load_audio_flag=True,
        compute_mel_flag=True
    )

    # init all models
    text_encoder = TextEncoder().to(device)       
    audio_encoder = AudioEncoder().to(device) 
    prosody_predictor = ProsodyPredictor().to(device) 

    criterion = ProsodyLoss().to(device)
    mse_align = nn.MSELoss() # 用于文字和音频特征的对齐损失 for aligning text and audio features

    # init optimizers
    optimizer = optim.AdamW([
        {'params': text_encoder.parameters(), 'lr': 2e-5},
        {'params': audio_encoder.parameters(), 'lr': 1e-4},
        {'params': prosody_predictor.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    # training loop
    epochs = config.EPOCHS
    for epoch in range(epochs):
        text_encoder.train()
        audio_encoder.train()
        prosody_predictor.train()

        train_metrics = {
            'total': 0, 'pause': 0, 'duration': 0, 
            'breath': 0, 'event': 0, 'emotion': 0, 'align': 0
        }

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for batch in pbar:
            # move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mel = batch['mel'].to(device) if batch['mel'] is not None else None

            # target labels for loss
            target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad() # clear gradients before backward

            #forward pass
            text_features = text_encoder(input_ids, attention_mask)  # (B, T, D) 
            
            align_loss = 0
            if mel is not None:
                audio_features = audio_encoder(mel)  # (B, T, D) 
                align_loss = mse_align(text_features, audio_features) # calculate alignment loss
            
            # predict prosody features based on text features
            predictions = prosody_predictor(text_features)  # dict of predicted features

            # calculate loss
            loss_dict = criterion(predictions, target, mask=attention_mask)

            # 最终总损失 = 韵律损失 + 对齐损失, 对齐损失给 0.5 的权重，防止带偏了预测主任务
            total_loss = loss_dict['total_loss'] + 0.5 * align_loss

            #backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0) # gradient clipping to prevent exploding gradients
            optimizer.step()

            # accumulate loss for logging
            num_batches = len(train_loader)
            train_metrics['total'] += total_loss.item() / num_batches
            train_metrics['pause'] += loss_dict['pause_loss'].item() / num_batches
            train_metrics['duration'] += loss_dict['duration_loss'].item() / num_batches
            train_metrics['breath'] += loss_dict['breath_loss'].item() / num_batches
            train_metrics['event'] += loss_dict['event_loss'].item() / num_batches
            train_metrics['emotion'] += loss_dict['emotion_loss'].item() / num_batches
            if mel is not None:
                train_metrics['align'] += align_loss.item() / num_batches
            pbar.set_postfix({'L': f"{total_loss.item():.3f}"})



        #----Valid----------
        text_encoder.eval()
        audio_encoder.eval()
        prosody_predictor.eval()

        val_metrics = {
            'total': 0, 'pause': 0, 'duration': 0, 
            'breath': 0, 'event': 0, 'emotion': 0, 'align': 0
        }

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val")
            for batch in vbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                mel = batch['mel'].to(device) if batch['mel'] is not None else None
                target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                text_features = text_encoder(input_ids, attention_mask)
                align_loss = 0
                if mel is not None:
                    audio_features = audio_encoder(mel)
                    align_loss = mse_align(text_features, audio_features)
                
                predictions = prosody_predictor(text_features)
                loss_dict = criterion(predictions, target, mask=attention_mask)
                total_loss = loss_dict['total_loss'] + 0.5 * align_loss

                # 累加验证指标
                nv = len(val_loader)
                val_metrics['total'] += total_loss.item() / nv
                val_metrics['pause'] += loss_dict['pause_loss'].item() / nv
                val_metrics['duration'] += loss_dict['duration_loss'].item() / nv
                val_metrics['breath'] += loss_dict['breath_loss'].item() / nv
                val_metrics['event'] += loss_dict['event_loss'].item() / nv
                val_metrics['emotion'] += loss_dict['emotion_loss'].item() / nv
                if mel is not None:
                    val_metrics['align'] += align_loss.item() / nv

        # history log
        epoch_log = {'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics}
        history.append(epoch_log)

        print(f"\n[Epoch {epoch+1}] 报告:")
        print(f" > Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
        print(f" > Val Details -> Pause: {val_metrics['pause']:.4f} | Dur: {val_metrics['duration']:.4f} | Breath: {val_metrics['breath']:.4f} | Event: {val_metrics['event']:.4f} | Emotion: {val_metrics['emotion']:.4f} | Align: {val_metrics['align']:.4f}")

        # 保存 Checkpoint
        # TODO
        checkpoints_dir = os.path.join(config.CHECKPOINT_DIR, 'checkpoint_v1_1000')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)

        # use valid dataset to find "best_val_loss"
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_epoch = epoch + 1
            save_path = os.path.join(checkpoints_dir, 'best_brain_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'text_encoder_state_dict': text_encoder.state_dict(),
                'audio_encoder_state_dict': audio_encoder.state_dict(),
                'prosody_predictor_state_dict': prosody_predictor.state_dict(),
                'metrics': val_metrics
            }, save_path)
            print(f"发现更优模型 (Val Loss)! 已更新至 best_brain_model.pth")

    print(f"\n--- 训练总结 ---")
    print(f"最佳轮次是 Epoch {best_epoch}, 最小验证 Loss: {best_val_loss:.4f}")
    
    with open(os.path.join(checkpoints_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=4)



if __name__ == "__main__":
    train()


