import os
import sys
import time
import json
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 路径对齐
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import config
from models.conformer.model import MyBuddyModel
from models.conformer.loss import MyBuddyLoss
from models.conformer.data_loader import create_dataloader

def train():

    device = torch.device(config.DEVICE)
    print(f"My Buddy 声学模型启动! 设备: {device}")

    # TODO
    train_json = os.path.join(config.PROCESSED_DATA_DIR, 'processed_1000train_data.json') 
    val_json = os.path.join(config.PROCESSED_DATA_DIR, 'processed_1000val_data.json')
    
    mel_dir = os.path.join(config.PROCESSED_DATA_DIR, 'mel1000_v2')
    bert_feat_dir = os.path.join(config.PROCESSED_DATA_DIR, 'bert_feat1000_v2')
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'conformer_v2_1000')
    log_dir = os.path.join(config.LOG_DIR, 'conformer')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

  
    train_loader = create_dataloader(
        json_path=train_json,
        mel_dir=mel_dir,
        bert_feat_dir=bert_feat_dir,
        batch_size=getattr(config, 'BATCH_SIZE', 16),
        shuffle=True,
        num_workers=0
    )

    val_loader = None
    if os.path.exists(val_json):
        val_loader = create_dataloader(
            json_path=val_json,
            mel_dir=mel_dir,
            bert_feat_dir=bert_feat_dir,
            batch_size=getattr(config, 'BATCH_SIZE', 16),
            shuffle=False,
            num_workers=0
        )
        print(f"验证集已就绪: {len(val_loader.dataset)} 条样本")
    else:
        print("提醒: 未找到 val_data.json, 将仅在训练集上运行。")

    # 初始化模型、损失、优化器
    model = MyBuddyModel(config).to(device)
    criterion = MyBuddyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=getattr(config, 'LEARNING_RATE', 2e-4), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

 
    global_step = 0
    best_val_loss = float('inf')
    epochs = getattr(config, 'EPOCHS', 200)

    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in pbar:
            bert_feat = batch['bert_feat'].to(device)
            emotion_id = batch['emotion_id'].to(device)
            durations = batch['durations'].to(device)
            mel_target = batch['mel'].to(device)
            mel_mask = batch['mel_mask'].to(device)

            optimizer.zero_grad()
            mel_pred = model(bert_feat, emotion_id, durations)

            # 长度对齐
            t_min = min(mel_pred.size(1), mel_target.size(1))
            mel_pred = mel_pred[:, :t_min, :]
            mel_target = mel_target[:, :t_min, :]
            mel_mask = mel_mask[:, :t_min]

            loss_dict = criterion(mel_pred, mel_target, mel_mask)
            loss_dict['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss_dict['total'].item()
            global_step += 1
            pbar.set_postfix({'loss': f"{loss_dict['total'].item():.4f}"})

            if global_step % 10 == 0:
                writer.add_scalar('Step/TotalLoss', loss_dict['total'].item(), global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Epoch/TrainLoss', avg_train_loss, epoch)

        # --- 验证阶段 ---
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    mel_pred = model(batch['bert_feat'].to(device), 
                                     batch['emotion_id'].to(device), 
                                     batch['durations'].to(device))
                    
                    # 验证集同样需要对齐
                    t_min = min(mel_pred.size(1), batch['mel'].size(1))
                    loss_dict = criterion(mel_pred[:, :t_min, :], 
                                         batch['mel'][:, :t_min, :].to(device), 
                                         batch['mel_mask'][:, :t_min].to(device))
                    total_val_loss += loss_dict['total'].item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Epoch/ValLoss', avg_val_loss, epoch)
            print(f"Epoch {epoch} 验证 Loss: {avg_val_loss:.4f}")

            # 根据验证集保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'conformer_best.pth'))
                print(f"发现更好的验证效果，已更新最佳模型。")

        # 调度器步进
        scheduler.step()

        # 定期保存检查点
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'my_buddy_epoch_{epoch}.pth'))

    writer.close()
    print("训练流程圆满结束！")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练已手动停止。")