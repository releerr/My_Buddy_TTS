import os
import sys
import time
import json
import gc
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 路径对齐
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import config
from models.acoustic.model import build_acoustic_model
from models.acoustic.loss import MyBuddyLoss
from models.acoustic.data_loader import create_dataloader


def resolve_stage_settings():
    stage = os.environ.get(
        "ACOUSTIC_STAGE",
        getattr(config, "ACOUSTIC_STAGE", "full"),
    ).strip().lower()
    strategy = os.environ.get(
        "ACOUSTIC_SUBSET_STRATEGY",
        getattr(config, "ACOUSTIC_SUBSET_STRATEGY", "shortest"),
    ).strip().lower()

    if stage == "short":
        return {
            "stage": stage,
            "run_name": getattr(config, "ACOUSTIC_SHORT_RUN"),
            "max_train_samples": getattr(config, "ACOUSTIC_SHORT_SAMPLES"),
            "max_val_samples": getattr(config, "ACOUSTIC_SHORT_SAMPLES"),
            "epochs": getattr(config, "ACOUSTIC_SHORT_EPOCHS"),
            "sample_strategy": strategy,
            "init_run_name": None,
        }
    if stage == "medium":
        return {
            "stage": stage,
            "run_name": getattr(config, "ACOUSTIC_MEDIUM_RUN"),
            "max_train_samples": getattr(config, "ACOUSTIC_MEDIUM_SAMPLES"),
            "max_val_samples": getattr(config, "ACOUSTIC_MEDIUM_SAMPLES"),
            "epochs": getattr(config, "ACOUSTIC_MEDIUM_EPOCHS"),
            "sample_strategy": strategy,
            "init_run_name": getattr(config, "ACOUSTIC_SHORT_RUN"),
        }
    return {
        "stage": "full",
        "run_name": getattr(config, "ACOUSTIC_FULL_RUN"),
        "max_train_samples": None,
        "max_val_samples": None,
        "epochs": getattr(config, "ACOUSTIC_FULL_EPOCHS"),
        "sample_strategy": strategy,
        "init_run_name": getattr(config, "ACOUSTIC_MEDIUM_RUN"),
    }

def train():

    device = torch.device(config.DEVICE)
    print(f"Attention Acoustic v2 声学模型启动! 设备: {device}")

    # TODO
    train_json = config.TRAIN_JSON
    val_json = config.VAL_JSON
    stage_settings = resolve_stage_settings()
    
    mel_dir = config.ACOUSTIC_MEL_DIR
    bert_feat_dir = config.ACOUSTIC_BERT_FEAT_DIR
    prosody_feat_dir = config.ACOUSTIC_PROSODY_FEAT_DIR
    run_name = os.environ.get("ACOUSTIC_RUN_NAME", stage_settings["run_name"])
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    log_dir = os.path.join(config.LOG_DIR, run_name)
    
    print(f"正在确认：日志将保存到这个绝对路径: {os.path.abspath(log_dir)}")
    print(f"检查点将保存到这个绝对路径: {os.path.abspath(checkpoint_dir)}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

  
    batch_size = int(os.environ.get("ACOUSTIC_BATCH_SIZE", getattr(config, 'ACOUSTIC_BATCH_SIZE', 8)))
    num_workers = int(os.environ.get("ACOUSTIC_NUM_WORKERS", str(getattr(config, 'ACOUSTIC_NUM_WORKERS', 2))))
    val_interval = int(os.environ.get("ACOUSTIC_VAL_INTERVAL", str(getattr(config, 'ACOUSTIC_VAL_INTERVAL', 2))))
    log_interval = int(os.environ.get("ACOUSTIC_LOG_INTERVAL", str(getattr(config, 'ACOUSTIC_LOG_INTERVAL', 50))))
    checkpoint_interval = int(os.environ.get("ACOUSTIC_CKPT_INTERVAL", str(getattr(config, 'ACOUSTIC_CKPT_INTERVAL', 10))))
    max_train_samples_env = os.environ.get("ACOUSTIC_MAX_TRAIN_SAMPLES")
    max_val_samples_env = os.environ.get("ACOUSTIC_MAX_VAL_SAMPLES")
    overfit_mode = os.environ.get("ACOUSTIC_OVERFIT_MODE", "0") == "1"
    audio_ids_env = os.environ.get("ACOUSTIC_AUDIO_IDS", "").strip()
    selected_audio_ids = [x.strip() for x in audio_ids_env.split(",") if x.strip()]
    max_train_samples = int(max_train_samples_env) if max_train_samples_env else None
    max_val_samples = int(max_val_samples_env) if max_val_samples_env else None
    sample_strategy = os.environ.get("ACOUSTIC_SUBSET_STRATEGY", stage_settings["sample_strategy"])
    bucket_by_length = os.environ.get(
        "ACOUSTIC_BUCKET_BY_LENGTH",
        "1" if getattr(config, 'ACOUSTIC_BUCKET_BY_LENGTH', True) else "0"
    ) != "0"

    if overfit_mode:
        if max_train_samples is None and not selected_audio_ids:
            max_train_samples = 16
        if max_val_samples is None:
            max_val_samples = max_train_samples
        if val_interval < 1:
            val_interval = 1
        if "ACOUSTIC_LR" not in os.environ:
            learning_rate_override = 5e-4
        else:
            learning_rate_override = None
        print("已开启小样本过拟合调试模式。")
    else:
        learning_rate_override = None
        if max_train_samples is None:
            max_train_samples = stage_settings["max_train_samples"]
        if max_val_samples is None:
            max_val_samples = stage_settings["max_val_samples"]

    train_loader = create_dataloader(
        json_path=train_json,
        mel_dir=mel_dir,
        bert_feat_dir=bert_feat_dir,
        prosody_feat_dir=prosody_feat_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        bucket_by_length=bucket_by_length,
        max_samples=max_train_samples,
        audio_ids=selected_audio_ids or None,
        sample_strategy=sample_strategy,
    )

    val_loader = None
    if overfit_mode:
        val_loader = create_dataloader(
            json_path=train_json,
            mel_dir=mel_dir,
            bert_feat_dir=bert_feat_dir,
            prosody_feat_dir=prosody_feat_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            bucket_by_length=bucket_by_length,
            max_samples=max_val_samples,
            audio_ids=selected_audio_ids or None,
            sample_strategy=sample_strategy,
        )
        print(f"过拟合验证集已就绪: {len(val_loader.dataset)} 条样本（与训练集同源）")
    elif os.path.exists(val_json):
        val_loader = create_dataloader(
            json_path=val_json,
            mel_dir=mel_dir,
            bert_feat_dir=bert_feat_dir,
            prosody_feat_dir=prosody_feat_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            bucket_by_length=bucket_by_length,
            max_samples=max_val_samples,
            audio_ids=selected_audio_ids or None,
            sample_strategy=sample_strategy,
        )
        print(f"验证集已就绪: {len(val_loader.dataset)} 条样本")
    else:
        print("提醒: 未找到 val_data.json, 将仅在训练集上运行。")

    # 初始化模型、损失、优化器
    model = build_acoustic_model(config).to(device)
    criterion = MyBuddyLoss().to(device)
    learning_rate = float(os.environ.get("ACOUSTIC_LR", str(getattr(config, 'ACOUSTIC_LEARNING_RATE', 2e-4))))
    if learning_rate_override is not None:
        learning_rate = learning_rate_override
    weight_decay = 0.0 if overfit_mode else 1e-2
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = []
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    epochs = int(os.environ.get("ACOUSTIC_EPOCHS", str(stage_settings["epochs"])))
    history_path = os.path.join(log_dir, 'train_history.json')
    resume_path = os.path.join(checkpoint_dir, 'latest_model.pth')

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            if history:
                # 既然是从 1 开始存，最后一条的 epoch 值就是我们跑完的轮数
                start_epoch = history[-1]['epoch'] 
                print(f"检测到历史记录，已完成 {start_epoch} 轮，将从第 {start_epoch + 1} 轮开始...")

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        # 确保 best_val_loss 同步，否则保存最佳模型会出错
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"成功恢复断点权重。当前最佳验证 Loss: {best_val_loss:.4f}")
    else:
        init_run_name = os.environ.get("ACOUSTIC_INIT_RUN_NAME", stage_settings["init_run_name"] or "").strip()
        init_ckpt_name = os.environ.get("ACOUSTIC_INIT_CKPT_NAME", "latest_model.pth").strip()
        if init_run_name:
            init_path = os.path.join(config.CHECKPOINT_DIR, init_run_name, init_ckpt_name)
            if os.path.exists(init_path):
                init_ckpt = torch.load(init_path, map_location=device)
                init_state = init_ckpt['model_state_dict'] if isinstance(init_ckpt, dict) and 'model_state_dict' in init_ckpt else init_ckpt
                model.load_state_dict(init_state)
                print(f"已从上一课程阶段初始化权重: {init_path}")

    print(
        f"训练配置: batch_size={batch_size}, num_workers={num_workers}, "
        f"bucket_by_length={bucket_by_length}, val_interval={val_interval}, "
        f"train_samples={len(train_loader.dataset)}, "
        f"stage={stage_settings['stage']}, sample_strategy={sample_strategy}"
    )
    if selected_audio_ids:
        print(f"指定样本 ID: {selected_audio_ids}")


    for epoch in range(start_epoch + 1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch in pbar:
            bert_feat = batch['bert_feat'].to(device)
            prosody_feat = batch['prosody_feat'].to(device)
            emotion_id = batch['emotion_id'].to(device)
            durations = batch['durations'].to(device)
            mel_target = batch['mel'].to(device)
            mel_mask = batch['mel_mask'].to(device)
            token_mask = (durations > 0).float()

            optimizer.zero_grad(set_to_none=True)
            mel_pred = model(bert_feat, prosody_feat, emotion_id, durations)
            attn_weights = getattr(model, "latest_attn", None)

            # # 长度对齐
            # t_min = min(mel_pred.size(1), mel_target.size(1))
            # mel_pred = mel_pred[:, :t_min, :]
            # mel_target = mel_target[:, :t_min, :]
            # mel_mask = mel_mask[:, :t_min]

            loss_dict = criterion(mel_pred, mel_target, mel_mask, attn_weights=attn_weights, token_mask=token_mask)
            loss_dict['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_train_loss += loss_dict['total'].item()
            global_step += 1
            pbar.set_postfix({'loss': f"{loss_dict['total'].item():.4f}"})

            if global_step % log_interval == 0:
                writer.add_scalar('Step/TotalLoss', loss_dict['total'].item(), global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Epoch/TrainLoss', avg_train_loss, epoch)

        # --- 验证阶段 ---
        avg_val_loss = None
        if val_loader is not None and epoch % val_interval == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    v_bert = batch['bert_feat'].to(device)
                    v_prosody = batch['prosody_feat'].to(device)
                    v_emo = batch['emotion_id'].to(device)
                    v_dur = batch['durations'].to(device)
                    v_mel = batch['mel'].to(device)
                    v_mask = batch['mel_mask'].to(device)
                    v_token_mask = (v_dur > 0).float()
                    v_pred = model(v_bert, v_prosody, v_emo, v_dur)
                    v_attn = getattr(model, "latest_attn", None)
                    # 验证集同样需要对齐
                    # t_min = min(mel_pred.size(1), batch['mel'].size(1))
                    loss_dict = criterion(v_pred, v_mel, v_mask, attn_weights=v_attn, token_mask=v_token_mask)
                    total_val_loss += loss_dict['total'].item()

            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar('Epoch/ValLoss', avg_val_loss, epoch)
            print(f"Epoch {epoch} 验证 Loss: {avg_val_loss:.4f}")

        # 调度器步进
        if not overfit_mode:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss if val_loader else None,
            "lr": current_lr
        }

        history.append(epoch_log)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"✨ [Epoch {epoch}] 训练日志已成功更新到: {history_path}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': avg_train_loss
        }

        torch.save(checkpoint_data, resume_path)

        if val_loader is not None and avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss # 更新最佳值
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'acoustic_best.pth'))
            print(f"发现更优模型！已保存至 acoustic_best.pth")

        # 定期保存检查点
        if epoch % checkpoint_interval == 0:
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
            print(f"--- 已进行周期性备份: epoch_{epoch}.pth ---")
        
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    writer.close()
    with open(os.path.join(log_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    print("训练流程圆满结束！最终模型与历史日志已保存。")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练已手动停止。")
