import os
import sys
import json
import torch
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config

from utils.audio_tools import load_audio, waveform_to_bigvgan_log_mel
from models.bert_brain.model import TextEncoder
from models.bert_brain.prosody_predictor import ProsodyPredictor
from transformers import BertTokenizer

def prepare_data(json_path, output_mel_dir, output_bert_dir, output_prosody_dir):
    """
    一键生成声学模型所需的特征包 (.pt)
    """
    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(output_bert_dir, exist_ok=True)
    os.makedirs(output_prosody_dir, exist_ok=True)

    # 加载数据索引
    if not os.path.exists(json_path):
        print(f"找不到索引文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"准备处理 {len(data)} 条样本...")

    device = config.DEVICE
    print(f"使用设备: {device}")

    # 初始化模型并加载第一部分预训练权重
    # TODO
    text_encoder = TextEncoder().to(device)
    prosody_predictor = ProsodyPredictor().to(device)
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        config.BERT_BRAIN_RUN_NAME,
        'latest_model.pth'
    ) 
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 兼容两种保存格式
        if isinstance(checkpoint, dict) and 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            if 'prosody_predictor_state_dict' in checkpoint:
                prosody_predictor.load_state_dict(checkpoint['prosody_predictor_state_dict'])
        else:
            text_encoder.load_state_dict(checkpoint)
        print(f"成功加载预训练 BERT 编码器: {checkpoint_path}")
    else:
        print(f"未找到预训练权重: {checkpoint_path}，请确认第一部分训练已完成！")
        sys.exit(1)  

    text_encoder.eval()
    prosody_predictor.eval()
    tokenizer = BertTokenizer.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR)


    for item in tqdm(data, desc="特征提取中"):
        audio_path = item['audio_filepath']
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(config.RAW_DATA_DIR, os.path.basename(audio_path))

        audio_id = os.path.splitext(os.path.basename(audio_path))[0]

        # ---------- 梅尔谱提取 ----------
        mel_pt_path = os.path.join(output_mel_dir, f"{audio_id}.pt")
        if not os.path.exists(mel_pt_path):
            try:
                waveform, sr = load_audio(audio_path)
                _, log_mel = waveform_to_bigvgan_log_mel(
                    waveform,
                    n_mels=config.MEL_BINS,
                )
                mel_tensor = torch.from_numpy(log_mel.T).float()
                torch.save(mel_tensor, mel_pt_path)
            except Exception as e:
                print(f"音频 {audio_id} 处理失败: {e}")
                continue

        # ---------- BERT 特征提取 ----------
        bert_pt_path = os.path.join(output_bert_dir, f"{audio_id}.pt")
        if not os.path.exists(bert_pt_path):
            phonemes = item['phonemes'].split()
            num_phones = len(phonemes)
            
            encoded = tokenizer(
                item['text'],
                padding=False,
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attn_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                # bert_feat 形状: (1, T_token, 256)
                bert_feat = text_encoder(input_ids, attn_mask)
            
            # 裁剪 [CLS] 和 [SEP]
            final_feat = bert_feat.squeeze(0)[1:-1, :] 
            
            # 长度强制对齐逻辑
            if final_feat.size(0) != num_phones:
                if final_feat.size(0) > num_phones:
                    final_feat = final_feat[:num_phones, :]
                else:
                    # 确保 padding 在正确的设备和类型上
                    padding = torch.zeros(num_phones - final_feat.size(0), 256, device=final_feat.device)
                    final_feat = torch.cat([final_feat, padding], dim=0)

            torch.save(final_feat.cpu(), bert_pt_path)

        prosody_pt_path = os.path.join(output_prosody_dir, f"{audio_id}.pt")
        if not os.path.exists(prosody_pt_path):
            phonemes = item['phonemes'].split()
            num_phones = len(phonemes)

            encoded = tokenizer(
                item['text'],
                padding=False,
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attn_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                text_feat = text_encoder(input_ids, attn_mask)
                final_feat = text_feat.squeeze(0)[1:-1, :]
                if final_feat.size(0) != num_phones:
                    if final_feat.size(0) > num_phones:
                        final_feat = final_feat[:num_phones, :]
                    else:
                        padding = torch.zeros(num_phones - final_feat.size(0), 256, device=final_feat.device)
                        final_feat = torch.cat([final_feat, padding], dim=0)

                prosody_outputs = prosody_predictor(final_feat.unsqueeze(0))
                pause_prob = torch.softmax(prosody_outputs['pause_logits'], dim=-1)[0, :, 1:2]
                breath_prob = torch.softmax(prosody_outputs['breath_logits'], dim=-1)[0, :, 1:2]
                pause_duration = prosody_outputs['duration_preds'][0].unsqueeze(-1)
                token_duration = prosody_outputs['token_duration_preds'][0].unsqueeze(-1)
                event_probs = torch.sigmoid(prosody_outputs['event_logits'][0])
                emotion_probs = torch.softmax(prosody_outputs['emotion_logits'], dim=-1)
                emotion_probs = emotion_probs.unsqueeze(1).expand(-1, final_feat.size(0), -1).squeeze(0)

                prosody_feat = torch.cat(
                    [
                        pause_prob,
                        breath_prob,
                        pause_duration,
                        token_duration,
                        event_probs,
                        emotion_probs,
                    ],
                    dim=-1,
                )
                torch.save(prosody_feat.cpu(), prosody_pt_path)

    print(f"预处理完成！特征存放在: {output_mel_dir}、{output_bert_dir} 和 {output_prosody_dir}")

if __name__ == '__main__':
    # TODO
    json_file = config.FULL_DATA_JSON
    mel_out = config.ACOUSTIC_MEL_DIR
    bert_out = config.ACOUSTIC_BERT_FEAT_DIR
    prosody_out = config.ACOUSTIC_PROSODY_FEAT_DIR

    prepare_data(json_file, mel_out, bert_out, prosody_out)
