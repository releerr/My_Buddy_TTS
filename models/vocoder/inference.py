import os
import sys
import json
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import config
from models.bert_brain.model import TextEncoder
from models.conformer.model import MyBuddyModel
from models.bert_brain.prosody_predictor import ProsodyPredictor    


def load_all_models(vocoder_path):
    device = torch.device(config.DEVICE)
    print(f"正在加载组件... 设备: {device}")
    
    # 加载 TextEncoder
    tokenizer = BertTokenizer.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR)
    text_encoder = TextEncoder().to(device)
    # TODO
    brain_ckpt = os.path.join(config.CHECKPOINT_DIR, 'bert_brain_checkpoint_v1_1000', 'best_brain_model.pth')
    brain_data = torch.load(brain_ckpt, map_location=device)
    text_encoder.load_state_dict(brain_data['text_encoder_state_dict'] if 'text_encoder_state_dict' in brain_data else brain_data)
    text_encoder.eval()

    #加载 ProsodyPredictor
    prosody_predictor = ProsodyPredictor(input_dim=256).to(device) # 维度与 BERT 输出一致
    if 'prosody_predictor_state_dict' in brain_data:
        prosody_predictor.load_state_dict(brain_data['prosody_predictor_state_dict'])
    else:
        prosody_predictor.load_state_dict(brain_data) 
    prosody_predictor.eval()

    #  加载 Conformer
    my_buddy_model = MyBuddyModel(config).to(device)
    # TODO
    conformer_ckpt = os.path.join(config.CHECKPOINT_DIR, 'conformer_v2_1000', 'conformer_best.pth')
    my_buddy_model.load_state_dict(torch.load(conformer_ckpt, map_location=device))
    my_buddy_model.eval()

    # 加载 BigVGAN
    import bigvgan
    print(f"正在加载 Vocoder: {vocoder_path}")

    try:
        # 官方加载方式
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_path, use_cuda_kernel=False)
    except:
        # 手动加载
        with open(os.path.join(vocoder_path, "config.json")) as f:
            hparams = json.load(f)
            
        class AttrDict(dict):
            def __getattr__(self, name):
                return self[name]

        h = AttrDict(hparams)
        bigvgan_model = bigvgan.BigVGAN(h).to(device)
        state_dict = torch.load(os.path.join(vocoder_path, "bigvgan_generator.pt"), map_location=device)
        bigvgan_model.load_state_dict(state_dict['generator'] if 'generator' in state_dict else state_dict)
    
    bigvgan_model.to(device)
    bigvgan_model.remove_weight_norm()
    bigvgan_model.eval()
    
    return tokenizer, text_encoder, prosody_predictor, my_buddy_model, bigvgan_model

def synthesize(text, out_path, models):
    tokenizer, text_encoder, prosody_predictor, my_buddy_model, bigvgan_model = models
    device = torch.device(config.DEVICE)
    
    print(f"正在处理文本: {text}")
    
    with torch.inference_mode():
        # 语义特征提取
        inputs = tokenizer(text, return_tensors='pt').to(device)
        bert_feat = text_encoder(inputs['input_ids'], inputs['attention_mask'])
        bert_feat = bert_feat[:, 1:-1, :] 

        # 声学特征生成
        outputs = prosody_predictor(bert_feat)

        if isinstance(outputs, dict):
            log_dur_pred = outputs['duration_preds']
        else:
            log_dur_pred = outputs
        durations = torch.exp(log_dur_pred) - 1.0 
        durations = torch.clamp(durations, min=1.0) # 确保每个音节至少占 1 帧
        # 强制让每个字占 12 帧（约 0.14 秒/字）
        durations = (torch.ones_like(durations) * 12).long()
        print(f"预测时长序列: {durations.squeeze().tolist()}")

        emotion_id = torch.tensor([0]).to(device)
        
        # 生成原始 Mel (1, T, 128)
        mel_pred = my_buddy_model(bert_feat, emotion_id, durations) 
        # 对 Mel 能量进行整体补偿（因为模型现在输出太弱了）， 我们把数值整体往上提 10 个单位，尝试激活声码器
        mel_128 = mel_pred * 1.5 + 8.0          
        print(f"DEBUG: Mel range: {mel_128.min().item():.2f} to {mel_128.max().item():.2f}")

        # ==========================================
        # 调试代码：保存可视化频谱图
        # ==========================================
        plt.figure(figsize=(12, 6))
        spec_to_show = mel_128.squeeze().cpu().numpy().T 
        plt.imshow(spec_to_show, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar(label='Amplitude')
        plt.title(f"Debug Mel (Max: {spec_to_show.max():.2f}, Min: {spec_to_show.min():.2f})")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Bins (0-127)")
        debug_img_path = out_path.replace(".wav", "_debug.png")
        plt.savefig(debug_img_path)
        plt.close()
        print(f"频谱可视化已保存至: {debug_img_path}")
        # ==========================================

        # 嗓子合成 (BigVGAN need B, C, T)
        mel_input = mel_128.transpose(1, 2) 

        audio_output = bigvgan_model(mel_input) # (1, 1, samples)

    # 保存音频
    audio_tensor = audio_output.squeeze(0).cpu()
    
    try:
        import soundfile as sf
        audio_np = audio_tensor.numpy()
        if audio_np.ndim == 2 and audio_np.shape[0] <= 4:
            audio_np = audio_np.T
  
        sf.write(out_path, audio_np, config.SAMPLE_RATE)
        print(f"音频合成完毕: {out_path}")
    except Exception as e:
        print(f"保存失败: {e}")

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    
    VOC_PATH = config.VOC_PATH
    all_models = load_all_models(VOC_PATH)
    
    # TODO
    test_text = "你好呀，我是你的机器人好朋友。很高兴认识你！"
    output_file = os.path.join(BASE_DIR, "outputs", "my_buddy_v2_1000_test2.wav")
    synthesize(test_text, output_file, all_models)