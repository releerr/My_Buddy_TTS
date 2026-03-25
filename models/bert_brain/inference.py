import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from model import TextEncoder
from prosody_predictor import ProsodyPredictor
from utils.audio_tools import load_audio, waveform_to_mel
from utils.data_loader import TTSDataset

class BuddyInference:
    def __init__(self, checkpoint_path=None):
        self.device = config.DEVICE
        # tokenizer (use tokenizer, not the model)
        self.tokenizer = BertTokenizer.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR)
        # instantiate TextEncoder then move to device
        self.text_encoder = TextEncoder().to(self.device)
        self.prosody_predictor = ProsodyPredictor().to(self.device)

        # load weights
        # TODO
        # ckpt_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, 'bert_brain_checkpoint_v4_5000', 'best_brain_model.pth')
        # ckpt_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, 'bert_brain_checkpoint_v4_5000', 'epoch_10.pth')
        ckpt_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, 'bert_brain_checkpoint_v4_5000', 'latest_model.pth')

        print(f"--- 正在加载模型: {ckpt_path} ---")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # load state dicts (if keys exist)
        if 'text_encoder_state_dict' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        if 'prosody_predictor_state_dict' in checkpoint:
            self.prosody_predictor.load_state_dict(checkpoint['prosody_predictor_state_dict'])

        self.text_encoder.eval()
        self.prosody_predictor.eval()

        # mapping
        from utils.data_loader import EMOTION_MAP, EVENT_MAP
        self.id_to_emotion = {v: k for k, v in EMOTION_MAP.items()}
        self.id_to_event = {v: k for k, v in EVENT_MAP.items()}

    def run(self, text):
        encoded = self.tokenizer(
            text,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device) 

        # convert to human readable text for futher align performance
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # forward
        with torch.no_grad():
            text_features = self.text_encoder(input_ids, attention_mask)
            preds = self.prosody_predictor(text_features)

        mask = attention_mask.squeeze(0).bool()
        valid_indices = torch.where(mask)[0].tolist()

        emo_idx = torch.argmax(preds['emotion_logits'], dim=-1).item()
        emotion_label = self.id_to_emotion.get(emo_idx, "unknown")

        token_results = []
        pause_preds = torch.argmax(preds['pause_logits'], dim=-1).squeeze(0)
        breath_preds = torch.argmax(preds['breath_logits'], dim=-1).squeeze(0)
        event_probs = torch.sigmoid(preds['event_logits']).squeeze(0)

        for i in valid_indices:
            token_text = tokens[i]
            if token_text in ['[CLS]', '[SEP]', '[PAD]']: continue
            # multi-label
            active_events = []
            for e_idx, prob in enumerate(event_probs[i]):
                if prob > 0.5:
                    active_events.append(self.id_to_event.get(e_idx, f"evt_{e_idx}"))
            
            token_results.append({
                'token': token_text,
                'pause': pause_preds[i].item(),
                'duration': round(preds['duration_preds'][0][i].item(), 3),
                'breath': breath_preds[i].item(),
                'events': active_events
            })
        return {
            'text': text,
            'emotion': emotion_label,
            'token_details': token_results
        }

# for command line
def main():
    import argparse
    parser = argparse.ArgumentParser(description="My Buddy Brain Inference")

    # python inference.py --text "今天天气真好" --checkpoint best_brain_model.pth
    parser.add_argument('--text', type=str, required=True, help='输入要预测的文本')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型路径')
    args = parser.parse_args()

    engine = BuddyInference(args.checkpoint)
    result = engine.run(args.text)

 
    print(f"\n" + "="*40)
    print(f"输入文本: {result['text']}")
    print(f"预测情感: 【{result['emotion']}】")
    print("-" * 40)
    print(f"{'Token':<10} | {'停顿':<4} | {'时长':<6} | {'呼吸':<4} | {'事件'}")
    for item in result['token_details']:
        evt_str = ",".join(item['events']) if item['events'] else "-"
        print(f"{item['token']:<10} | {item['pause']:<6} | {item['duration']:<8} | {item['breath']:<6} | {evt_str}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()





