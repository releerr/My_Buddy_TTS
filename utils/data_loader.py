import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import sys
import os
from utils.audio_tools import load_audio, waveform_to_mel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'surprised': 4,
    'interrogative': 5
}

EVENT_MAP = {
    'laugh': 0, 'cry': 1, 'sigh': 2, 'cough': 3, 'sneeze': 4, 
    'yawn': 5, 'hesitation': 6, 'question': 7, 'surprise': 8, 'applause': 9
}

class TTSDataset(Dataset):
    def __init__(self, data_path, max_text_len=config.MAX_LENGTH, load_audio_flag=True, compute_mel_flag=True):
        self.max_text_len = max_text_len
        self.load_audio_flag = load_audio_flag
        self.compute_mel_flag = compute_mel_flag

        with open(data_path, 'r',encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR) #使用本地BERT模型的tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        phonemes = item.get('phonemes', '').split()
        if phonemes:
            assert len(phonemes) == len(text), f"---字符与拼音长度不符 ID: {item.get('id')}"

        # text tokenization get tensor after padding
        # and alignment to token level
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        input_ids = encoded['input_ids'].squeeze(0)             
        attention_mask = encoded['attention_mask'].squeeze(0)
        offsets = encoded['offset_mapping'].squeeze(0)

        seq_len = len(input_ids)

        # token-level event labels (initialized after seq_len is known)
        event_labels = torch.zeros(seq_len, len(EVENT_MAP), dtype=torch.float)

        # itialize tokens-level labels with default values
        # 停顿分类与时长（时长仅在有停顿的位置有效，损失计算时通过 mask 处理）
        pause_labels = torch.zeros(seq_len, dtype=torch.long) # 0/1
        pause_duration = torch.zeros(seq_len, dtype=torch.float)
        pause_mask = torch.zeros(seq_len, dtype=torch.bool) # 标记真实停顿位置

        stress_labels = torch.zeros(seq_len, dtype=torch.long) # 0/1
        breath_labels = torch.zeros(seq_len, dtype=torch.long) # 0/1

        # character position to token index mapping
        def char_pos_to_token_idx(char_pos):
            for i, (start, end) in enumerate(offsets):
                if start <= char_pos < end:
                    return i
            return None # 对应特殊 token（如 [CLS]、[SEP]）

        events = item.get('event_tags', [])
        for event in events:
            # accept both 'event_type' and legacy 'type' keys
            event_type = event.get('event_type') or event.get('type')
            if event_type is None:
                # malformed entry, skip
                continue
            if event_type in EVENT_MAP:
                event_idx = EVENT_MAP[event_type]
                start_char = event.get('start', 0)
                end_char = event.get('end', start_char)
                for char_pos in range(start_char, end_char):
                    token_idx = char_pos_to_token_idx(char_pos)
                    if token_idx is not None:
                        event_labels[token_idx, event_idx] = 1.0

        # 填充停顿标签
        pause_positions = item.get('pause_positions', [])
        pause_durations = item.get('pause_durations', [])
        for pos, dur in zip(pause_positions, pause_durations):
            token_idx = char_pos_to_token_idx(pos)
            if token_idx is not None:
                pause_labels[token_idx] = 1
                pause_duration[token_idx] = dur
                pause_mask[token_idx] = True

        # 填充重音标签
        stress_positions = item.get('stress_positions', [])
        for pos in stress_positions:
            token_idx = char_pos_to_token_idx(pos)
            if token_idx is not None:
                stress_labels[token_idx] = 1

        # 填充呼吸标签
        breath_positions = item.get('breath_positions', [])
        for pos in breath_positions:
            token_idx = char_pos_to_token_idx(pos)
            if token_idx is not None:
                breath_labels[token_idx] = 1

        # 整句情感标签
        emotion = item.get('emotion_type', 'neutral')
        emotion_label = torch.tensor(EMOTION_MAP.get(emotion, 0), dtype=torch.long)

        #audio processing and mel spectrogram computation
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pause_labels': pause_labels,
            'pause_duration': pause_duration,
            'pause_mask': pause_mask,
            'stress_labels': stress_labels,
            'breath_labels': breath_labels,
            'emotion_label': emotion_label,
            'event_labels': event_labels,
            'text_len': len(text), 
        }
        if self.load_audio_flag:
            audio_path = os.path.basename(item['audio_filepath'])
            actual_audio_path = os.path.join(config.RAW_DATA_DIR, audio_path)
            if audio_path and os.path.exists(actual_audio_path):
                try:
                    waveform, sr = load_audio(actual_audio_path)
                    result['audio_sr'] = sr
                    if self.compute_mel_flag:
                        # 计算梅尔谱，返回线性谱和归一化对数谱
                        mel_linear, mel_norm = waveform_to_mel(waveform, apply_preemphasis=True)
                        mel_tensor = torch.from_numpy(mel_norm.T).float()
                        result['mel'] = mel_tensor
                except Exception as e:
                     print(f"---音频处理失败: {audio_path}, 错误: {e}")
                     # 若音频加载失败，仍返回文本数据，梅尔谱置为 None
                     result['mel'] = None
            else:
                print(f"---音频文件不存在: {audio_path}")
                print(f"   预期路径: {os.path.abspath(actual_audio_path)}")
                result['mel'] = None
        return result
    
# custom 自定义批处理函数，处理变长数据
def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    pause_labels = torch.stack([b['pause_labels'] for b in batch])
    pause_duration = torch.stack([b['pause_duration'] for b in batch])
    pause_mask = torch.stack([b['pause_mask'] for b in batch])
    stress_labels = torch.stack([b['stress_labels'] for b in batch])
    breath_labels = torch.stack([b['breath_labels'] for b in batch])
    emotion_label = torch.stack([b['emotion_label'] for b in batch])
    event_labels = torch.stack([b['event_labels'] for b in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pause_labels': pause_labels,
        'pause_duration': pause_duration,
        'pause_mask': pause_mask,
        'stress_labels': stress_labels,
        'breath_labels': breath_labels,
        'emotion_label': emotion_label,
        'event_labels': event_labels,
    }

    # 处理梅尔谱，进行 padding
    if 'mel' in batch[0]:
        mels = [b['mel'] for b in batch if b['mel'] is not None]
        if mels:
            mel_len = torch.LongTensor([mel.shape[0] for mel in mels])
            mel_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0)
            result['mel'] = mel_padded
            result['mel_len'] = mel_len
        else:
            result['mel'] = None
            result['mel_len'] = None
    return result
    

# create data loader
def create_dataloader(data_path, batch_size=config.BATCH_SIZE, shuffle=True, max_text_len=config.MAX_LENGTH, load_audio_flag=True, compute_mel_flag=True,num_workers=0):
    dataset = TTSDataset(
        data_path=data_path,
        max_text_len=max_text_len,
        load_audio_flag=load_audio_flag,
        compute_mel_flag=compute_mel_flag
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False # Mac 上无需 pin_memory
    )
    return dataloader


# test 
if __name__ == "__main__":
    test_json = os.path.join(config.PROCESSED_DATA_DIR, 'sample_data_with_events.json')
    if not os.path.exists(test_json):
        print(f"---测试数据文件不存在: {test_json}")
    else:
        dataloader = create_dataloader(test_json, compute_mel_flag=True)
        for batch in dataloader:
            print("\n\nBatch keys:", batch.keys())
            print("Input IDs shape:", batch['input_ids'].shape)
            print("Attention Mask shape:", batch['attention_mask'].shape)
            print("Pause Labels shape:", batch['pause_labels'].shape)
            print("Pause Duration shape:", batch['pause_duration'].shape)
            print("Pause Mask shape:", batch['pause_mask'].shape)
            print("Stress Labels shape:", batch['stress_labels'].shape)
            print("Breath Labels shape:", batch['breath_labels'].shape)
            print("Emotion Label shape:", batch['emotion_label'].shape)
            print("Event Labels shape:", batch['event_labels'].shape) # should be (batch_size, seq_len, num_events)
            if 'mel' in batch and batch['mel'] is not None:
                print("Mel Spectrogram shape:", batch['mel'].shape)
                print("Mel Lengths:", batch['mel_len'])
            break