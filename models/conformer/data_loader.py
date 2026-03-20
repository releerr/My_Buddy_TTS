import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

# 路径对齐
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import config

# 情感映射（与 model/loss 保持一致）
EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'surprised': 4,
    'interrogative': 5
}

class ConformerDataset(Dataset):
    """声学模型数据集"""
    def __init__(self, json_path, mel_dir, bert_feat_dir, 
                 hop_size=config.HOP_SIZE, sr=config.SAMPLE_RATE):
        """
        Args:
            json_path: 数据 JSON 文件路径
            mel_dir: 预计算梅尔谱 .pt 目录
            bert_feat_dir: 预计算 BERT 特征 .pt 目录
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.mel_dir = mel_dir
        self.bert_feat_dir = bert_feat_dir
        self.frame_rate = hop_size / sr  # 每帧秒数

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 获取文件名（不含扩展名）
        audio_id = os.path.splitext(os.path.basename(item['audio_filepath']))[0]

        # 加载预计算特征
        mel = torch.load(os.path.join(self.mel_dir, f"{audio_id}.pt"))
        bert_feat = torch.load(os.path.join(self.bert_feat_dir, f"{audio_id}.pt"))

        # 时长对齐逻辑 (Duration Alignment)
        phonemes = item['phonemes'].split()
        num_phones = len(phonemes)
        total_frames = mel.size(0)

        # 转换停顿帧数
        pause_positions = item.get('pause_positions', [])
        pause_durations_sec = item.get('pause_durations', [])
        pause_frames = [int(round(d / self.frame_rate)) for d in pause_durations_sec]

        # 计算分配
        total_pause_frames = sum(pause_frames)
        non_pause_frames = max(0, total_frames - total_pause_frames)
        non_pause_positions = [i for i in range(num_phones) if i not in pause_positions]
        non_pause_count = len(non_pause_positions)

        # 初始化时长
        durations = [0] * num_phones
        
        # 填充停顿帧
        for pos, frames in zip(pause_positions, pause_frames):
            if pos < num_phones:
                durations[pos] = frames

        # 填充非停顿帧（带余数分配，确保总和绝对等于 total_frames）
        if non_pause_count > 0:
            avg_f = non_pause_frames // non_pause_count
            remainder = non_pause_frames % non_pause_count
            for i, pos in enumerate(non_pause_positions):
                # 将余数分摊给前几个音素，确保总数闭合
                durations[pos] = avg_f + (1 if i < remainder else 0)

        durations = torch.tensor(durations, dtype=torch.long)

        # 情感标签
        emotion = item.get('emotion_type', 'neutral')
        emotion_id = EMOTION_MAP.get(emotion, 0)

        return {
            'bert_feat': bert_feat,     # (T_token, 256)
            'emotion_id': emotion_id,   # 标量
            'mel': mel,                 # (T_mel, 80)
            'durations': durations      # (T_token,)
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    bert_feats = [b['bert_feat'] for b in batch]
    mels = [b['mel'] for b in batch]
    durations = [b['durations'] for b in batch]
    emotion_ids = torch.tensor([b['emotion_id'] for b in batch])

    mel_lengths = torch.tensor([m.size(0) for m in mels])

    # 自动 Padding 到 Batch 内最大长度
    bert_padded = pad_sequence(bert_feats, batch_first=True)
    mel_padded = pad_sequence(mels, batch_first=True)
    duration_padded = pad_sequence(durations, batch_first=True)

    batch_size, max_mel_len, _ = mel_padded.shape
    mel_mask = torch.arange(max_mel_len).expand(batch_size, max_mel_len) < mel_lengths.unsqueeze(1)
    mel_mask = mel_mask.float() # 转为 0/1 浮点数


    return {
        'bert_feat': bert_padded,
        'emotion_id': emotion_ids,
        'mel': mel_padded,
        'durations': duration_padded,
        'mel_mask': mel_mask
    }

def create_dataloader(json_path, mel_dir, bert_feat_dir,
                     batch_size=16, shuffle=True, num_workers=0):
    dataset = ConformerDataset(json_path, mel_dir, bert_feat_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )



if __name__ == "__main__":
    import tempfile
    import shutil

    # 尝试使用真实数据测试
    sample_json = os.path.join(config.PROCESSED_DATA_DIR,'sample_Data.json')
    mel_dir = os.path.join(config.PROCESSED_DATA_DIR, 'mel')
    bert_dir = os.path.join(config.PROCESSED_DATA_DIR, 'bert_feat')

    if os.path.exists(sample_json) and os.path.exists(mel_dir) and os.path.exists(bert_dir):
        print("🔍 数据测试...")
        try:
            # 加载真实数据集（只取前2条，加快速度）
            dataset = ConformerDataset(sample_json, mel_dir, bert_dir)
            sample = dataset[0]
            print("---样本加载成功:")
            print("   bert_feat shape:", sample['bert_feat'].shape)
            print("   mel shape:", sample['mel'].shape)
            print("   durations:", sample['durations'])
            print("   emotion_id:", sample['emotion_id'])

            # 测试 DataLoader
            loader = create_dataloader(sample_json, mel_dir, bert_dir, batch_size=2)
            batch = next(iter(loader))
            print("---batch 加载成功:")
            print("   bert_feat batch shape:", batch['bert_feat'].shape)
            print("   mel batch shape:", batch['mel'].shape)
            print("   mel_mask shape:", batch['mel_mask'].shape)
            print("   durations shape:", batch['durations'].shape)
        except Exception as e:
            print(f" ---数据测试失败: {e}")
            print("  可能原因：特征文件未生成或格式不匹配，请先运行 scripts/prepare_conformer_data.py")
            print("  将回退到虚拟数据测试...")
            use_virtual = True
        else:
            use_virtual = False
    else:
        print("---未找到真实数据文件，将使用虚拟数据测试...")
        use_virtual = True

    if use_virtual:
        # 创建临时虚拟数据测试
        test_dir = tempfile.mkdtemp()
        try:
            # 1. 创建 dummy JSON
            json_path = os.path.join(test_dir, "test.json")
            dummy_data = [
                {
                    "audio_filepath": "dummy.wav",
                    "phonemes": "a1 b2 c3",
                    "pause_positions": [1],
                    "pause_durations": [0.2],
                    "emotion_type": "neutral"
                }
            ]
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f)

            # 2. 创建 dummy 特征目录和文件
            mel_dir_v = os.path.join(test_dir, "mel")
            bert_dir_v = os.path.join(test_dir, "bert_feat")
            os.makedirs(mel_dir_v)
            os.makedirs(bert_dir_v)

            torch.save(torch.randn(50, 80), os.path.join(mel_dir_v, "dummy.pt"))
            torch.save(torch.randn(3, 256), os.path.join(bert_dir_v, "dummy.pt"))

            # 3. 测试数据集
            dataset = ConformerDataset(json_path, mel_dir_v, bert_dir_v)
            sample = dataset[0]
            print("虚拟样本加载成功:")
            print("   bert_feat shape:", sample['bert_feat'].shape)
            print("   mel shape:", sample['mel'].shape)
            print("   durations:", sample['durations'])
            print("   emotion_id:", sample['emotion_id'])

            # 4. 测试 DataLoader
            loader = create_dataloader(json_path, mel_dir_v, bert_dir_v, batch_size=2)
            batch = next(iter(loader))
            print("虚拟 batch 加载成功:")
            print("   bert_feat batch shape:", batch['bert_feat'].shape)
            print("   mel batch shape:", batch['mel'].shape)
            print("   mel_mask shape:", batch['mel_mask'].shape)
            print("   durations shape:", batch['durations'].shape)
        finally:
            shutil.rmtree(test_dir)
            print("虚拟测试临时文件已清理")