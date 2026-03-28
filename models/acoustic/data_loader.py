import math
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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


def normalize_mel(mel):
    return (mel - config.MEL_MEAN) / config.MEL_STD


def denormalize_mel(mel):
    return mel * config.MEL_STD + config.MEL_MEAN

class AcousticDataset(Dataset):
    """声学模型数据集"""
    def __init__(self, json_path, mel_dir, bert_feat_dir, prosody_feat_dir,
                 hop_size=config.HOP_SIZE, sr=config.SAMPLE_RATE,
                 max_samples=None, audio_ids=None, sample_strategy="head"):
        """
        Args:
            json_path: 数据 JSON 文件路径
            mel_dir: 预计算梅尔谱 .pt 目录
            bert_feat_dir: 预计算 BERT 特征 .pt 目录
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if audio_ids:
            audio_id_set = set(audio_ids)
            self.data = [
                item for item in self.data
                if os.path.splitext(os.path.basename(item['audio_filepath']))[0] in audio_id_set
            ]
        elif max_samples is not None and sample_strategy != "head":
            indexed = list(enumerate(self.data))
            if sample_strategy == "shortest":
                indexed.sort(
                    key=lambda x: self._estimate_mel_length(x[1], hop_size, sr)
                )
                self.data = [item for _, item in indexed[:max_samples]]
                max_samples = None
            elif sample_strategy == "random":
                rng = random.Random(getattr(config, "SEED", 42))
                rng.shuffle(indexed)
                self.data = [item for _, item in indexed[:max_samples]]
                max_samples = None
        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.mel_dir = mel_dir
        self.bert_feat_dir = bert_feat_dir
        self.prosody_feat_dir = prosody_feat_dir
        self.frame_rate = hop_size / sr  # 每帧秒数
        self._mel_lengths = {}

    def _estimate_mel_length(self, item, hop_size, sr):
        duration_sec = float(item.get("duration", 0.0) or 0.0)
        if duration_sec > 0:
            return max(1, int(round(duration_sec * sr / hop_size)))
        pause_frames = sum(
            int(round(float(d) / (hop_size / sr)))
            for d in item.get("pause_durations", [])
        )
        phoneme_count = len(item.get("phonemes", "").split())
        return max(1, pause_frames + phoneme_count * 10)

    def __len__(self):
        return len(self.data)

    def get_mel_length(self, idx):
        if idx not in self._mel_lengths:
            item = self.data[idx]
            audio_id = os.path.splitext(os.path.basename(item['audio_filepath']))[0]
            mel = torch.load(os.path.join(self.mel_dir, f"{audio_id}.pt"), map_location="cpu")
            self._mel_lengths[idx] = int(mel.size(0))
        return self._mel_lengths[idx]

    def __getitem__(self, idx):
        item = self.data[idx]
        # 获取文件名（不含扩展名）
        audio_id = os.path.splitext(os.path.basename(item['audio_filepath']))[0]

        # 加载预计算特征
        mel = torch.load(os.path.join(self.mel_dir, f"{audio_id}.pt")).float()
        bert_feat = torch.load(os.path.join(self.bert_feat_dir, f"{audio_id}.pt"))
        prosody_feat = torch.load(os.path.join(self.prosody_feat_dir, f"{audio_id}.pt")).float()
        mel = normalize_mel(mel)

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
            'prosody_feat': prosody_feat,  # (T_token, prosody_dim)
            'emotion_id': emotion_id,   # 标量
            'mel': mel,                 # (T_mel, 80)
            'durations': durations      # (T_token,)
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    bert_feats = [b['bert_feat'] for b in batch]
    prosody_feats = [b['prosody_feat'] for b in batch]
    mels = [b['mel'] for b in batch]
    durations = [b['durations'] for b in batch]
    emotion_ids = torch.tensor([b['emotion_id'] for b in batch])

    mel_lengths = torch.tensor([m.size(0) for m in mels])

    # 自动 Padding 到 Batch 内最大长度
    bert_padded = pad_sequence(bert_feats, batch_first=True)
    prosody_padded = pad_sequence(prosody_feats, batch_first=True)
    mel_padded = pad_sequence(mels, batch_first=True)
    duration_padded = pad_sequence(durations, batch_first=True)

    batch_size, max_mel_len, _ = mel_padded.shape
    mel_mask = torch.arange(max_mel_len).expand(batch_size, max_mel_len) < mel_lengths.unsqueeze(1)
    mel_mask = mel_mask.float() # 转为 0/1 浮点数


    return {
        'bert_feat': bert_padded,
        'prosody_feat': prosody_padded,
        'emotion_id': emotion_ids,
        'mel': mel_padded,
        'durations': duration_padded,
        'mel_mask': mel_mask
    }


class LengthBucketBatchSampler(Sampler):
    """
    Group samples with similar mel lengths into the same batch.
    This greatly reduces padding waste for long-form audio.
    """
    def __init__(self, dataset, batch_size, shuffle=True, bucket_size_multiplier=25, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)

        indexed_lengths = [(idx, dataset.get_mel_length(idx)) for idx in range(len(dataset))]
        indexed_lengths.sort(key=lambda x: x[1])
        self.sorted_indices = [idx for idx, _ in indexed_lengths]

    def __iter__(self):
        indices = list(self.sorted_indices)
        chunks = [
            indices[i:i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        if self.shuffle:
            random.shuffle(chunks)

        batches = []
        for chunk in chunks:
            if self.shuffle:
                random.shuffle(chunk)
            chunk.sort(key=lambda idx: self.dataset.get_mel_length(idx), reverse=True)

            for i in range(0, len(chunk), self.batch_size):
                batch = chunk[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

def create_dataloader(json_path, mel_dir, bert_feat_dir, prosody_feat_dir,
                     batch_size=16, shuffle=True, num_workers=0, bucket_by_length=True,
                     max_samples=None, audio_ids=None, sample_strategy="head"):
    dataset = AcousticDataset(
        json_path,
        mel_dir,
        bert_feat_dir,
        prosody_feat_dir,
        max_samples=max_samples,
        audio_ids=audio_ids,
        sample_strategy=sample_strategy,
    )
    loader_kwargs = {
        'dataset': dataset,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    if bucket_by_length:
        loader_kwargs['batch_sampler'] = LengthBucketBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        loader_kwargs['batch_size'] = batch_size
        loader_kwargs['shuffle'] = shuffle

    return DataLoader(**loader_kwargs)



if __name__ == "__main__":
    import tempfile
    import shutil

    # 尝试使用真实数据测试
    sample_json = os.path.join(config.PROCESSED_DATA_DIR,'sample_Data.json')
    mel_dir = config.ACOUSTIC_MEL_DIR
    bert_dir = config.ACOUSTIC_BERT_FEAT_DIR
    prosody_dir = config.ACOUSTIC_PROSODY_FEAT_DIR

    if os.path.exists(sample_json) and os.path.exists(mel_dir) and os.path.exists(bert_dir) and os.path.exists(prosody_dir):
        print("🔍 数据测试...")
        try:
            # 加载真实数据集（只取前2条，加快速度）
            dataset = AcousticDataset(sample_json, mel_dir, bert_dir, prosody_dir)
            sample = dataset[0]
            print("---样本加载成功:")
            print("   bert_feat shape:", sample['bert_feat'].shape)
            print("   prosody_feat shape:", sample['prosody_feat'].shape)
            print("   mel shape:", sample['mel'].shape)
            print("   durations:", sample['durations'])
            print("   emotion_id:", sample['emotion_id'])

            # 测试 DataLoader
            loader = create_dataloader(sample_json, mel_dir, bert_dir, prosody_dir, batch_size=2)
            batch = next(iter(loader))
            print("---batch 加载成功:")
            print("   bert_feat batch shape:", batch['bert_feat'].shape)
            print("   prosody_feat batch shape:", batch['prosody_feat'].shape)
            print("   mel batch shape:", batch['mel'].shape)
            print("   mel_mask shape:", batch['mel_mask'].shape)
            print("   durations shape:", batch['durations'].shape)
        except Exception as e:
            print(f" ---数据测试失败: {e}")
            print("  可能原因：特征文件未生成或格式不匹配，请先运行 scripts/prepare_acoustic_data.py")
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
            prosody_dir_v = os.path.join(test_dir, "prosody_feat")
            os.makedirs(mel_dir_v)
            os.makedirs(bert_dir_v)
            os.makedirs(prosody_dir_v)

            torch.save(torch.randn(50, config.MEL_BINS), os.path.join(mel_dir_v, "dummy.pt"))
            torch.save(torch.randn(3, 256), os.path.join(bert_dir_v, "dummy.pt"))
            torch.save(torch.randn(3, config.ACOUSTIC_PROSODY_DIM), os.path.join(prosody_dir_v, "dummy.pt"))

            # 3. 测试数据集
            dataset = AcousticDataset(json_path, mel_dir_v, bert_dir_v, prosody_dir_v)
            sample = dataset[0]
            print("虚拟样本加载成功:")
            print("   bert_feat shape:", sample['bert_feat'].shape)
            print("   prosody_feat shape:", sample['prosody_feat'].shape)
            print("   mel shape:", sample['mel'].shape)
            print("   durations:", sample['durations'])
            print("   emotion_id:", sample['emotion_id'])

            # 4. 测试 DataLoader
            loader = create_dataloader(json_path, mel_dir_v, bert_dir_v, prosody_dir_v, batch_size=2)
            batch = next(iter(loader))
            print("虚拟 batch 加载成功:")
            print("   bert_feat batch shape:", batch['bert_feat'].shape)
            print("   prosody_feat batch shape:", batch['prosody_feat'].shape)
            print("   mel batch shape:", batch['mel'].shape)
            print("   mel_mask shape:", batch['mel_mask'].shape)
            print("   durations shape:", batch['durations'].shape)
        finally:
            shutil.rmtree(test_dir)
            print("虚拟测试临时文件已清理")
