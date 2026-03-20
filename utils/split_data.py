import json
import os
import random
import sys
from collections import Counter

# 路径对齐
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import config

def split_full_dataset(input_json_name='processed_5643_full_data.json', train_ratio=0.8, val_ratio=0.1):
    """
    1. 读取全量数据
    2. 彻底打乱顺序 (Shuffle)
    3. 按比例切分为 Train / Val / Test
    4. 统计情感分布，确保科学性
    """
    input_path = os.path.join(config.PROCESSED_DATA_DIR, input_json_name)
    
    if not os.path.exists(input_path):
        print(f" 错误：在 {config.PROCESSED_DATA_DIR} 下找不到文件 {input_json_name}")
        return


    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f" 发现原始数据: {total_count} 条")

    # 打乱顺序
    # 使用 config.SEED (42) 
    random.seed(config.SEED)
    random.shuffle(data)
    print(" 数据已完成随机打乱 (Shuffle)。")


    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 情感分布统计 (辅助检查)
    def get_emotion_dist(dataset):
        emotions = [item.get('emotion', 'unknown') for item in dataset]
        return Counter(emotions)

    print("\n 数据集分布统计:")
    print(f"   - 训练集: {len(train_data)} 条 | 情感分布: {get_emotion_dist(train_data)}")
    print(f"   - 验证集: {len(val_data)} 条 | 情感分布: {get_emotion_dist(val_data)}")
    print(f"   - 测试集: {len(test_data)} 条 | 情感分布: {get_emotion_dist(test_data)}")

    # 5. 保存
    save_map = {
        'processed_train_data.json': train_data,
        'processed_val_data.json': val_data,
        'processed_test_data.json': test_data
    }

    for filename, subset in save_map.items():
        save_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=4)
        print(f" 已保存: {filename}")

    print("\n 数据准备就绪！")

if __name__ == "__main__":
    split_full_dataset()