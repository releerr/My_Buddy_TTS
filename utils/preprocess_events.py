import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.text_tools import detect_sound_events

def preprocess_events(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"---原始样本数: {len(data)}")
    updated_count = 0

    for item in data:
        text = item.get('text', '')
        sound_events = detect_sound_events(text)
        item['event_tags'] = sound_events
        if sound_events:
            updated_count += 1

    with open(output_json_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"---处理完成！新增了 {updated_count} 个样本的事件标签。")
    print(f"---结果已保存至: {output_json_path}")


if __name__ == "__main__":
    # TODO
    input_json = "/Users/rxia/nian/提交大赛AI模型/V1/V1/data/1000full_train_data.json"
    # TODO
    output_json = os.path.join(config.PROCESSED_DATA_DIR, 'processed_1000full_train_data.json')
    preprocess_events(input_json, output_json)