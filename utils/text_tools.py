import re
import os
import pypinyin
from pypinyin import Style
from typing import List, Dict, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Chinese character to digits mapping
DIGIT_MAP = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
}

# common onomatopeia 
SOUND_EVENTS = {
    'laugh': ['哈','哈哈', '呵呵', '嘿嘿', '嘻嘻', '咯咯','嘻嘻','哈哈哈','哈哈哈哈','哈哈哈哈哈','哈哈哈哈哈哈','哈哈哈哈哈哈哈'],
    'cry': ['呜','呜呜', '哇','哇哇', '哭泣', '啜泣', '抽泣','呜呜呜','呜呜呜呜','呜呜呜呜呜','呜呜呜呜呜呜','呜呜呜呜呜呜呜'],
    'sigh': ['唉', '哎', '嗐', '呼','唉声叹气','叹气','叹息','叹息声'],
    'cough': ['咳嗽', '咳', '咳咳', '咳嗽声'],
    'sneeze': ['打喷嚏', '喷嚏', '喷嚏声'],
    'yawn': ['哈欠', '打哈欠', '哈欠声'],
    'hesitation':['嗯', '呃', '哦', '唔','额','嗯嗯','啊啊','呃呃','哦哦','唔唔','那个','就是'],
    'question':['咦','诶','吗','啊','呢', '吧', '怎么', '为什么', '什么','吗？','呢？','吧？','怎么？','为什么？','什么？'],
    'surprise': ['惊喜'],
    # 'door_open': ['开门', '门开了', '门开'],
    # 'door_close': ['关门', '门关了', '门关'],
    # 'footsteps': ['脚步声', '走路声', '脚步'],
    # 'knock': ['敲门', '敲门声', '咚咚'],
    # 'phone_ring': ['电话铃声', '电话响了', '电话响'],
    # 'typing': ['打字声', '键盘声', '敲键盘'],
    'applause': ['鼓掌', '掌声', '拍手']
}
# Convert Chinese characters to Pinyin with tone numbers.
def chinese_to_pinyin(text: str, tone_style: int = 3) -> List[str]:
    pinyins = pypinyin.pinyin(text, style=Style.TONE3, errors='default',neutral_tone_with_five=True)
    return [item[0] for item in pinyins]


def pinyin_list_to_string(pinyin_list: List[str]) -> str:
    return ' '.join(pinyin_list)



def normalize_text(text: str) -> str:
    # normalize digits in text to Chinese characters
    def replace_digits(match):
        digit_str = match.group()
        return ''.join(DIGIT_MAP.get(d,d) for d in digit_str)
    
    text = re.sub(r'\d+', replace_digits, text) # replace digits with Chinese characters
    text = text.lower()  # convert English letters to lowercase
    #unify punctuation（英文标点转中文）
    punctuation_map = {
        '...': '……',
        '.': '。',
        ',': '，',
        '!': '！',
        '?': '？',
        ';': '；',
        ':': '：',
        '(': '（',
        ')': '）',
        '"': '“',
        "'": '‘',
        '-': '—',
    }
    for en in sorted(punctuation_map.keys(), key=len, reverse=True):
        cn = punctuation_map[en]
        text = text.replace(en, cn)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


#input text, output pinyin str
def g2p_processor(text:str,apply_normalization:bool=True) -> str:
    if apply_normalization:
        text = normalize_text(text)
    pinyin_list = chinese_to_pinyin(text)
    # print("---Pinyin list:", pinyin_list)
    return pinyin_list_to_string(pinyin_list)


#detect sound events in text, return a list of detected events
def detect_sound_events(text:str) -> List[Dict[str, Any]]:
    detected_events = []
    for event_type, keywords in SOUND_EVENTS.items():
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text):
                start,end = match.span()
                detected_events.append({
                    'type': event_type,
                    'start': start,
                    'end': end
                })
    detected_events.sort(key=lambda x: x['start'])  # sort by start index
    return detected_events





# test script
if __name__ == "__main__":
    test_text = "我有2个苹果，哈哈！你好吗？"
    print("---Original text:", test_text)
    normalized = normalize_text(test_text)
    print("---Normalized text:", normalized)
    pinyin = g2p_processor(test_text)
    print("---Pinyin:", pinyin)
    events = detect_sound_events(test_text)
    print("--- Detected sound events:", events)