# load chinese-roberta-wwm-ext model as the text encoder, and add a linear layer to project the 768-dim output to 256-dim, which will be used as the input for the prosody predictor.
import torch
import torch.nn as nn
from transformers import BertModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class TextEncoder(nn.Module):
    def __init__(self, output_dim=config.OUTPUT_DIM):
        super().__init__()
        print(f"--- 正在加载文本编码器权重: {config.CHINESE_ROBERTA_MODEL_DIR}")
        self.bert = BertModel.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR)
        self.hidden_size = self.bert.config.hidden_size # 通常是768
        self.output_dim = output_dim
        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        dropped = self.dropout(last_hidden_state)
        projected = self.projection(dropped)  # 降维映射 (batch_size, seq_len, output_dim)
        return projected

    def get_output_dim(self):
        return self.output_dim
    

#test
if __name__ == "__main__":
    from transformers import BertTokenizer
    device = config.DEVICE
    print(f"--- 当前测试设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(config.CHINESE_ROBERTA_MODEL_DIR)
    encoder = TextEncoder(output_dim=256).to(device)

    text = "今天天气真好！哈哈哈，我好开心啊！"
    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=config.MAX_LENGTH
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)   

    #start forward
    with torch.no_grad():
        embeddings = encoder(input_ids, attention_mask)

    print(f"\n[--- test result ---]")
    print(f"--- Input Text: {text}")
    print(f"--- Input ID Shape: {input_ids.shape}")  # suppose to be (1, 128)
    print(f"--- Encoder Output Shape: {embeddings.shape}")  # suppose to be (1, 128, 256)
    print(f"--- Encoder Output Sample 0, first 5 tokens and first 5 dimensions: {embeddings[0, :5, :5]}")  # print the first sample [CLS],first 5 tokens and first 5 dimensions of the output vector

