import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


#predicted prosody, 1D CNN and multi-task heads for pause, duration, breath, emotion...
class ProsodyPredictor(nn.Module):
    def __init__(self, input_dim=config.OUTPUT_DIM, cnn_channels=256, droptout=0.1):
        super().__init__()

        # 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(droptout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(droptout) # output 256*128,256 is features, 128 is seq_len
        )

        head_hidden_dim = 128
        # multi-task heads
        self.pause_classifier = nn.Sequential(
            nn.Linear(cnn_channels, head_hidden_dim), # 256 -> 128
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2) # 128 -> 2 (binary classification for pause)
        )

        self.duration_regressor = nn.Sequential(
            nn.Linear(cnn_channels, head_hidden_dim), # 256 -> 128
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1) # 128 -> 1 (regression for duration)
        )

        self.breath_classifier = nn.Sequential(
            nn.Linear(cnn_channels, head_hidden_dim), # 256 -> 128
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2) # 128 -> 2 (binary classification for breath)
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(cnn_channels, head_hidden_dim), # 256 -> 128
            nn.ReLU(),
            nn.Linear(head_hidden_dim, config.NUM_EMOTIONS) # 128 -> 6 (multi-label classification for emotion)
        )

        self.event_classifier = nn.Sequential(
            nn.Linear(cnn_channels, head_hidden_dim), # 256 -> 128
            nn.ReLU(),
            nn.Linear(head_hidden_dim, config.NUM_SOUND_EVENTS) # 128 -> num_sound_events (multi-label classification for sound events)
        )


    def forward(self, x):
        # x: (batch_size, seq_len, features=256)
        x_cnn = x.transpose(1, 2) # (batch_size, features=256, seq_len)
        cnn_out_features = self.cnn(x_cnn) # (batch_size, cnn_channels=256, seq_len)
        cnn_out_features = cnn_out_features.transpose(1, 2) # (batch_size, seq_len, cnn_channels)

        pause_logits = self.pause_classifier(cnn_out_features) # (batch_size, seq_len, 2)
        duration_preds = self.duration_regressor(cnn_out_features).squeeze(-1) # (batch_size, seq_len)
        breath_logits = self.breath_classifier(cnn_out_features) # (batch_size, seq_len, 2)
        event_logits = self.event_classifier(cnn_out_features) # (batch_size, seq_len, num_sound_events)
        cls_features = cnn_out_features[:, 0, :] # get [CLS] position (first token along seq_len)
        emotion_logits = self.emotion_classifier(cls_features) # (batch_size, num_emotions)

        return {
            'pause_logits': pause_logits,
            'duration_preds': duration_preds,
            'breath_logits': breath_logits,
            'emotion_logits': emotion_logits,
            'event_logits': event_logits
        }
    

# test
if __name__ == "__main__":
    device = config.DEVICE
    print(f"--- 正在测试 ProsodyPredictor ---")
    dummy_input = torch.randn(2, 128, config.OUTPUT_DIM).to(device) # batch_size=2, seq_len=128, features=256
    predictor = ProsodyPredictor().to(device)

    with torch.no_grad():
        outputs = predictor(dummy_input)

    print(f"---输出结果确认:")
    print(f"pause_logits shape: {outputs['pause_logits'].shape} (should be [2, 128, 2])")
    print(f"duration_preds shape: {outputs['duration_preds'].shape} (should be [2, 128])")
    print(f"breath_logits shape: {outputs['breath_logits'].shape} (should be [2, 128, 2])")
    print(f"emotion_logits shape: {outputs['emotion_logits'].shape} (should be [2, 6])")
    print(f"event_logits shape: {outputs['event_logits'].shape} (should be [2, 128, num_sound_events])")    


   