import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class ProsodyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # default weights 
        self.weights = weights if weights is not None else {
            'pause': 2.0,
            'duration': 5.0,
            'breath': 1.0, # because breath points are less frequent, we can give it a smaller weight
            'emotion': 0.3, # emotion labels are at sentence level, and we want to focus more on token-level predictions, so we can give it a smaller weight
            'event': 1.0
        }
        pause_weights = torch.tensor([3.2, 1.0]).to(config.DEVICE)
        self.ce_weighted = nn.CrossEntropyLoss(weight=pause_weights, reduction='none')
        self.ce_simple = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.emtion_ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, mask=None):
        # prediction: ProsodyPredictor output dict
        # targets: data_loader output dict
        # mask: (B,T) binary mask for valid tokens, 1 for valid, 0 for padding, used to ignore padding tokens in loss calculation
        loss_dict = {}
        B,T = targets['pause_labels'].shape
        if mask is None:
            mask = torch.ones((B,T), device=config.DEVICE)

        # pause loss
        pause_logits = predictions['pause_logits'].view(-1, 2)
        pause_labels = targets['pause_labels'].view(-1)
        pause_loss = self.ce_weighted(pause_logits, pause_labels).view(B, T) 
        loss_dict['pause_loss'] = (pause_loss * mask).sum() / (mask.sum() + 1e-6)

        # duration loss
        duration_loss = self.mse(predictions['duration_preds'], targets['pause_duration'])
        dur_active_mask = mask * (targets['pause_labels'].float())
        if dur_active_mask.sum() > 1e-4:
            loss_dict['duration_loss'] = (duration_loss * dur_active_mask).sum() / (dur_active_mask.sum() + 1e-6)
        else:
            loss_dict['duration_loss'] = torch.tensor(0.0, device=config.DEVICE)

        # breath loss
        breath_loss = self.ce_simple(predictions['breath_logits'].view(-1, 2), targets['breath_labels'].view(-1)).view(B,T) 
        loss_dict['breath_loss'] = (breath_loss * mask).sum() / (mask.sum() + 1e-6)

        # event loss (only if event classes exist and shapes match)
        if config.NUM_SOUND_EVENTS > 0 and predictions['event_logits'].shape[-1] == config.NUM_SOUND_EVENTS:
            event_targets = targets['event_labels'].float()
            event_loss = self.bce(predictions['event_logits'], event_targets)
            event_loss = event_loss.mean(dim=-1) # average over event labels to get a single loss value per token
            loss_dict['event_loss'] = (event_loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            # no event classes configured or mismatch: set zero loss to avoid NaNs
            loss_dict['event_loss'] = torch.tensor(0.0, device=predictions['pause_logits'].device)

        # emotion loss
        loss_dict['emotion_loss'] = self.emtion_ce(predictions['emotion_logits'], targets['emotion_label'])


        # total loss
        # initialize as a scalar tensor on the correct device so downstream .item() works
        total_loss = torch.tensor(0.0, device=predictions['pause_logits'].device)       
        for key, weight in self.weights.items():
            loss_key = f"{key}_loss"
            if loss_key in loss_dict:
                total_loss = total_loss + weight * loss_dict[loss_key]

        loss_dict['total_loss'] = total_loss
        return loss_dict
    

# test
if __name__ == "__main__":
    device = config.DEVICE
    criterion = ProsodyLoss().to(device)
    
    B, T = 2, 128
    dummy_predictions = {
        'pause_logits': torch.randn(B, T, 2).to(device),
        'duration_preds': torch.randn(B, T).to(device),
        'breath_logits': torch.randn(B, T, 2).to(device),
        'emotion_logits': torch.randn(B, config.NUM_EMOTIONS).to(device),
        'event_logits': torch.randn(B, T, config.NUM_SOUND_EVENTS).to(device)
    }

    dummy_targets = {
        'pause_labels': torch.randint(0, 2, (B, T)).to(device),
        'pause_duration': torch.abs(torch.randn(B, T)).to(device),
        'breath_labels': torch.randint(0, 2, (B, T)).to(device),
        'emotion_label': torch.randint(0, config.NUM_EMOTIONS, (B,)).to(device),
        'event_labels': torch.randint(0, 2, (B, T, config.NUM_SOUND_EVENTS)).to(device)
    }

    mask = torch.ones((B, T), device=device) # all tokens are valid in this dummy example
    results = criterion(dummy_predictions, dummy_targets, mask)
    print("--- ProsodyLoss 测试通过 ---")
    for k, v in results.items():
        val = v.item()
        status = "正常" if not torch.isnan(v) else "警告:仍有NAN"
        print(f"{k} Loss: {val:.4f} ({status})")


