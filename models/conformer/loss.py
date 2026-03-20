import torch
import torch.nn as nn
import torch.nn.functional as F

class MyBuddyLoss(nn.Module):
    """
    My Buddy 专用声学损失函数：
    """
    def __init__(self, use_spectral_convergence=True, spec_weight=0.5, l1_weight=5.0):
        super().__init__()
        self.use_spectral_convergence = use_spectral_convergence
        self.spec_weight = spec_weight
        self.l1_weight = l1_weight 

    def forward(self, pred, target, mask=None):
        """
        pred: (batch, time, mel_bins) - 模型预测生成的梅尔谱
        target: (batch, time, mel_bins) - 录音提取的真实梅尔谱
        mask: (batch, time) - 布尔掩码或 0/1 掩码,1 表示真实音频帧
        """
        # 自动长度对齐，防止因为 duration 四舍五入导致的 1-2 帧长度不一致
        t_min = min(pred.size(1), target.size(1))
        pred = pred[:, :t_min, :]
        target = target[:, :t_min, :]
        if mask is not None:
            mask = mask[:, :t_min]

        if mask is not None:
            # 确保维度对齐 (batch, time, 1)
            m = mask.unsqueeze(-1).float()
            pred = pred * m
            target = target * m
            
            num_valid = m.sum() * target.size(-1) + 1e-8
            
            mse_loss = F.mse_loss(pred, target, reduction='sum') / num_valid
            l1_loss = F.l1_loss(pred, target, reduction='sum') / num_valid
        else:
            mse_loss = F.mse_loss(pred, target)
            l1_loss = F.l1_loss(pred, target)

        # 谱收敛损失 Spectral Convergence Loss
        # 逻辑：||target - pred||_F / ||target||_F
        spec_loss = torch.tensor(0.0, device=pred.device)
        if self.use_spectral_convergence:
            # 使用 Frobenius 范数计算形状相似度
            # 在梅尔谱上，直接计算差值的范数即可
            numerator = torch.norm(target - pred, p='fro', dim=(1, 2))
            denominator = torch.norm(target, p='fro', dim=(1, 2)) + 1e-8
            spec_loss = (numerator / denominator).mean()

        # total loss
        total_loss = mse_loss + (self.l1_weight * l1_loss) + (self.spec_weight * spec_loss)
        
        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
            "spec": spec_loss
        }

class DurationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_dur, target_dur, mask=None):
        """
        pred_dur/target_dur: (batch, token_len)
        mask: (batch, token_len) 有效音素掩码
        """
        if mask is not None:
            # 仅对非 Padding 的音素计算时长误差
            loss = F.mse_loss(pred_dur * mask, target_dur * mask, reduction='sum') / (mask.sum() + 1e-8)
        else:
            loss = F.mse_loss(pred_dur, target_dur)
        return loss


# test
if __name__ == "__main__":
    batch_size = 2
    time = 100
    mel_bins = 128

    pred = torch.randn(batch_size, time, mel_bins)
    target = torch.randn(batch_size, time, mel_bins)
    mask = torch.ones(batch_size, time).bool()
    mask[:, 80:] = False  # 模拟部分填充

    criterion = MyBuddyLoss(use_spectral_convergence=True)
    loss_dict = criterion(pred, target, mask)

    print("Loss dict:", {k: v.item() for k, v in loss_dict.items()})
    print("测试通过！")