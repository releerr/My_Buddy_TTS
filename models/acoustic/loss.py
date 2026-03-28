import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class MyBuddyLoss(nn.Module):
    """
    My Buddy 声学损失函数：
    """
    def __init__(
        self,
        use_spectral_convergence=True,
        spec_weight=0.5,
        l1_weight=3.0,
        weighted_l1_weight=2.0,
        energy_weight=2.0,
        peak_weight=2.0,
        peak_focus=1.5,
        guided_attn_weight=None,
        guided_attn_sigma=None,
        delta_weight=None,
        band_weight=None,
        high_band_weight=None,
        high_band_start=None,
    ):
        super().__init__()
        self.use_spectral_convergence = use_spectral_convergence
        self.spec_weight = spec_weight
        self.l1_weight = l1_weight
        self.weighted_l1_weight = weighted_l1_weight
        self.energy_weight = energy_weight
        self.peak_weight = peak_weight
        self.peak_focus = peak_focus
        self.guided_attn_weight = (
            config.ACOUSTIC_GUIDED_ATTN_WEIGHT if guided_attn_weight is None else guided_attn_weight
        )
        self.guided_attn_sigma = (
            config.ACOUSTIC_GUIDED_ATTN_SIGMA if guided_attn_sigma is None else guided_attn_sigma
        )
        self.delta_weight = (
            config.ACOUSTIC_DELTA_WEIGHT if delta_weight is None else delta_weight
        )
        self.band_weight = (
            config.ACOUSTIC_BAND_WEIGHT if band_weight is None else band_weight
        )
        self.high_band_weight = (
            config.ACOUSTIC_HIGH_BAND_WEIGHT if high_band_weight is None else high_band_weight
        )
        self.high_band_start = (
            config.ACOUSTIC_HIGH_BAND_START if high_band_start is None else high_band_start
        )

    def masked_mean(self, value, mask):
        if mask is None:
            return value.mean()
        return torch.sum(value * mask) / (mask.sum() + 1e-8)

    def guided_attention_loss(self, attn_weights, frame_mask=None, token_mask=None):
        """
        attn_weights: (batch, T_frame, T_token)
        frame_mask: (batch, T_frame)
        token_mask: (batch, T_token)
        """
        if attn_weights is None:
            device = frame_mask.device if frame_mask is not None else (
                token_mask.device if token_mask is not None else "cpu"
            )
            return torch.tensor(0.0, device=device)

        bsz, t_frame, t_token = attn_weights.shape
        device = attn_weights.device
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)

        for b in range(bsz):
            frame_len = int(frame_mask[b, :t_frame].sum().item()) if frame_mask is not None else t_frame
            token_len = int(token_mask[b, :t_token].sum().item()) if token_mask is not None else t_token
            frame_len = max(frame_len, 1)
            token_len = max(token_len, 1)

            attn = attn_weights[b, :frame_len, :token_len]
            frame_pos = torch.arange(frame_len, device=device).float().unsqueeze(1) / frame_len
            token_pos = torch.arange(token_len, device=device).float().unsqueeze(0) / token_len
            guide = 1.0 - torch.exp(
                -((frame_pos - token_pos) ** 2) / (2 * (self.guided_attn_sigma ** 2))
            )

            total_loss = total_loss + torch.sum(attn * guide)
            total_weight = total_weight + torch.tensor(attn.numel(), device=device, dtype=attn.dtype)

        return total_loss / (total_weight + 1e-8)

    def forward(self, pred, target, mask=None, attn_weights=None, token_mask=None):
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
            m = mask[:, :t_min].unsqueeze(-1).float()
            num_valid = m.sum() * target.size(-1) + 1e-8

            mse_loss = torch.sum(((pred - target) * m) ** 2) / num_valid
            l1_loss = torch.sum(torch.abs((pred - target) * m)) / num_valid
            peak_weights = 1.0 + self.peak_focus * torch.sigmoid(target)
            weighted_l1 = torch.sum(torch.abs(pred - target) * peak_weights * m) / (
                torch.sum(peak_weights * m) + 1e-8
            )
            pred_energy = pred.mean(dim=-1)
            target_energy = target.mean(dim=-1)
            pred_peak = pred.amax(dim=-1)
            target_peak = target.amax(dim=-1)
            num_frames = m.squeeze(-1).sum() + 1e-8
            energy_loss = torch.sum(torch.abs(pred_energy - target_energy) * m.squeeze(-1)) / num_frames
            peak_loss = torch.sum(torch.abs(pred_peak - target_peak) * m.squeeze(-1)) / num_frames
            band_mask = m
        else:
            mse_loss = F.mse_loss(pred, target)
            l1_loss = F.l1_loss(pred, target)
            peak_weights = 1.0 + self.peak_focus * torch.sigmoid(target)
            weighted_l1 = (torch.abs(pred - target) * peak_weights).mean()
            energy_loss = F.l1_loss(pred.mean(dim=-1), target.mean(dim=-1))
            peak_loss = F.l1_loss(pred.amax(dim=-1), target.amax(dim=-1))
            band_mask = None

        # 谱收敛损失 Spectral Convergence Loss
        # 逻辑：||target - pred||_F / ||target||_F
        spec_loss = torch.tensor(0.0, device=pred.device)
        if self.use_spectral_convergence:
            # 使用 Frobenius 范数计算形状相似度
            # 在梅尔谱上，直接计算差值的范数即可
            if mask is not None:
                m = mask[:, :t_min].unsqueeze(-1).float()
                p_masked = pred * m
                t_masked = target * m
            else:
                p_masked, t_masked = pred, target
            numerator = torch.norm(t_masked - p_masked, p='fro', dim=(1, 2))
            denominator = torch.norm(t_masked, p='fro', dim=(1, 2)) + 1.0 
            spec_loss = (numerator / denominator).mean()

        guided_attn = self.guided_attention_loss(
            attn_weights,
            frame_mask=mask[:, :t_min] if mask is not None else None,
            token_mask=token_mask,
        )

        delta_loss = torch.tensor(0.0, device=pred.device)
        if t_min > 1:
            pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
            target_delta = target[:, 1:, :] - target[:, :-1, :]
            if mask is not None:
                delta_mask = (mask[:, 1:t_min] * mask[:, :t_min - 1]).unsqueeze(-1).float()
                delta_loss = self.masked_mean(torch.abs(pred_delta - target_delta), delta_mask)
            else:
                delta_loss = F.l1_loss(pred_delta, target_delta)

        band_loss = torch.tensor(0.0, device=pred.device)
        high_band_loss = torch.tensor(0.0, device=pred.device)
        if pred.size(-1) > 1:
            pred_band = pred[:, :, 1:] - pred[:, :, :-1]
            target_band = target[:, :, 1:] - target[:, :, :-1]
            if band_mask is not None:
                band_loss = self.masked_mean(torch.abs(pred_band - target_band), band_mask)
            else:
                band_loss = F.l1_loss(pred_band, target_band)

            high_start = min(max(int(self.high_band_start), 0), pred.size(-1) - 1)
            pred_high = pred[:, :, high_start:]
            target_high = target[:, :, high_start:]
            if pred_high.numel() > 0:
                if band_mask is not None:
                    high_band_loss = self.masked_mean(torch.abs(pred_high - target_high), band_mask)
                else:
                    high_band_loss = F.l1_loss(pred_high, target_high)

        # total loss
        total_loss = (
            mse_loss
            + (self.l1_weight * l1_loss)
            + (self.weighted_l1_weight * weighted_l1)
            + (self.energy_weight * energy_loss)
            + (self.peak_weight * peak_loss)
            + (self.spec_weight * spec_loss)
            + (self.guided_attn_weight * guided_attn)
            + (self.delta_weight * delta_loss)
            + (self.band_weight * band_loss)
            + (self.high_band_weight * high_band_loss)
        )
        
        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
            "weighted_l1": weighted_l1,
            "energy": energy_loss,
            "peak": peak_loss,
            "spec": spec_loss,
            "guided_attn": guided_attn,
            "delta": delta_loss,
            "band": band_loss,
            "high_band": high_band_loss,
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
