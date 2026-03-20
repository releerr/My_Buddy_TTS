import json
import matplotlib.pyplot as plt
import os

def plot_training_history(json_path):
    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['total'] for h in history]
    val_loss = [h['val']['total'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Total Loss')
    plt.plot(epochs, val_loss, label='Val Total Loss')
    
    if 'lr' in history[0]:
 
        lrs = [h['lr'] * 10000 for h in history] # 放大一点方便看趋势
        plt.plot(epochs, lrs, '--', label='Learning Rate (x10k)')

    plt.title('BERT Brain Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(json_path), 'loss_curve.png'))
    print(" Loss 曲线已生成至 loss_curve.png")


if __name__ == "__main__":
    # TODO
    plot_training_history('/Users/rxia/nian/提交大赛AI模型/V1/V1/models/checkpoints/bert_brain_checkpoint_v3_5000/train_history.json')