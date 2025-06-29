import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE

def plot_loss(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_embeddings(embeddings, labels, save_path=None, title='t-SNE Visualization'):
    """
    使用 t-SNE 降维可视化节点嵌入
    Args:
        embeddings: Tensor of shape (N, D)
        labels: Ground-truth labels, shape (N,)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='Set2', s=60, alpha=0.8)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Class')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_attention_weights(att_weights, save_path=None):
    """
    可视化注意力权重（如 alpha_m 模态权重）
    """
    if isinstance(att_weights, torch.Tensor):
        att_weights = att_weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    sns.heatmap(att_weights, cmap='YlGnBu', annot=True)
    plt.title('Attention Weights')
    plt.xlabel('Modalities')
    plt.ylabel('Samples')
    if save_path:
        plt.savefig(save_path)
    plt.show()
