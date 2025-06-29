# 1. 导入必要的包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 2. 导入模型和数据加载模块
from models.your_model_main import HyperBridge
from models.modules.hyper_generator import HybridHyperedgeGenerator
from data.load_dataset import load_medmnist
from utils.metrics import accuracy, auc_score

# 3. 训练配置参数
epochs = 100
lr = 0.001
lambda_struct = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置
config = {
    'img_channels': 3,
    'hidden': 128,
    'vocab_size': 10000,
    'embed_dim': 128,
    'text_hidden': 64,
    'sig_in': 64,
    'n_class': 9,  # 根据你的数据集调整
    'top_k': 10,
    'thresh': 0.5,
    'K': 5,
    'tau': 0.5
}

# 4. 加载数据
train_loader, val_loader, test_loader = load_medmnist()

# 5. 初始化模型
model = HyperBridge(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 6. 训练过程
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # 根据数据格式调整，这里假设只有图像和标签
        if len(data) == 2:
            images, labels = data[0].to(device), data[1].to(device)
            # 创建模拟的文本和信号数据
            batch_size = images.shape[0]
            text_data = torch.randint(0, config['vocab_size'], (batch_size, 20)).to(device)
            signal_data = torch.randn(batch_size, config['sig_in']).to(device)
        else:
            # 如果有多模态数据
            images, text_data, signal_data, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        
        # 前向传播 - HyperBridge返回(logits, reg_loss)
        preds, reg_loss = model(images, text_data, signal_data)

        # 计算损失
        task_loss = criterion(preds, labels)
        total_loss_batch = task_loss + lambda_struct * reg_loss

        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")