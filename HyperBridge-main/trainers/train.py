# 1. 导入必要的包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 2. 导入模型和数据加载模块
from models.hypergraph_model import HyperGraphNet
from data.load_dataset import load_medmnist
from utils.metrics import accuracy, auc_score
from utils.graph_utils import construct_laplacian
from models.modules.pruning_regularizer import spectral_cut_loss

# 3. 训练配置参数
epochs = 100
lr = 0.001
lambda_struct = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. 加载数据
train_loader, val_loader, test_loader = load_medmnist()

# 5. 初始化模型
model = HyperGraphNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 6. 训练过程
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        preds, F, L = model(inputs)  # F 是节点嵌入，L 是拉普拉斯

        task_loss = criterion(preds, labels)
        structure_loss = spectral_cut_loss(F, L)
        loss = task_loss + lambda_struct * structure_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")