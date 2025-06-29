# 训练脚本修复说明

## 🔧 修复的导入问题

### 1. 原有问题
```python
# 错误的导入
from models.hypergraph_model import HybridHyperedgeGenerator  # 文件不存在
```

### 2. 修复方案
```python
# 正确的导入
from models.your_model_main import HyperBridge
from models.modules.hyper_generator import HybridHyperedgeGenerator
```

## 📋 主要修改内容

### 1. 模型导入和初始化
- **之前**: `HyperGraphNet()` (已删除的模型)
- **现在**: `HyperBridge(config)` (新的多模态超图模型)

### 2. 模型配置
添加了完整的配置字典：
```python
config = {
    'img_channels': 3,     # 图像通道数
    'hidden': 128,         # 隐藏层维度
    'vocab_size': 10000,   # 词汇表大小
    'embed_dim': 128,      # 嵌入维度
    'text_hidden': 64,     # 文本隐藏层
    'sig_in': 64,          # 信号输入维度
    'n_class': 9,          # 分类数
    'top_k': 10,           # 超图邻居数
    'thresh': 0.5,         # 超边阈值
    'K': 5,                # 切比雪夫多项式阶数
    'tau': 0.5             # 小波参数
}
```

### 3. 模型接口适配
- **之前**: `preds, F, L = model(inputs)` 返回预测、特征和拉普拉斯矩阵
- **现在**: `preds, reg_loss = model(images, text_data, signal_data)` 返回预测和正则化损失

### 4. 多模态数据处理
新模型需要三种输入：
- `images`: 图像数据 `[B, C, H, W]`
- `text_data`: 文本序列 `[B, seq_len]`
- `signal_data`: 信号特征 `[B, sig_dim]`

### 5. 损失计算
- **之前**: `loss = task_loss + lambda_struct * spectral_cut_loss(F, L)`
- **现在**: `loss = task_loss + lambda_struct * reg_loss` (正则化损失由模型内部计算)

## 🚀 使用方法

### 运行训练
```bash
cd trainers
python train.py
```

### 注意事项
1. **数据格式**: 当前假设数据只有图像和标签，会自动生成模拟的文本和信号数据
2. **配置调整**: 根据实际数据集调整 `n_class` 等参数
3. **内存使用**: 如果遇到内存不足，可以减少 `batch_size` 或 `hidden` 维度

### 自定义数据加载
如果你有真实的多模态数据，需要修改数据加载部分：
```python
# 替换模拟数据生成部分
images, text_data, signal_data, labels = your_multimodal_dataloader()
```

## 📊 预期输出
```
Epoch 0, Batch 0, Loss: 2.3456
Epoch 0, Batch 100, Loss: 2.1234
Epoch 0, Average Loss: 2.0123
...
```

## 🛠️ 进一步优化建议

1. **添加验证循环**: 定期在验证集上评估模型
2. **模型保存**: 保存最佳模型检查点
3. **学习率调度**: 添加学习率衰减
4. **早停机制**: 防止过拟合
5. **可视化**: 添加训练过程可视化

## ❓ 如果遇到问题

1. **导入错误**: 确保所有模块文件存在且路径正确
2. **维度错误**: 检查配置中的维度设置
3. **内存不足**: 减少批次大小或模型规模
4. **数据格式**: 确保数据格式与模型期望一致
