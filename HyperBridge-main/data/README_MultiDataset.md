# 多数据集超图可视化模块

本模块扩展了原有的超图处理功能，支持同时处理和对比多个数据集的超图结构。

## ✨ 新增功能

### 1. 多数据集支持
- **PathMNIST**: 病理学图像数据集
- **BloodMNIST**: 血液细胞图像数据集  
- **Synthetic**: 自动生成的合成数据集

### 2. 综合可视化分析
- 超图结构对比
- 统计信息对比
- 特征分布可视化
- 相似度矩阵对比
- 网络属性分析

## 🚀 使用方法

### 基本使用

```bash
# 运行所有三个数据集的对比分析
python data/preprocess.py

# 只运行单个数据集（PathMNIST）
python data/preprocess.py single

# 使用演示脚本
python data/demo_multi_hypergraph.py
```

### 高级使用

```bash
# 交互式模式
python data/demo_multi_hypergraph.py --interactive

# 自定义数据集组合
python data/demo_multi_hypergraph.py --datasets pathmnist bloodmnist --samples 128

# 单数据集模式
python data/demo_multi_hypergraph.py --single
```

## 📊 可视化输出

### 1. 超图结构对比 (`comparative_hypergraph_structures.png`)
- 展示每个数据集的超图布局
- 显示节点连接关系
- 对比不同数据集的拓扑结构

### 2. 统计信息对比 (`comparative_statistics.png`)
- 节点数量对比
- 超边数量对比
- 平均超边大小
- 平均节点度

### 3. 特征分布对比 (`comparative_feature_distributions.png`)
- 使用t-SNE降维展示特征分布
- 不同类别的聚类效果
- 数据集间的特征差异

### 4. 相似度矩阵对比 (`comparative_similarity_matrices.png`)
- 样本间相似度热力图
- 对比不同数据集的相似度模式

### 5. 网络属性对比 (`comparative_network_properties.png`)
- 超边大小分布
- 节点度分布
- 连通组件分析

## 🔧 核心类和函数

### `MultiDatasetHypergraphVisualizer`
专门用于多数据集对比可视化的类。

```python
visualizer = MultiDatasetHypergraphVisualizer()
visualizer.create_comparative_visualization(hypergraph_datasets, save_path)
```

### `load_medmnist_dataset(dataset_name, batch_size, num_samples)`
通用的MedMNIST数据集加载函数。

```python
# 加载PathMNIST
images, labels = load_medmnist_dataset('pathmnist', num_samples=256)

# 加载BloodMNIST  
images, labels = load_medmnist_dataset('bloodmnist', num_samples=256)
```

### `load_synthetic_dataset(num_samples, image_size, num_classes)`
生成具有不同模式的合成数据集。

```python
images, labels = load_synthetic_dataset(
    num_samples=256, 
    image_size=(3, 28, 28), 
    num_classes=5
)
```

### `visualize_multiple_datasets()`
主要的多数据集处理和可视化函数。

## 📁 输出结构

```
multi_dataset_hypergraph_results/
├── pathmnist_hypergraph.pkl              # PathMNIST超图数据
├── bloodmnist_hypergraph.pkl             # BloodMNIST超图数据
├── synthetic_hypergraph.pkl              # 合成数据超图数据
├── comparative_hypergraph_structures.png # 超图结构对比
├── comparative_statistics.png            # 统计信息对比
├── comparative_feature_distributions.png # 特征分布对比
├── comparative_similarity_matrices.png   # 相似度矩阵对比
└── comparative_network_properties.png    # 网络属性对比
```

## 🎯 自定义使用

### 添加新数据集
```python
# 1. 实现数据加载函数
def load_your_dataset(num_samples=256):
    # 加载和预处理数据
    images = ...  # torch.Tensor of shape (N, C, H, W)
    labels = ...  # torch.Tensor of shape (N,)
    return images, labels

# 2. 转换为超图
converter = HypergraphConverter()
hypergraph_data = converter.convert_to_hypergraph(images, labels)

# 3. 添加到数据集字典
hypergraph_datasets['YourDataset'] = hypergraph_data

# 4. 创建可视化
visualizer = MultiDatasetHypergraphVisualizer()
visualizer.create_comparative_visualization(hypergraph_datasets, save_dir)
```

### 调整超图参数
```python
converter = HypergraphConverter(
    similarity_threshold=0.8,    # 相似度阈值
    max_hyperedge_size=15,       # 最大超边大小
    feature_dim=128              # 特征维度
)
```

## 📋 依赖要求

```txt
torch
numpy
matplotlib
seaborn
scikit-learn
hypernetx
medmnist
networkx
```

## 🔍 故障排除

### 常见问题

1. **MedMNIST安装失败**
   ```bash
   pip install medmnist
   ```

2. **HyperNetX版本兼容性**
   ```bash
   pip install hypernetx>=2.0.0
   ```

3. **内存不足**
   - 减少 `num_samples` 参数
   - 降低 `feature_dim` 参数

4. **可视化失败**
   - 检查图形后端设置
   - 确保有足够的显示内存

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 性能考虑

- **样本数量**: 建议每个数据集使用256-512个样本
- **特征维度**: 64-128维提供良好的平衡
- **超边大小**: 限制在10-15以内避免过度复杂

## 🎨 可视化定制

### 修改图形样式
```python
visualizer = MultiDatasetHypergraphVisualizer(figsize=(24, 18))
```

### 自定义颜色方案
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn')  # 或其他样式
```

## 🤝 贡献

欢迎提交Issues和Pull Requests来改进这个模块！

## 📄 许可证

请参考项目根目录的LICENSE文件。
