# 数据加载和超图转换模块修复说明

## 🔧 修复的问题

### 1. 缺失的函数导入错误
**问题**: `from preprocess import process_multimodal_datasets` - 该函数不存在
**解决方案**: 在`preprocess.py`中创建了`process_multimodal_datasets`函数

### 2. 数据加载器兼容性
**问题**: `load_medmnist`返回的数据格式与超图处理函数不匹配
**解决方案**: 修改了数据加载器格式和错误处理

## ✨ 新增功能

### 1. 通用MedMNIST数据集加载 (`load_medmnist_dataset`)
```python
images, labels = load_medmnist_dataset('pathmnist', num_samples=256)
```

### 2. 多模态数据集超图处理 (`process_multimodal_datasets`)
```python
hypergraph_results = process_multimodal_datasets(data_loaders, num_samples=256)
```

### 3. 增强的错误处理
- 数据集加载失败时继续处理其他数据集
- 详细的错误信息和调试输出
- 自动创建保存目录

## 📁 文件结构

```
data/
├── load_dataset.py              # 数据加载模块 (已修复)
├── preprocess.py               # 预处理和超图转换 (已扩展)
├── test_data_pipeline.py       # 测试脚本 (新增)
└── demo_multi_hypergraph.py    # 演示脚本
```

## 🚀 使用方法

### 方法1: 使用修复后的 load_dataset.py
```bash
cd data
python load_dataset.py
```

### 方法2: 单独使用超图转换功能
```python
from preprocess import process_multimodal_datasets
from load_dataset import load_medmnist

# 加载数据
loaders = load_medmnist()

# 转换为超图
results = process_multimodal_datasets(loaders, num_samples=256)
```

### 方法3: 运行测试验证功能
```bash
cd data
python test_data_pipeline.py
```

## 📊 支持的数据集

当前支持的MedMNIST数据集：
- **PathMNIST**: 病理学图像 (9类)
- **BloodMNIST**: 血液细胞图像 (8类)
- **OrganSMNIST**: 腹部器官图像 (11类)

## ⚙️ 配置参数

### 数据加载配置
```python
loaders = load_medmnist(
    batch_size=64,      # 批次大小
    download=True       # 是否下载数据集
)
```

### 超图转换配置
```python
results = process_multimodal_datasets(
    data_loaders,
    num_samples=256,           # 每个数据集的样本数
    save_dir="hypergraph_results"  # 保存目录
)
```

### HypergraphConverter配置
```python
converter = HypergraphConverter(
    similarity_threshold=0.7,   # 相似度阈值
    max_hyperedge_size=12,     # 最大超边大小
    feature_dim=64             # 特征维度
)
```

## 📈 输出结果

### 1. 超图数据文件
```
hypergraph_results/
├── pathmnist/
│   ├── pathmnist_hypergraph.pkl
│   ├── Pathmnist_similarity_matrix.png
│   ├── Pathmnist_feature_distribution.png
│   ├── Pathmnist_hypergraph_structure.png
│   └── Pathmnist_hypergraph_statistics.png
├── bloodmnist/
│   └── ... (类似结构)
└── organsmnist/
    └── ... (类似结构)
```

### 2. 数据结构
```python
hypergraph_data = {
    'features': np.ndarray,           # 特征矩阵 [N, feature_dim]
    'similarity_matrix': np.ndarray,  # 相似度矩阵 [N, N]
    'hyperedges_dict': dict,          # 超边字典
    'labels': np.ndarray,             # 标签数组
    'n_nodes': int,                   # 节点数量
    'n_hyperedges': int               # 超边数量
}
```

## 🔍 函数详解

### `process_multimodal_datasets(data_loaders, num_samples, save_dir)`
**功能**: 将多个数据集转换为超图结构并生成可视化
**输入**: 
- `data_loaders`: 数据加载器字典
- `num_samples`: 每个数据集使用的样本数
- `save_dir`: 结果保存目录
**输出**: 超图结果字典

### `load_medmnist_dataset(dataset_name, batch_size, num_samples)`
**功能**: 加载单个MedMNIST数据集
**输入**:
- `dataset_name`: 数据集名称
- `batch_size`: 批次大小
- `num_samples`: 样本数量
**输出**: 图像张量和标签张量

## ⚠️ 注意事项

### 1. 内存使用
- 大量样本可能消耗大量内存
- 建议根据系统内存调整 `num_samples`
- 默认设置为256样本用于平衡性能和质量

### 2. 网络连接
- 首次运行需要下载数据集
- 需要稳定的网络连接
- 数据集大小约几十MB到几百MB

### 3. 依赖项
确保安装以下包：
```bash
pip install medmnist hypernetx torch torchvision matplotlib seaborn scikit-learn
```

## 🐛 故障排除

### 问题1: 导入错误
**症状**: `ModuleNotFoundError: No module named 'preprocess'`
**解决**: 确保在正确的目录中运行脚本，或检查Python路径

### 问题2: 数据下载失败
**症状**: 网络错误或下载超时
**解决**: 
- 检查网络连接
- 手动下载数据集到指定目录
- 设置 `download=False` 使用本地数据

### 问题3: 内存不足
**症状**: `RuntimeError: CUDA out of memory` 或系统内存不足
**解决**:
- 减少 `num_samples` 参数
- 减少 `batch_size` 参数
- 关闭其他占用内存的程序

### 问题4: 可视化失败
**症状**: matplotlib 或 seaborn 相关错误
**解决**:
- 确保安装了图形界面支持
- 在服务器环境中设置 matplotlib backend
- 检查图形库版本兼容性

## 📚 相关文档

- `preprocess.py`: 完整的预处理和可视化功能
- `load_dataset.py`: 数据加载接口
- `test_data_pipeline.py`: 测试和验证脚本
- `demo_multi_hypergraph.py`: 交互式演示脚本

## 🤝 贡献

如果遇到问题或有改进建议：
1. 检查此文档的故障排除部分
2. 运行测试脚本验证环境
3. 提交详细的错误信息和复现步骤
