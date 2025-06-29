"""
HyperBridge模型配置文件
Configuration file for HyperBridge model
"""

import torch


class HyperBridgeConfig:
    """HyperBridge模型配置类"""
    
    def __init__(self):
        # 数据集相关配置
        self.img_channels = 3          # 图像通道数
        self.img_size = 28             # 图像尺寸
        self.vocab_size = 10000        # 词汇表大小
        self.max_seq_len = 50          # 最大序列长度
        self.sig_in = 100              # 信号输入维度
        
        # 模型架构配置
        self.hidden = 128              # 隐藏层维度
        self.embed_dim = 128           # 词嵌入维度
        self.text_hidden = 64          # 文本LSTM隐藏层维度
        self.n_class = 10              # 分类数
        
        # 超图生成配置
        self.top_k = 10                # 超边生成的top-k邻居数
        self.thresh = 0.5              # 超边筛选阈值
        
        # WaveletChebConv配置
        self.K = 5                     # Chebyshev多项式阶数
        self.tau = 0.5                 # 小波核参数
        
        # 训练配置
        self.batch_size = 32           # 批次大小
        self.learning_rate = 0.001     # 学习率
        self.weight_decay = 1e-5       # 权重衰减
        self.num_epochs = 100          # 训练轮数
        self.reg_lambda = 0.01         # 正则化权重
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 其他配置
        self.dropout = 0.1             # Dropout比率
        self.grad_clip = 1.0           # 梯度裁剪阈值
        self.save_interval = 10        # 模型保存间隔
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            # 模型结构参数
            'img_channels': self.img_channels,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'text_hidden': self.text_hidden,
            'sig_in': self.sig_in,
            'hidden': self.hidden,
            'n_class': self.n_class,
            'top_k': self.top_k,
            'thresh': self.thresh,
            'K': self.K,
            'tau': self.tau,
            'dropout': self.dropout
        }
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
    
    def __str__(self):
        """打印配置信息"""
        lines = ["HyperBridge Configuration:"]
        lines.append("-" * 30)
        
        # 数据相关
        lines.append("Data Configuration:")
        lines.append(f"  Image: {self.img_channels}x{self.img_size}x{self.img_size}")
        lines.append(f"  Text: vocab_size={self.vocab_size}, max_len={self.max_seq_len}")
        lines.append(f"  Signal: {self.sig_in}D")
        lines.append(f"  Classes: {self.n_class}")
        
        # 模型架构
        lines.append("\nModel Architecture:")
        lines.append(f"  Hidden dim: {self.hidden}")
        lines.append(f"  Text embed: {self.embed_dim}")
        lines.append(f"  LSTM hidden: {self.text_hidden}")
        lines.append(f"  Dropout: {self.dropout}")
        
        # 超图配置
        lines.append("\nHypergraph Configuration:")
        lines.append(f"  Top-K: {self.top_k}")
        lines.append(f"  Threshold: {self.thresh}")
        lines.append(f"  Chebyshev K: {self.K}")
        lines.append(f"  Wavelet tau: {self.tau}")
        
        # 训练配置
        lines.append("\nTraining Configuration:")
        lines.append(f"  Batch size: {self.batch_size}")
        lines.append(f"  Learning rate: {self.learning_rate}")
        lines.append(f"  Weight decay: {self.weight_decay}")
        lines.append(f"  Epochs: {self.num_epochs}")
        lines.append(f"  Reg lambda: {self.reg_lambda}")
        lines.append(f"  Device: {self.device}")
        
        return "\n".join(lines)


# 预定义配置
def get_default_config():
    """获取默认配置"""
    return HyperBridgeConfig()


def get_small_config():
    """获取小型模型配置（用于快速测试）"""
    config = HyperBridgeConfig()
    config.update(
        hidden=64,
        embed_dim=64,
        text_hidden=32,
        batch_size=16,
        K=3,
        top_k=5
    )
    return config


def get_large_config():
    """获取大型模型配置（用于性能要求高的场景）"""
    config = HyperBridgeConfig()
    config.update(
        hidden=256,
        embed_dim=256,
        text_hidden=128,
        batch_size=64,
        K=7,
        top_k=15,
        vocab_size=50000
    )
    return config


def get_config_for_dataset(dataset_name):
    """根据数据集获取特定配置"""
    config = HyperBridgeConfig()
    
    if dataset_name.lower() == 'pathmnist':
        config.update(
            img_channels=3,
            img_size=28,
            n_class=9,  # PathMNIST有9个类别
            vocab_size=5000,  # 假设病理报告词汇量
            sig_in=50   # 假设有一些病理信号特征
        )
    elif dataset_name.lower() == 'bloodmnist':
        config.update(
            img_channels=3,
            img_size=28,
            n_class=8,  # BloodMNIST有8个类别
            vocab_size=3000,
            sig_in=30
        )
    elif dataset_name.lower() == 'synthetic':
        config.update(
            img_channels=3,
            img_size=28,
            n_class=5,
            vocab_size=1000,
            sig_in=100
        )
    else:
        print(f"Warning: Unknown dataset '{dataset_name}', using default config")
    
    return config


# 配置验证函数
def validate_config(config):
    """验证配置的合理性"""
    issues = []
    
    # 检查基本参数
    if config.hidden <= 0:
        issues.append("hidden dimension must be positive")
    
    if config.top_k >= config.batch_size:
        issues.append("top_k should be smaller than batch_size")
    
    if config.thresh < 0 or config.thresh > 1:
        issues.append("threshold should be between 0 and 1")
    
    if config.K <= 0:
        issues.append("Chebyshev polynomial order K must be positive")
    
    if config.learning_rate <= 0:
        issues.append("learning rate must be positive")
    
    # 检查维度兼容性
    if config.n_class <= 0:
        issues.append("number of classes must be positive")
    
    if config.vocab_size <= 0:
        issues.append("vocabulary size must be positive")
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Configuration validation passed ✅")
        return True


if __name__ == "__main__":
    # 演示不同配置
    print("🔧 HyperBridge Configuration Examples")
    print("=" * 60)
    
    # 默认配置
    print("\n1. Default Configuration:")
    default_config = get_default_config()
    print(default_config)
    validate_config(default_config)
    
    # 小型配置
    print("\n" + "=" * 60)
    print("\n2. Small Configuration:")
    small_config = get_small_config()
    print(small_config)
    validate_config(small_config)
    
    # 数据集特定配置
    print("\n" + "=" * 60)
    print("\n3. PathMNIST Configuration:")
    pathmnist_config = get_config_for_dataset('pathmnist')
    print(pathmnist_config)
    validate_config(pathmnist_config)
