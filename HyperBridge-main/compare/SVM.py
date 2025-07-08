#!/usr/bin/env python3
"""
One-Class SVM Anomaly Detection Comparison
使用One-Class SVM进行异常检测的对比实验
在PathMNIST、OrganSMNIST和BloodMNIST数据集上进行实验
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, f1_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist


class OneClassSVMAnomalyDetector:
    """
    基于One-Class SVM的异常检测器
    """
    
    def __init__(self, config):
        self.config = config
        
        # 初始化One-Class SVM
        self.model = OneClassSVM(
            kernel=config.get('kernel', 'rbf'),
            gamma=config.get('gamma', 'scale'),
            nu=config.get('nu', 0.5),
            degree=config.get('degree', 3),
            coef0=config.get('coef0', 0.0),
            tol=config.get('tol', 1e-3),
            shrinking=config.get('shrinking', True),
            cache_size=config.get('cache_size', 200),
            verbose=config.get('verbose', False),
            max_iter=config.get('max_iter', -1)
        )
        
        # 特征缩放器
        self.scaler = StandardScaler()
        
        # 训练历史
        self.feature_importance = None
        
    def prepare_data(self, dataset_name='pathmnist', num_samples=500):
        """
        准备异常检测数据
        
        Args:
            dataset_name: 数据集名称
            num_samples: 总样本数
        """
        print(f"Preparing data from {dataset_name.upper()}...")
        
        # 加载数据
        data_loaders = load_medmnist(batch_size=32)
        if dataset_name not in data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        train_loader = data_loaders[dataset_name]['train']
        
        # 收集所有数据
        all_images = []
        all_labels = []
        
        for images, labels in train_loader:
            all_images.append(images)
            all_labels.append(labels.squeeze())
            
            if len(torch.cat(all_images)) >= num_samples:
                break
        
        images = torch.cat(all_images)[:num_samples]
        labels = torch.cat(all_labels)[:num_samples]
        
        # 处理图像格式
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # 创建异常标签 (0: 正常, 1: 异常)
        # 选择部分类别作为异常 - 与多模态超图神经网络保持一致
        unique_labels = torch.unique(labels)
        num_anomaly_classes = max(1, int(len(unique_labels) * 0.3))  # 30%的类别作为异常
        anomaly_classes = unique_labels[:num_anomaly_classes]
        
        anomaly_labels = torch.zeros_like(labels)
        for cls in anomaly_classes:
            anomaly_labels[labels == cls] = 1
        
        print(f"Anomaly labeling strategy:")
        print(f"  Total classes: {len(unique_labels)}")
        print(f"  Anomaly classes: {num_anomaly_classes} ({anomaly_classes.tolist()})")
        print(f"  Anomaly ratio: {anomaly_labels.float().mean():.3f}")
        
        # 提取图像特征 (PCA降维)
        image_features = self._extract_image_features(images)
        
        # 模拟文本特征 (基于标签生成模式) - 与原方法保持一致
        text_features = self._generate_text_features(labels, anomaly_labels, num_samples)
        
        # 模拟信号特征 (基于图像统计量) - 与原方法保持一致
        signal_features = self._generate_signal_features(images, anomaly_labels, num_samples)
        
        # 添加传统统计特征
        statistical_features = self._extract_statistical_features(images)
        
        # 合并所有特征：图像PCA + 模拟文本 + 模拟信号 + 统计特征
        all_features = np.concatenate([
            image_features, 
            text_features, 
            signal_features, 
            statistical_features
        ], axis=1)
        
        # 划分训练和测试集
        train_size = int(0.8 * num_samples)
        indices = np.random.permutation(num_samples)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        self.train_data = {
            'features': all_features[train_indices],
            'labels': anomaly_labels[train_indices].numpy()
        }
        
        self.test_data = {
            'features': all_features[test_indices],
            'labels': anomaly_labels[test_indices].numpy()
        }
        
        print(f"Data prepared:")
        print(f"  Total samples: {num_samples}")
        print(f"  Training samples: {len(train_indices)} (Normal: {np.sum(self.train_data['labels'] == 0)}, Anomaly: {np.sum(self.train_data['labels'] == 1)})")
        print(f"  Test samples: {len(test_indices)} (Normal: {np.sum(self.test_data['labels'] == 0)}, Anomaly: {np.sum(self.test_data['labels'] == 1)})")
        print(f"  Feature dimensions:")
        print(f"    - Image PCA features: {image_features.shape[1]}")
        print(f"    - Simulated text features: {text_features.shape[1]}")
        print(f"    - Simulated signal features: {signal_features.shape[1]}")
        print(f"    - Statistical features: {statistical_features.shape[1]}")
        print(f"    - Total feature dimension: {all_features.shape[1]}")
        print(f"  Anomaly labeling consistent with multimodal hypergraph method")
        
        return self.train_data, self.test_data
    
    def _extract_image_features(self, images):
        """提取图像特征"""
        batch_size, channels, height, width = images.shape
        
        # 展平图像并降采样
        images_flat = images.view(batch_size, -1).numpy()
        
        # 使用PCA降维以减少计算量
        if images_flat.shape[1] > 100:
            pca = PCA(n_components=50, random_state=42)
            images_pca = pca.fit_transform(images_flat)
        else:
            images_pca = images_flat
        
        return images_pca
    
    def _extract_statistical_features(self, images):
        """提取图像统计特征"""
        batch_size = images.shape[0]
        features = []
        
        for i in range(batch_size):
            img = images[i].numpy()
            
            # 基本统计量
            stats = [
                img.mean(),           # 均值
                img.std(),            # 标准差
                img.max(),            # 最大值
                img.min(),            # 最小值
                img.var(),            # 方差
                np.median(img),       # 中位数
                np.percentile(img, 25),  # 25%分位数
                np.percentile(img, 75),  # 75%分位数
            ]
            
            # 每个通道的统计量
            for c in range(img.shape[0]):
                channel = img[c]
                stats.extend([
                    channel.mean(),
                    channel.std(),
                    channel.max(),
                    channel.min()
                ])
            
            features.append(stats)
        
        return np.array(features)
    
    def _generate_text_features(self, labels, anomaly_labels, num_samples, text_dim=32):
        """
        生成模拟文本特征 (与多模态超图神经网络保持一致)
        正常样本有规律的模式，异常样本随机
        """
        text_features = np.zeros((num_samples, text_dim))
        
        for i, label in enumerate(labels):
            if anomaly_labels[i] == 0:  # 正常样本
                pattern = np.zeros(text_dim)
                pattern[label.item() % text_dim] = 1.0
                pattern += np.random.randn(text_dim) * 0.1  # 添加小噪声
            else:  # 异常样本
                pattern = np.random.randn(text_dim) * 0.5  # 随机模式
            text_features[i] = pattern
        
        return text_features
    
    def _generate_signal_features(self, images, anomaly_labels, num_samples, signal_dim=24):
        """
        生成模拟信号特征 (基于图像统计量，与多模态超图神经网络保持一致)
        """
        signal_features = np.zeros((num_samples, signal_dim))
        
        for i in range(num_samples):
            img = images[i].numpy()
            # 基础统计量
            stats = np.array([
                img.mean(), img.std(), img.max(), img.min(),
                img.var(), np.median(img)
            ])
            
            # 异常样本添加噪声
            if anomaly_labels[i] == 1:
                stats += np.random.randn(len(stats)) * 0.3
            
            # 扩展到指定维度
            repeats = signal_dim // len(stats)
            signal_pattern = np.tile(stats, repeats)
            
            # 如果维度不够，添加随机填充
            if len(signal_pattern) < signal_dim:
                padding = np.random.randn(signal_dim - len(signal_pattern)) * 0.1
                signal_pattern = np.concatenate([signal_pattern, padding])
            
            signal_features[i] = signal_pattern[:signal_dim]
        
        return signal_features
    
    def train(self):
        """训练One-Class SVM模型"""
        print("\n🚀 Training One-Class SVM...")
        
        # 特征标准化
        features_scaled = self.scaler.fit_transform(self.train_data['features'])
        
        # One-Class SVM只用正常样本训练
        normal_mask = self.train_data['labels'] == 0
        normal_features = features_scaled[normal_mask]
        
        print(f"Training on {len(normal_features)} normal samples...")
        
        # 训练模型 (One-Class SVM是无监督的，只用正常样本)
        self.model.fit(normal_features)
        
        # 获取训练数据的异常分数
        train_scores = self.model.decision_function(features_scaled)
        train_predictions = self.model.predict(features_scaled)
        
        # 转换预测结果 (One-Class SVM: 1为正常，-1为异常)
        train_predictions_binary = (train_predictions == -1).astype(int)
        
        print("✅ Training completed!")
        print(f"Training set anomaly detection rate: {np.mean(train_predictions_binary):.3f}")
        
        return train_scores, train_predictions_binary
    
    def evaluate(self, dataset_name='dataset'):
        """评估模型性能"""
        print(f"\n📊 Evaluating on {dataset_name.upper()}...")
        
        # 特征标准化
        test_features_scaled = self.scaler.transform(self.test_data['features'])
        
        # 预测
        anomaly_scores = self.model.decision_function(test_features_scaled)
        predictions = self.model.predict(test_features_scaled)
        
        # 转换预测结果和分数
        predictions_binary = (predictions == -1).astype(int)
        # 将decision_function的分数转换为[0,1]范围的异常分数
        # One-Class SVM的decision_function返回到超平面的距离，负值表示异常
        anomaly_scores_normalized = 1 / (1 + np.exp(anomaly_scores))  # sigmoid变换
        
        true_labels = self.test_data['labels']
        
        # 计算指标
        auc = roc_auc_score(true_labels, anomaly_scores_normalized)
        
        # 寻找最佳阈值
        fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores_normalized)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # 使用最佳阈值进行预测
        optimal_predictions = (anomaly_scores_normalized > optimal_threshold).astype(int)
        
        # 计算F1分数和准确率
        f1 = f1_score(true_labels, optimal_predictions)
        accuracy = np.mean(optimal_predictions == true_labels)
        
        # 精确率-召回率曲线
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, anomaly_scores_normalized)
        pr_auc = np.trapz(precision, recall)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, optimal_predictions)
        
        # 分类报告
        report = classification_report(true_labels, optimal_predictions, 
                                     target_names=['Normal', 'Anomaly'])
        
        results = {
            'auc_roc': auc,
            'auc_pr': pr_auc,
            'f1_score': f1,
            'accuracy': accuracy,
            'optimal_threshold': optimal_threshold,
            'confusion_matrix': cm,
            'classification_report': report,
            'anomaly_scores': anomaly_scores_normalized,
            'predictions': optimal_predictions,
            'true_labels': true_labels,
            'test_features': test_features_scaled
        }
        
        print(f"\n📈 {dataset_name.upper()} EVALUATION RESULTS:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  AUC-PR: {pr_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        return results


def main():
    """主函数"""
    # One-Class SVM配置
    config = {
        'kernel': 'rbf',        # 核函数: 'linear', 'poly', 'rbf', 'sigmoid'
        'gamma': 'scale',       # 核系数: 'scale', 'auto' 或数值
        'nu': 0.3,             # 异常样本的上界和支持向量的下界
        'degree': 3,            # 多项式核的度数
        'coef0': 0.0,          # 核函数的独立项
        'tol': 1e-3,           # 停止准则的容忍度
        'shrinking': True,      # 是否使用收缩启发式
        'cache_size': 200,      # 内核缓存大小(MB)
        'verbose': False,       # 是否启用详细输出
        'max_iter': -1          # 最大迭代次数
    }
    
    print("🔬 One-Class SVM Anomaly Detection Comparison")
    print("📊 Using consistent labeling strategy with Multimodal Hypergraph method")
    print("="*70)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    print("Features used:")
    print("  ✓ Image PCA features (dimension reduction)")
    print("  ✓ Simulated text features (pattern-based)")
    print("  ✓ Simulated signal features (image statistics)")
    print("  ✓ Statistical features (traditional)")
    print("  ✓ Same anomaly labeling as hypergraph method")
    print("  ✓ Training on normal samples only (One-Class)")
    print("="*70)
    
    # 测试数据集
    datasets_to_test = ['pathmnist', 'bloodmnist', 'organsmnist']
    all_results = {}
    
    # 汇总结果表格
    print(f"\n{'='*80}")
    print("📊 ONE-CLASS SVM RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'AUC-ROC':<10} {'AUC-PR':<10} {'F1-Score':<10} {'Accuracy':<10}")
    print(f"{'-'*80}")
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing One-Class SVM on {dataset_name.upper()}")
        
        try:
            # 初始化检测器
            detector = OneClassSVMAnomalyDetector(config)
            
            # 准备数据
            train_data, test_data = detector.prepare_data(dataset_name, num_samples=500)
            
            # 训练模型
            train_scores, train_predictions = detector.train()
            
            # 评估模型
            results = detector.evaluate(dataset_name)
            
            # 保存结果
            all_results[dataset_name] = results
            
            # 简洁输出到表格
            print(f"{dataset_name.upper():<15} {results['auc_roc']:<10.4f} {results['auc_pr']:<10.4f} {results['f1_score']:<10.4f} {results['accuracy']:<10.4f}")
            
        except Exception as e:
            print(f"❌ Error in One-Class SVM for {dataset_name}: {str(e)}")
            print(f"{dataset_name.upper():<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "-"*80)
    
    # 最终汇总
    print(f"\n{'='*80}")
    print("🎉 ONE-CLASS SVM COMPARISON COMPLETED!")
    print(f"{'='*80}")
    
    if all_results:
        print("\n📈 PERFORMANCE SUMMARY:")
        avg_auc_roc = np.mean([r['auc_roc'] for r in all_results.values()])
        avg_auc_pr = np.mean([r['auc_pr'] for r in all_results.values()])
        avg_f1 = np.mean([r['f1_score'] for r in all_results.values()])
        avg_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
        
        print(f"Average AUC-ROC: {avg_auc_roc:.4f}")
        print(f"Average AUC-PR: {avg_auc_pr:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        
        print(f"\n🔍 BEST PERFORMANCE:")
        best_dataset_auc = max(all_results.items(), key=lambda x: x[1]['auc_roc'])
        best_dataset_f1 = max(all_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"Best AUC-ROC: {best_dataset_auc[0].upper()} ({best_dataset_auc[1]['auc_roc']:.4f})")
        print(f"Best F1-Score: {best_dataset_f1[0].upper()} ({best_dataset_f1[1]['f1_score']:.4f})")


if __name__ == "__main__":
    main()
