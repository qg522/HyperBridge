#!/usr/bin/env python3
"""
Isolation Forest Anomaly Detection Comparison
使用Isolation Forest进行异常检测的对比实验
在PathMNIST、OrganSMNIST和BloodMNIST数据集上进行实验
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
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


class IsolationForestAnomalyDetector:
    """
    基于Isolation Forest的异常检测器
    """
    
    def __init__(self, config):
        self.config = config
        
        # 初始化Isolation Forest
        self.model = IsolationForest(
            n_estimators=config.get('n_estimators', 100),
            max_samples=config.get('max_samples', 'auto'),
            contamination=config.get('contamination', 'auto'),
            max_features=config.get('max_features', 1.0),
            bootstrap=config.get('bootstrap', False),
            n_jobs=config.get('n_jobs', -1),
            random_state=config.get('random_state', 42)
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
        """训练Isolation Forest模型"""
        print("\n🚀 Training Isolation Forest...")
        
        # 特征标准化
        features_scaled = self.scaler.fit_transform(self.train_data['features'])
        
        # 训练模型 (Isolation Forest是无监督的，不需要标签)
        self.model.fit(features_scaled)
        
        # 获取训练数据的异常分数
        train_scores = self.model.decision_function(features_scaled)
        train_predictions = self.model.predict(features_scaled)
        
        # 转换预测结果 (Isolation Forest: 1为正常，-1为异常)
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
        
        # 计算F1分数
        f1 = f1_score(true_labels, optimal_predictions)
        
        # 计算准确率
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
    
    def visualize_results(self, results, dataset_name, save_dir):
        """可视化结果"""
        print(f"\n🎨 Creating visualizations for {dataset_name.upper()}...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ROC曲线
        plt.subplot(3, 4, 1)
        fpr, tpr, _ = roc_curve(results['true_labels'], results['anomaly_scores'])
        plt.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {results["auc_roc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Precision-Recall曲线
        plt.subplot(3, 4, 2)
        precision, recall, _ = precision_recall_curve(results['true_labels'], results['anomaly_scores'])
        plt.plot(recall, precision, linewidth=3, label=f'PR (AUC = {results["auc_pr"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 混淆矩阵
        plt.subplot(3, 4, 3)
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {dataset_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. 异常分数分布
        plt.subplot(3, 4, 4)
        true_labels = results['true_labels']
        scores = results['anomaly_scores']
        
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        
        plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(results['optimal_threshold'], color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold: {results["optimal_threshold"]:.3f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Score Distribution - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 特征空间可视化 (t-SNE)
        plt.subplot(3, 4, 5)
        if results['test_features'].shape[1] > 2:
            try:
                tsne = TSNE(n_components=2, random_state=42, 
                           perplexity=min(30, len(results['test_features'])//3))
                features_2d = tsne.fit_transform(results['test_features'])
            except:
                # 如果t-SNE失败，使用PCA
                pca = PCA(n_components=2, random_state=42)
                features_2d = pca.fit_transform(results['test_features'])
        else:
            features_2d = results['test_features']
        
        normal_mask = true_labels == 0
        anomaly_mask = true_labels == 1
        
        plt.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                   c='blue', alpha=0.6, s=30, label='Normal')
        plt.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                   c='red', alpha=0.8, s=30, label='Anomaly')
        plt.title(f'Feature Space (t-SNE) - {dataset_name.upper()}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        
        # 6. F1分数 vs 阈值
        plt.subplot(3, 4, 6)
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for t in thresholds:
            pred = (scores > t).astype(int)
            try:
                f1 = f1_score(true_labels, pred)
            except:
                f1 = 0
            f1_scores.append(f1)
        
        plt.plot(thresholds, f1_scores, linewidth=2)
        plt.axvline(results['optimal_threshold'], color='red', linestyle='--',
                   label=f'Optimal: {results["optimal_threshold"]:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs Threshold - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 异常分数 vs 特征散点图
        plt.subplot(3, 4, 7)
        plt.scatter(features_2d[:, 0], scores, c=true_labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('Feature Component 1')
        plt.ylabel('Anomaly Score')
        plt.title(f'Score vs Features - {dataset_name.upper()}')
        plt.colorbar(label='True Label')
        
        # 8. 模型配置信息
        plt.subplot(3, 4, 8)
        config_text = f"""
ISOLATION FOREST CONFIG
=====================
Dataset: {dataset_name.upper()}
N Estimators: {self.config.get('n_estimators', 100)}
Max Samples: {self.config.get('max_samples', 'auto')}
Contamination: {self.config.get('contamination', 'auto')}
Max Features: {self.config.get('max_features', 1.0)}

FEATURE STRATEGY
===============
✓ Image PCA Features
✓ Simulated Text Features
✓ Simulated Signal Features  
✓ Statistical Features
✓ Consistent with Hypergraph Method

PERFORMANCE METRICS
==================
AUC-ROC: {results['auc_roc']:.4f}
AUC-PR: {results['auc_pr']:.4f}
F1 Score: {results['f1_score']:.4f}
Accuracy: {results['accuracy']:.4f}
Optimal Threshold: {results['optimal_threshold']:.4f}

DATA STATISTICS
==============
Test Samples: {len(true_labels)}
Normal: {np.sum(true_labels == 0)}
Anomaly: {np.sum(true_labels == 1)}
Feature Dim: {results['test_features'].shape[1]}
        """
        plt.text(0.05, 0.95, config_text.strip(), transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        plt.axis('off')
        
        # 9-12. 额外的分析图表
        # 9. 正常样本示例
        plt.subplot(3, 4, 9)
        normal_indices = np.where(true_labels == 0)[0]
        if len(normal_indices) > 0:
            best_normal_idx = normal_indices[np.argmin(scores[normal_indices])]
            plt.text(0.1, 0.5, f'Best Normal Sample\nIndex: {best_normal_idx}\nScore: {scores[best_normal_idx]:.4f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title(f'Best Normal Sample - {dataset_name.upper()}')
        plt.axis('off')
        
        # 10. 异常样本示例
        plt.subplot(3, 4, 10)
        anomaly_indices = np.where(true_labels == 1)[0]
        if len(anomaly_indices) > 0:
            best_anomaly_idx = anomaly_indices[np.argmax(scores[anomaly_indices])]
            plt.text(0.1, 0.5, f'Best Anomaly Sample\nIndex: {best_anomaly_idx}\nScore: {scores[best_anomaly_idx]:.4f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Best Anomaly Sample - {dataset_name.upper()}')
        plt.axis('off')
        
        # 11. 精确率和召回率
        plt.subplot(3, 4, 11)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(true_labels, results['predictions'])
        recall = recall_score(true_labels, results['predictions'])
        
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        values = [precision, recall, results['f1_score'], results['accuracy']]
        colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel('Score')
        plt.title(f'Performance Metrics - {dataset_name.upper()}')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 12. 特征重要性 (简化版)
        plt.subplot(3, 4, 12)
        # 计算特征与异常分数的相关性作为重要性的代理
        feature_importance = np.abs(np.corrcoef(results['test_features'].T, scores)[:-1, -1])
        top_features = np.argsort(feature_importance)[-10:]  # 前10个重要特征
        
        plt.barh(range(len(top_features)), feature_importance[top_features], color='lightcoral')
        plt.xlabel('Importance (|Correlation|)')
        plt.ylabel('Feature Index')
        plt.title(f'Top Feature Importance - {dataset_name.upper()}')
        plt.yticks(range(len(top_features)), top_features)
        
        plt.suptitle(f'Isolation Forest Anomaly Detection Results - {dataset_name.upper()}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图片
        save_path = f"{save_dir}/isolation_forest_results_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        
        print(f"✅ Visualizations saved to {save_path}")


def main():
    """主函数"""
    # Isolation Forest配置
    config = {
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 'auto',  # 自动检测异常比例
        'max_features': 1.0,
        'bootstrap': False,
        'n_jobs': -1,
        'random_state': 42
    }
    
    print("🔬 Isolation Forest Anomaly Detection Comparison")
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
    print("="*70)
    
    # 测试数据集
    datasets_to_test = ['pathmnist', 'bloodmnist', 'organsmnist']
    all_results = {}
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing Isolation Forest on {dataset_name.upper()}")
        
        try:
            # 初始化检测器
            detector = IsolationForestAnomalyDetector(config)
            
            # 准备数据
            train_data, test_data = detector.prepare_data(dataset_name, num_samples=500)
            
            # 训练模型
            train_scores, train_predictions = detector.train()
            
            # 评估模型
            results = detector.evaluate(dataset_name)
            
            # 可视化结果
            save_dir = f'isolation_forest_results/{dataset_name}'
            detector.visualize_results(results, dataset_name, save_dir)
            
            # 保存结果
            all_results[dataset_name] = results
            
            print(f"\n📊 {dataset_name.upper()} SUMMARY:")
            print(f"  AUC-ROC: {results['auc_roc']:.4f}")
            print(f"  AUC-PR: {results['auc_pr']:.4f}")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error in Isolation Forest for {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "-"*80)
    
    # 创建对比图表
    if len(all_results) > 1:
        create_comparison_plot(all_results)
    
    # 输出最终汇总结果
    print(f"\n🎯 ISOLATION FOREST FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<12} {'AUC-ROC':<8} {'AUC-PR':<8} {'F1-Score':<8} {'Accuracy':<8}")
    print("-"*80)
    for dataset_name, results in all_results.items():
        print(f"{dataset_name.upper():<12} {results['auc_roc']:<8.4f} {results['auc_pr']:<8.4f} {results['f1_score']:<8.4f} {results['accuracy']:<8.4f}")
    
    # 计算平均值
    if len(all_results) > 1:
        avg_auc_roc = np.mean([results['auc_roc'] for results in all_results.values()])
        avg_auc_pr = np.mean([results['auc_pr'] for results in all_results.values()])
        avg_f1 = np.mean([results['f1_score'] for results in all_results.values()])
        avg_accuracy = np.mean([results['accuracy'] for results in all_results.values()])
        
        print("-"*80)
        print(f"{'AVERAGE':<12} {avg_auc_roc:<8.4f} {avg_auc_pr:<8.4f} {avg_f1:<8.4f} {avg_accuracy:<8.4f}")
    print("="*80)
    
    print(f"\n🎉 Isolation Forest comparison completed!")
    print(f"📁 Results saved in 'isolation_forest_results' directory")


def create_comparison_plot(all_results):
    """创建不同数据集的对比图表"""
    print("\n📊 Creating comparison plots...")
    
    datasets = list(all_results.keys())
    metrics = ['auc_roc', 'auc_pr', 'f1_score', 'accuracy']
    metric_names = ['AUC-ROC', 'AUC-PR', 'F1 Score', 'Accuracy']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [all_results[dataset][metric] for dataset in datasets]
        
        bars = axes[i].bar(datasets, values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[i].set_title(f'{metric_name} Comparison')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Isolation Forest Performance Comparison Across Datasets', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存对比图
    os.makedirs('isolation_forest_results', exist_ok=True)
    plt.savefig('isolation_forest_results/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示
    
    print("✅ Comparison plot saved to isolation_forest_results/comparison_plot.png")


if __name__ == "__main__":
    main()
