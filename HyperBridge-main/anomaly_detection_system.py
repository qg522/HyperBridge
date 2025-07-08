#!/usr/bin/env python3
"""
Multimodal Hypergraph Neural Network for Anomaly Detection
基于多模态超图神经网络的异常检测系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist
from models.modules.hyper_generator import HybridHyperedgeGenerator
from models.modules.wavelet_cheb_conv import WaveletChebConv
from models.modules.pruning_regularizer import SpectralCutRegularizer


class MultimodalHypergraphAnomalyDetector(nn.Module):
    """
    多模态超图神经网络异常检测器
    
    Architecture:
    1. 多模态特征提取器 (图像/文本/信号)
    2. 超图生成器 (动态构建多模态超图)
    3. 小波切比雪夫超图卷积层
    4. 表征学习与异常评分
    """
    
    def __init__(self, config):
        super(MultimodalHypergraphAnomalyDetector, self).__init__()
        self.config = config
        
        # 多模态特征提取器
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(192, config['hidden_dim']),  # 3*8*8=192
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'])
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(config['text_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'])
        )
        
        self.signal_encoder = nn.Sequential(
            nn.Linear(config['signal_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'])
        )
        
        # 超图生成器
        self.hypergraph_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[config['hidden_dim']] * 3,
            hidden_dim=config['hidden_dim'],
            top_k=config['top_k'],
            threshold=config['threshold']
        )
        
        # 小波切比雪夫超图卷积层
        self.hypergraph_conv1 = WaveletChebConv(
            in_dim=config['hidden_dim'] * 3,  # 融合后的特征
            out_dim=config['hidden_dim'],
            K=config['cheb_k'],
            tau=config['tau']
        )
        
        self.hypergraph_conv2 = WaveletChebConv(
            in_dim=config['hidden_dim'],
            out_dim=config['repr_dim'],
            K=config['cheb_k'],
            tau=config['tau']
        )
        
        # 表征学习层
        self.representation_layer = nn.Sequential(
            nn.Linear(config['repr_dim'], config['repr_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['repr_dim'] // 2, config['final_repr_dim'])
        )
        
        # 异常评分器
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(config['final_repr_dim'], config['final_repr_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['final_repr_dim'] // 2, 1),
            nn.Sigmoid()  # 输出[0,1]的异常分数
        )
        
        # 重构器 (用于自监督学习)
        self.reconstructor = nn.Sequential(
            nn.Linear(config['final_repr_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] * 3)  # 重构多模态特征
        )
        
        # 谱剪枝正则化器
        self.pruning_regularizer = SpectralCutRegularizer(
            use_rayleigh=True,
            reduction='mean'
        )
        
    def forward(self, images, text_features, signal_features, return_all=False):
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 图像数据
            text_features: [B, text_dim] 文本特征
            signal_features: [B, signal_dim] 信号特征
            return_all: 是否返回所有中间结果
        
        Returns:
            anomaly_scores: [B, 1] 异常分数
            reconstruction_loss: 重构损失
            spectral_loss: 谱剪枝损失
        """
        batch_size = images.shape[0]
        
        # 1. 多模态特征提取
        img_features = self.image_encoder(images)
        text_feat = self.text_encoder(text_features)
        signal_feat = self.signal_encoder(signal_features)
        
        # 2. 超图生成
        H, edge_weights = self.hypergraph_generator([img_features, text_feat, signal_feat])
        
        # 3. 计算超图拉普拉斯矩阵
        L = self._compute_hypergraph_laplacian(H, edge_weights)
        
        # 4. 节点特征准备 (融合多模态特征)
        node_features = torch.cat([img_features, text_feat, signal_feat], dim=1)
        
        # 5. 超图卷积层
        x1 = self.hypergraph_conv1(node_features, L)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.hypergraph_conv2(x1, L)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        # 6. 表征学习
        representations = self.representation_layer(x2)
        
        # 7. 异常评分
        anomaly_scores = self.anomaly_scorer(representations)
        
        # 8. 重构损失 (自监督)
        reconstructed = self.reconstructor(representations)
        original_features = torch.cat([img_features, text_feat, signal_feat], dim=1)
        reconstruction_loss = F.mse_loss(reconstructed, original_features)
        
        # 9. 谱剪枝正则化
        Dv, De = self._compute_degree_matrices(H)
        spectral_loss = self.pruning_regularizer(representations, H, Dv, De)
        
        if return_all:
            return {
                'anomaly_scores': anomaly_scores,
                'representations': representations,
                'reconstruction_loss': reconstruction_loss,
                'spectral_loss': spectral_loss,
                'hypergraph_H': H,
                'edge_weights': edge_weights,
                'multimodal_features': {
                    'image': img_features,
                    'text': text_feat,
                    'signal': signal_feat
                }
            }
        else:
            return anomaly_scores, reconstruction_loss, spectral_loss
    
    def _compute_hypergraph_laplacian(self, H, edge_weights=None):
        """计算归一化超图拉普拉斯矩阵"""
        n_nodes, n_edges = H.shape
        
        # 确保所有张量在同一设备上
        device = H.device
        
        # 节点度和边度
        if edge_weights is not None:
            edge_weights = edge_weights.to(device)
            Dv = torch.diag(torch.sum(H * edge_weights.unsqueeze(0), dim=1))
        else:
            Dv = torch.diag(torch.sum(H, dim=1))
        
        De = torch.diag(torch.sum(H, dim=0))
        
        # 确保度矩阵在正确设备上
        Dv = Dv.to(device)
        De = De.to(device)
        
        # 避免除零
        Dv_diag = torch.diag(Dv) + 1e-8
        De_diag = torch.diag(De) + 1e-8
        
        # 归一化拉普拉斯矩阵
        Dv_inv_sqrt = torch.diag(torch.pow(Dv_diag, -0.5))
        De_inv = torch.diag(torch.pow(De_diag, -1.0))
        
        W = torch.mm(torch.mm(torch.mm(Dv_inv_sqrt, H), De_inv), H.t())
        W = torch.mm(W, Dv_inv_sqrt)
        
        I = torch.eye(n_nodes, device=device)
        L = I - W
        
        return L
    
    def _compute_degree_matrices(self, H):
        """计算度矩阵"""
        device = H.device
        node_degrees = H.sum(dim=1) + 1e-8
        edge_degrees = H.sum(dim=0) + 1e-8
        
        Dv = torch.diag(node_degrees).to(device)
        De = torch.diag(edge_degrees).to(device)
        
        return Dv, De


class AnomalyDetectionTrainer:
    """异常检测训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = MultimodalHypergraphAnomalyDetector(config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
    def prepare_anomaly_data(self, dataset_name='pathmnist', anomaly_ratio=0.1, num_samples=500):
        """
        准备异常检测数据
        
        Args:
            dataset_name: 数据集名称
            anomaly_ratio: 异常样本比例
            num_samples: 总样本数
        """
        print(f"Preparing anomaly detection data from {dataset_name}...")
        
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
        
        images = torch.cat(all_images)[:num_samples].to(self.device)
        labels = torch.cat(all_labels)[:num_samples].to(self.device)
        
        # 处理图像格式
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # 创建异常标签 (0: 正常, 1: 异常)
        # 选择部分类别作为异常
        unique_labels = torch.unique(labels)
        num_anomaly_classes = max(1, int(len(unique_labels) * 0.3))  # 30%的类别作为异常
        anomaly_classes = unique_labels[:num_anomaly_classes]
        
        anomaly_labels = torch.zeros_like(labels, device=self.device)
        for cls in anomaly_classes:
            anomaly_labels[labels == cls] = 1
        
        # 模拟文本特征 (基于标签生成模式)
        text_features = torch.zeros(num_samples, self.config['text_dim'], device=self.device)
        for i, label in enumerate(labels):
            # 正常样本有规律的模式，异常样本随机
            if anomaly_labels[i] == 0:  # 正常样本
                pattern = torch.zeros(self.config['text_dim'], device=self.device)
                pattern[label.item() % self.config['text_dim']] = 1.0
                pattern += torch.randn(self.config['text_dim'], device=self.device) * 0.1
            else:  # 异常样本
                pattern = torch.randn(self.config['text_dim'], device=self.device) * 0.5
            text_features[i] = pattern
        
        # 模拟信号特征 (基于图像统计量)
        signal_features = torch.zeros(num_samples, self.config['signal_dim'], device=self.device)
        for i in range(num_samples):
            img = images[i]
            stats = torch.tensor([
                img.mean(), img.std(), img.max(), img.min(),
                img.var(), img.median()
            ], device=self.device)
            
            # 异常样本添加噪声
            if anomaly_labels[i] == 1:
                stats += torch.randn_like(stats) * 0.3
            
            # 扩展到指定维度
            repeats = self.config['signal_dim'] // len(stats)
            signal_pattern = stats.repeat(repeats)
            if len(signal_pattern) < self.config['signal_dim']:
                padding = torch.randn(self.config['signal_dim'] - len(signal_pattern), device=self.device) * 0.1
                signal_pattern = torch.cat([signal_pattern, padding])
            signal_features[i] = signal_pattern[:self.config['signal_dim']]
        
        # 划分训练和验证集
        train_size = int(0.8 * num_samples)
        indices = torch.randperm(num_samples, device=self.device)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_data = {
            'images': images[train_indices],
            'text': text_features[train_indices],
            'signal': signal_features[train_indices],
            'labels': anomaly_labels[train_indices]
        }
        
        self.val_data = {
            'images': images[val_indices],
            'text': text_features[val_indices],
            'signal': signal_features[val_indices],
            'labels': anomaly_labels[val_indices]
        }
        
        print(f"Data prepared:")
        print(f"  Total samples: {num_samples}")
        print(f"  Training samples: {len(train_indices)} (Normal: {(self.train_data['labels'] == 0).sum()}, Anomaly: {(self.train_data['labels'] == 1).sum()})")
        print(f"  Validation samples: {len(val_indices)} (Normal: {(self.val_data['labels'] == 0).sum()}, Anomaly: {(self.val_data['labels'] == 1).sum()})")
        
        return self.train_data, self.val_data
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_size = self.config['batch_size']
        
        # 随机打乱训练数据
        indices = torch.randperm(len(self.train_data['images']), device=self.device)
        
        num_batches = len(indices) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # 准备批数据
            images = self.train_data['images'][batch_indices].to(self.device)
            text = self.train_data['text'][batch_indices].to(self.device)
            signal = self.train_data['signal'][batch_indices].to(self.device)
            labels = self.train_data['labels'][batch_indices].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            anomaly_scores, reconstruction_loss, spectral_loss = self.model(images, text, signal)
            anomaly_scores = anomaly_scores.squeeze()
            
            # 计算损失
            # 1. 异常检测损失 (二分类交叉熵)
            detection_loss = F.binary_cross_entropy(anomaly_scores, labels)
            
            # 2. 总损失
            total_loss_batch = (detection_loss + 
                              self.config['lambda_recon'] * reconstruction_loss + 
                              self.config['lambda_spectral'] * spectral_loss)
            
            # 反向传播
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}/{num_batches}, '
                      f'Total Loss: {total_loss_batch.item():.4f}, '
                      f'Detection Loss: {detection_loss.item():.4f}, '
                      f'Recon Loss: {reconstruction_loss.item():.4f}, '
                      f'Spectral Loss: {spectral_loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            batch_size = self.config['batch_size']
            num_batches = len(self.val_data['images']) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                images = self.val_data['images'][start_idx:end_idx].to(self.device)
                text = self.val_data['text'][start_idx:end_idx].to(self.device)
                signal = self.val_data['signal'][start_idx:end_idx].to(self.device)
                labels = self.val_data['labels'][start_idx:end_idx].to(self.device).float()
                
                anomaly_scores, reconstruction_loss, spectral_loss = self.model(images, text, signal)
                anomaly_scores = anomaly_scores.squeeze()
                
                detection_loss = F.binary_cross_entropy(anomaly_scores, labels)
                total_loss_batch = (detection_loss + 
                                  self.config['lambda_recon'] * reconstruction_loss + 
                                  self.config['lambda_spectral'] * spectral_loss)
                
                total_loss += total_loss_batch.item()
                
                all_scores.extend(anomaly_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # 计算AUC
        auc = roc_auc_score(all_labels, all_scores)
        self.val_aucs.append(auc)
        
        print(f'Epoch {epoch} Validation - Loss: {avg_loss:.4f}, AUC: {auc:.4f}')
        
        return avg_loss, auc, all_scores, all_labels
    
    def train(self, num_epochs=50, save_dir='anomaly_detection_results'):
        """主训练循环"""
        print(f"\n{'='*80}")
        print("🚀 STARTING ANOMALY DETECTION TRAINING")
        print(f"{'='*80}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_auc = 0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_auc, val_scores, val_labels = self.validate_epoch(epoch)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"✅ New best AUC: {best_auc:.4f}")
        
        print(f"\n{'='*80}")
        print("🎉 TRAINING COMPLETED!")
        print(f"Best validation AUC: {best_auc:.4f}")
        print(f"{'='*80}")
        
        return best_auc
    
    def evaluate_and_visualize(self, save_dir='anomaly_detection_results', dataset_name='dataset'):
        """综合评估和可视化"""
        print("\n📊 Creating comprehensive evaluation...")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        self.model.eval()
        
        # 详细验证
        with torch.no_grad():
            images = self.val_data['images'].to(self.device)
            text = self.val_data['text'].to(self.device)
            signal = self.val_data['signal'].to(self.device)
            labels = self.val_data['labels'].cpu().numpy()
            
            # 获取所有结果
            results = self.model(images, text, signal, return_all=True)
            
            anomaly_scores = results['anomaly_scores'].cpu().numpy().flatten()
            representations = results['representations'].cpu().numpy()
        
        # 计算详细指标
        auc = roc_auc_score(labels, anomaly_scores)
        precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
        
        # 最优阈值
        from sklearn.metrics import f1_score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(labels, anomaly_scores > t) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        pred_labels = (anomaly_scores > best_threshold).astype(int)
        
        print(f"\n📈 FINAL EVALUATION METRICS:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Best F1 Score: {max(f1_scores):.4f}")
        print(f"  Best Threshold: {best_threshold:.4f}")
        print(f"  Precision: {precision_recall_curve(labels, anomaly_scores)[0].mean():.4f}")
        print(f"  Recall: {precision_recall_curve(labels, anomaly_scores)[1].mean():.4f}")
        
        # 创建可视化
        self._create_anomaly_visualizations(
            labels, anomaly_scores, representations, 
            save_dir, auc, best_threshold, dataset_name
        )
        
        return {
            'auc': auc,
            'best_f1': max(f1_scores),
            'best_threshold': best_threshold,
            'anomaly_scores': anomaly_scores,
            'representations': representations,
            'labels': labels
        }
    
    def _create_anomaly_visualizations(self, labels, scores, representations, 
                                     save_dir, auc, threshold, dataset_name='dataset'):
        """创建异常检测可视化"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 训练曲线
        plt.subplot(3, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. AUC曲线
        plt.subplot(3, 4, 2)
        plt.plot(self.val_aucs, label='Validation AUC', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title(f'AUC Evolution - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.subplot(3, 4, 3)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 精确率-召回率曲线
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.subplot(3, 4, 4)
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {dataset_name.upper()}')
        plt.grid(True, alpha=0.3)
        
        # 5. 异常分数分布
        plt.subplot(3, 4, 5)
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title(f'Score Distribution - {dataset_name.upper()}')
        plt.legend()
        
        # 6. 混淆矩阵
        pred_labels = (scores > threshold).astype(int)
        cm = confusion_matrix(labels, pred_labels)
        plt.subplot(3, 4, 6)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {dataset_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 7. 表征空间可视化 (t-SNE)
        plt.subplot(3, 4, 7)
        if representations.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(representations)//4))
            repr_2d = tsne.fit_transform(representations)
        else:
            repr_2d = representations
        
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        plt.scatter(repr_2d[normal_mask, 0], repr_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, s=30, label='Normal')
        plt.scatter(repr_2d[anomaly_mask, 0], repr_2d[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=30, label='Anomaly')
        plt.title(f'Representation Space (t-SNE) - {dataset_name.upper()}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # 8. 分数vs表征的散点图
        plt.subplot(3, 4, 8)
        plt.scatter(repr_2d[:, 0], scores, c=labels, cmap='coolwarm', alpha=0.7)
        plt.xlabel('t-SNE 1')
        plt.ylabel('Anomaly Score')
        plt.title(f'Score vs Representation - {dataset_name.upper()}')
        plt.colorbar(label='Label')
        
        # 9-12. 详细统计
        # 9. F1分数vs阈值
        plt.subplot(3, 4, 9)
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(labels, scores > t) for t in thresholds]
        plt.plot(thresholds, f1_scores, linewidth=2)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Best: {threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs Threshold - {dataset_name.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 10. 异常样本示例
        plt.subplot(3, 4, 10)
        # 找到分数最高的异常样本
        anomaly_indices = np.where(labels == 1)[0]
        if len(anomaly_indices) > 0:
            top_anomaly_idx = anomaly_indices[np.argmax(scores[anomaly_indices])]
            # 这里显示文本，因为图像可视化复杂
            plt.text(0.1, 0.5, f'Top Anomaly Sample\nIndex: {top_anomaly_idx}\nScore: {scores[top_anomaly_idx]:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.title(f'Top Anomaly Sample - {dataset_name.upper()}')
        plt.axis('off')
        
        # 11. 正常样本示例
        plt.subplot(3, 4, 11)
        normal_indices = np.where(labels == 0)[0]
        if len(normal_indices) > 0:
            top_normal_idx = normal_indices[np.argmin(scores[normal_indices])]
            plt.text(0.1, 0.5, f'Top Normal Sample\nIndex: {top_normal_idx}\nScore: {scores[top_normal_idx]:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title(f'Top Normal Sample - {dataset_name.upper()}')
        plt.axis('off')
        
        # 12. 总结统计
        plt.subplot(3, 4, 12)
        stats_text = f"""
ANOMALY DETECTION SUMMARY
=======================
Dataset: {dataset_name.upper()}
Dataset Size: {len(labels)}
Normal Samples: {np.sum(labels == 0)}
Anomaly Samples: {np.sum(labels == 1)}

Performance Metrics:
- AUC-ROC: {auc:.4f}
- Best F1: {max(f1_scores):.4f}
- Best Threshold: {threshold:.4f}
- Accuracy: {np.mean(pred_labels == labels):.4f}

Model Architecture:
- Multimodal Features
- Hypergraph Generation  
- Wavelet Convolution
- Spectral Regularization
        """
        plt.text(0.05, 0.95, stats_text.strip(), transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        plt.axis('off')
        
        plt.suptitle(f'Multimodal Hypergraph Anomaly Detection Results - {dataset_name.upper()}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_dir}/anomaly_detection_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {save_dir}/anomaly_detection_results.png")


def main():
    """主函数"""
    # 配置参数
    config = {
        # 模型架构
        'hidden_dim': 64,
        'text_dim': 32,
        'signal_dim': 24,
        'repr_dim': 32,
        'final_repr_dim': 16,
        
        # 超图参数
        'top_k': 8,
        'threshold': 0.3,
        
        # 小波参数
        'cheb_k': 5,
        'tau': 0.5,
        
        # 训练参数
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lambda_recon': 0.1,     # 重构损失权重
        'lambda_spectral': 0.01, # 谱正则化权重
    }
    
    print("🚀 Multimodal Hypergraph Anomaly Detection System")
    print("="*70)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    # 测试不同数据集
    datasets_to_test = ['pathmnist', 'bloodmnist', 'organsmnist']
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing anomaly detection on {dataset_name.upper()}")
        
        try:
            # 初始化训练器
            trainer = AnomalyDetectionTrainer(config)
            
            # 准备数据
            train_data, val_data = trainer.prepare_anomaly_data(
                dataset_name, anomaly_ratio=0.15, num_samples=400
            )
            
            # 训练
            save_dir = f'anomaly_detection_results/{dataset_name}'
            best_auc = trainer.train(num_epochs=30, save_dir=save_dir)
            
            # 评估和可视化
            final_results = trainer.evaluate_and_visualize(save_dir, dataset_name)
            
            print(f"\n📊 {dataset_name.upper()} FINAL RESULTS:")
            print(f"  Best AUC: {best_auc:.4f}")
            print(f"  Final AUC: {final_results['auc']:.4f}")
            print(f"  Best F1 Score: {final_results['best_f1']:.4f}")
            print(f"  Optimal Threshold: {final_results['best_threshold']:.4f}")
            
        except Exception as e:
            print(f"❌ Error in anomaly detection for {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "-"*80)
    
    print(f"\n🎉 Anomaly detection testing completed!")
    print(f"📁 Results saved in 'anomaly_detection_results' directory")


if __name__ == "__main__":
    main()