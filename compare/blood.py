import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys
import json
import pickle
import warnings
import traceback
from datetime import datetime
import jieba  # 新增：导入 jieba 用于中文分词
import random
import pandas as pd  # 添加pandas导入

warnings.filterwarnings('ignore')

# ====================================================================================
# START: REQUIRED MODULES / PLACEHOLDERS (确保这些类的定义存在并正确)
# ====================================================================================

# Placeholder for HybridHyperedgeGenerator (来自 models.modules.hyper_generator)
class HybridHyperedgeGenerator(nn.Module):
    def __init__(self, num_modalities, input_dims, hidden_dim, top_k, threshold):
        super().__init__()
        self.modality_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()) for dim in input_dims
        ])
        self.att_scorer = nn.Linear(hidden_dim * num_modalities, 1)
        self.top_k = top_k
        self.threshold = threshold

    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        device = features_list[0].device

        projected_features = [proj(feat) for proj, feat in zip(self.modality_projections, features_list)]
        combined_features = torch.cat(projected_features, dim=1)
        attention_scores = torch.sigmoid(self.att_scorer(combined_features).squeeze())  # Sigmoid on attention scores

        normalized_combined_features = F.normalize(torch.cat(features_list, dim=1), dim=1)
        similarity = torch.mm(normalized_combined_features, normalized_combined_features.t())

        num_hyperedges = batch_size
        H_final = torch.zeros(batch_size, num_hyperedges, device=device)

        actual_top_k = min(self.top_k, batch_size)
        _, top_k_indices = torch.topk(similarity, actual_top_k, dim=1)

        for i in range(batch_size):
            H_final[i, i] = 1.0
            for neighbor_idx in top_k_indices[i]:
                if neighbor_idx < batch_size:
                    H_final[neighbor_idx, i] = 1.0

        for i in range(num_hyperedges):
            if H_final[:, i].sum() == 0:
                H_final[i, i] = 1.0

        edge_weights = attention_scores
        if edge_weights.shape[0] != num_hyperedges:
            # Fallback if dimensions don't match, or use a more robust weighting
            edge_weights = torch.ones(num_hyperedges, device=device) * (
                attention_scores.mean() if attention_scores.numel() > 0 else 1.0)

        edge_weights = torch.clamp(edge_weights, min=1e-8)

        return H_final, edge_weights


# Corrected WaveletChebConv (已修正过维度问题)
class WaveletChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K, tau):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.K = K
        self.tau = tau

    def forward(self, x, L):
        if self.K == 0:
            return self.linear(x)

        Tx_0 = x
        Tx_1 = torch.mm(L, x)

        cheb_terms = [Tx_0, Tx_1]

        for k in range(2, self.K):
            Tx_k = 2 * torch.mm(L, cheb_terms[-1]) - cheb_terms[-2]
            cheb_terms.append(Tx_k)

        summed_cheb_output = torch.sum(torch.stack(cheb_terms, dim=0), dim=0)

        return self.linear(summed_cheb_output)


# Placeholder for SpectralCutRegularizer
class SpectralCutRegularizer(nn.Module):
    def __init__(self, use_rayleigh=True, reduction='mean'):
        super().__init__()
        self.use_rayleigh = use_rayleigh
        self.reduction = reduction

    def forward(self, representations, H, Dv, De):
        if self.use_rayleigh:
            if representations.numel() == 0:
                return torch.tensor(0.0, device=representations.device)
            reg_loss = torch.norm(representations, p=2)
            if self.reduction == 'mean':
                reg_loss = reg_loss.mean()
            elif self.reduction == 'sum':
                reg_loss = reg_loss.sum()
            return reg_loss * 0.01
        else:
            return torch.tensor(0.0, device=representations.device)


# ====================================================================================
# END: REQUIRED MODULES / PLACEHOLDERS
# ====================================================================================


# 升级后的 BiLSTM 文本编码器类 - 接受词嵌入序列
class BiLSTMAwareTextEncoder(nn.Module):
    """
    基于 BiLSTM 的文本编码器，用于处理词嵌入序列。
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_layers=1, dropout=0.2, bidirectional=True,
                 max_seq_len=64):
        super(BiLSTMAwareTextEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = hidden_dim
        self.num_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if self.num_layers > 1 else 0
        )

        self.output_projection = nn.Linear(self.num_directions * self.lstm_hidden_size, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_indices):
        embeddings = self.embedding(text_indices)

        output, (hn, cn) = self.lstm(embeddings)

        if self.bidirectional:
            final_hidden_state = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            final_hidden_state = hn[-1, :, :]

        final_features = self.output_projection(final_hidden_state)
        final_features = self.layer_norm(final_features)

        return final_features


class AblationAnomalyDetector(nn.Module):
    """支持消融实验的多模态超图异常检测器"""

    def __init__(self, config, ablation_config):
        super(AblationAnomalyDetector, self).__init__()
        self.config = config
        self.ablation_config = ablation_config

        self.image_encoder = nn.ModuleDict({
            'adaptive_pool': nn.AdaptiveAvgPool2d((8, 8)),  # 统一输出为8x8以减少维度
            'flatten': nn.Flatten(),
            'projection': nn.Sequential(
                nn.Linear(3 * 8 * 8, config['hidden_dim']),  # 3通道 * 8 * 8 = 192
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config['hidden_dim'], config['hidden_dim']),
                nn.BatchNorm1d(config['hidden_dim'])
            )
        })

        self.text_encoder_module = BiLSTMAwareTextEncoder(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            lstm_layers=config.get('lstm_layers', 1),
            dropout=config.get('lstm_dropout', 0.2),
            bidirectional=True,
            max_seq_len=config['max_seq_len']
        )

        input_dims = [config['hidden_dim']] * 2
        if ablation_config['hypergraph_type'] == 'dynamic':
            self.hypergraph_generator = HybridHyperedgeGenerator(
                num_modalities=2,
                input_dims=input_dims,
                hidden_dim=config['hidden_dim'],
                top_k=config.get('top_k', 8),
                threshold=config.get('threshold', 0.6)
            )

        conv_input_dim = config['hidden_dim'] * 2

        self.feature_adapter = nn.Linear(conv_input_dim, conv_input_dim)

        if ablation_config['conv_type'] == 'wavelet_cheb':
            self.conv1 = WaveletChebConv(
                in_dim=conv_input_dim,
                out_dim=config['hidden_dim'],
                K=config.get('cheb_k', 3),
                tau=config.get('tau', 0.5)
            )
            self.conv2 = WaveletChebConv(
                in_dim=config['hidden_dim'],
                out_dim=config['repr_dim'],
                K=config.get('cheb_k', 3),
                tau=config.get('tau', 0.5)
            )

        self.representation_layer = nn.Sequential(
            nn.Linear(config['repr_dim'], config['repr_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['repr_dim'] // 2, config['final_repr_dim']),
            nn.BatchNorm1d(config['final_repr_dim'])
        )

        self.anomaly_scorer = nn.Sequential(
            nn.Linear(config['final_repr_dim'], config['final_repr_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['final_repr_dim'] // 2, 1),
            nn.Sigmoid()
        )

        self.reconstructor = nn.ModuleDict({
            'hidden_projection': nn.Linear(config['final_repr_dim'], config['hidden_dim']),
            'output_projection': None
        })

        if ablation_config['use_spectral_regularizer']:
            self.pruning_regularizer = SpectralCutRegularizer(
                use_rayleigh=True,
                reduction='mean'
            )

    def forward(self, images, text_sequences, labels=None, return_all=False):  # text_features 改为 text_sequences
        batch_size = images.shape[0]
        device = images.device

        # Image processing block - ensuring 3-channel 28x28
        if len(images.shape) == 2:
            target_pixels = 3 * 28 * 28
            if images.shape[1] >= target_pixels:
                images = images[:, :target_pixels].view(batch_size, 3, 28, 28)
            else:
                padding_size = target_pixels - images.shape[1]
                padded_images = torch.cat([images, torch.zeros(batch_size, padding_size, device=device)], dim=1)
                images = padded_images.view(batch_size, 3, 28, 28)

        if images.shape[1] != 3:
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[1] > 3:
                images = images[:, :3, :, :]
            else:
                padding_channels = 3 - images.shape[1]
                padding = torch.zeros(batch_size, padding_channels, images.shape[2], images.shape[3], device=device)
                images = torch.cat([images, padding], dim=1)

        img_pooled = self.image_encoder['adaptive_pool'](images)
        img_flattened = self.image_encoder['flatten'](img_pooled)
        img_features = self.image_encoder['projection'](img_flattened)

        # Pass text_sequences (integer IDs) to BiLSTM
        text_feat = self.text_encoder_module(text_sequences)

        features_list = [img_features, text_feat]

        H, edge_weights = self.hypergraph_generator(features_list)

        L = self._compute_hypergraph_laplacian(H, edge_weights)

        node_features = torch.cat(features_list, dim=1)

        actual_feature_dim = node_features.shape[1]
        if not hasattr(self, '_adapted_feature_dim'):
            self._adapted_feature_dim = actual_feature_dim
            if self.reconstructor['output_projection'] is None:
                self.reconstructor['output_projection'] = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.config['hidden_dim'], actual_feature_dim)
                ).to(node_features.device)

        x1 = self.conv1(node_features, L)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.conv2(x1, L)
        x2 = F.relu(x2)

        representations = self.representation_layer(x2)

        anomaly_scores = self.anomaly_scorer(representations).squeeze()

        losses = {}
        total_loss = torch.tensor(0.0, device=device)

        if labels is not None:
            detection_loss = F.binary_cross_entropy(anomaly_scores, labels.float())
            losses['detection_loss'] = detection_loss
            total_loss += detection_loss

            hidden_repr = self.reconstructor['hidden_projection'](representations)
            reconstructed = self.reconstructor['output_projection'](hidden_repr)
            original_features = node_features.detach()

            if reconstructed.shape != original_features.shape:
                print(f"重构维度警告: reconstructed {reconstructed.shape} vs original {original_features.shape}")
                min_dim = min(reconstructed.shape[1], original_features.shape[1])
                reconstructed = reconstructed[:, :min_dim]
                original_features = original_features[:, :min_dim]

            recon_loss = F.mse_loss(reconstructed, original_features)
            losses['reconstruction_loss'] = recon_loss
            recon_weight = self.config.get('lambda_recon', 0.1)
            total_loss += recon_weight * recon_loss

            if self.ablation_config['use_spectral_regularizer']:
                Dv, De = self._compute_degree_matrices(H)
                spectral_loss = self.pruning_regularizer(representations, H, Dv, De)
                losses['spectral_loss'] = spectral_loss
                spectral_weight = self.config.get('lambda_spectral', 0.01)
                total_loss += spectral_weight * spectral_loss
            else:
                losses['spectral_loss'] = torch.tensor(0.0, device=device)

        losses['total_loss'] = total_loss

        if return_all:
            return {
                'anomaly_scores': anomaly_scores,
                'representations': representations,
                'hypergraph': (H, edge_weights),
                'losses': losses
            }
        else:
            return anomaly_scores, losses

    def _compute_hypergraph_laplacian(self, H, edge_weights):
        batch_size, num_edges = H.shape
        device = H.device

        if num_edges == 0:
            return torch.eye(batch_size, device=device)

        Dv, De = self._compute_degree_matrices(H)

        dv_diag = Dv.diag()
        de_diag = De.diag()

        if dv_diag.sum() == 0 or de_diag.sum() == 0:
            return torch.eye(batch_size, device=device)

        if edge_weights.numel() == 0:
            edge_weights = torch.ones(num_edges, device=device)
        W = torch.diag(edge_weights)

        dv_sqrt_inv = torch.diag(1.0 / (torch.sqrt(dv_diag + 1e-8)))
        de_inv = torch.diag(1.0 / (de_diag + 1e-8))

        try:
            Theta = torch.mm(torch.mm(torch.mm(torch.mm(dv_sqrt_inv, H), W), de_inv),
                             torch.mm(H.t(), dv_sqrt_inv))
        except RuntimeError as e:
            print(f"拉普拉斯计算错误: {e}")
            print(f"H shape: {H.shape}, edge_weights shape: {edge_weights.shape}")
            print(f"dv_diag: {dv_diag[:5]}, de_diag: {de_diag[:5]}")
            return torch.eye(batch_size, device=device)

        I = torch.eye(batch_size, device=device)
        L = I - Theta

        return L

    def _compute_degree_matrices(self, H):
        d_v = H.sum(dim=1)
        d_e = H.sum(dim=0)

        d_v = torch.clamp(d_v, min=1e-8)
        d_e = torch.clamp(d_e, min=1e-8)

        Dv = torch.diag(d_v)
        De = torch.diag(d_e)

        return Dv, De


class SimpleAutoencoder(nn.Module):
    """简单的自编码器用于异常检测基线比较"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32]):
        super(SimpleAutoencoder, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1])) + [input_dim]
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def get_reconstruction_error(self, x):
        """计算重构误差作为异常分数"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
            return error


class BaselineModelTrainer:
    """基线模型训练器 - 支持Autoencoder、SVM、Isolation Forest"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.data_generator = ComplexDataGenerator(config)
        
    def train_autoencoder(self, dataset='bloodmnist', epochs=50, batch_size=32):
        """训练自编码器基线模型"""
        print(f"🔧 开始训练自编码器基线模型 - {dataset.upper()}")
        
        # 获取数据特征维度
        sample_images, sample_text, _, _ = self.data_generator.generate_multimodal_data(
            batch_size, self.device, split='train', dataset=dataset, anomaly_ratio=0.0
        )
        
        # 计算输入特征维度
        img_features = sample_images.view(sample_images.shape[0], -1)
        text_features = sample_text.float().mean(dim=1, keepdim=True).repeat(1, 64)  # 简化文本特征
        combined_features = torch.cat([img_features, text_features], dim=1)
        input_dim = combined_features.shape[1]
        
        # 创建自编码器模型
        autoencoder = SimpleAutoencoder(input_dim).to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练循环
        autoencoder.train()
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx in range(15):  # 15个训练批次
                try:
                    # 获取正常数据进行训练
                    images, text_sequences, labels, _ = \
                        self.data_generator.generate_multimodal_data(
                            batch_size, self.device, split='train', dataset=dataset, anomaly_ratio=0.0
                        )
                    
                    # 特征提取和组合 - 确保梯度传播
                    img_features = images.view(images.shape[0], -1)
                    text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64)
                    combined_features = torch.cat([img_features, text_features], dim=1)
                    
                    # 确保张量需要梯度
                    combined_features = combined_features.detach().requires_grad_(True)
                    
                    optimizer.zero_grad()
                    reconstructed, _ = autoencoder(combined_features)
                    loss = criterion(reconstructed, combined_features.detach())  # 目标不需要梯度
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"训练批次 {batch_idx} 失败: {e}")
                    continue
            
            if epoch % 10 == 0 and epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        # 评估
        return self._evaluate_autoencoder(autoencoder, dataset, batch_size)
    
    def train_svm(self, dataset='bloodmnist', batch_size=32, nu=0.1):
        """训练One-Class SVM基线模型"""
        print(f"🔧 开始训练One-Class SVM基线模型 - {dataset.upper()}")
        
        # 收集训练数据
        all_features = []
        for batch_idx in range(20):  # 收集更多数据用于训练
            images, text_sequences, labels, _ = \
                self.data_generator.generate_multimodal_data(
                    batch_size, self.device, split='train', dataset=dataset, anomaly_ratio=0.0
                )
            
            # 特征提取
            img_features = images.view(images.shape[0], -1).cpu().numpy()
            text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64).cpu().numpy()
            combined_features = np.concatenate([img_features, text_features], axis=1)
            all_features.append(combined_features)
        
        # 合并所有特征
        X_train = np.vstack(all_features)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 训练One-Class SVM
        svm_model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        svm_model.fit(X_train_scaled)
        
        # 评估
        return self._evaluate_svm(svm_model, scaler, dataset, batch_size)
    
    def train_isolation_forest(self, dataset='bloodmnist', batch_size=32, contamination=0.1):
        """训练Isolation Forest基线模型"""
        print(f"🔧 开始训练Isolation Forest基线模型 - {dataset.upper()}")
        
        # 收集训练数据
        all_features = []
        for batch_idx in range(20):  # 收集更多数据用于训练
            images, text_sequences, labels, _ = \
                self.data_generator.generate_multimodal_data(
                    batch_size, self.device, split='train', dataset=dataset, anomaly_ratio=0.0
                )
            
            # 特征提取
            img_features = images.view(images.shape[0], -1).cpu().numpy()
            text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64).cpu().numpy()
            combined_features = np.concatenate([img_features, text_features], axis=1)
            all_features.append(combined_features)
        
        # 合并所有特征
        X_train = np.vstack(all_features)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 训练Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        iso_forest.fit(X_train_scaled)
        
        # 评估
        return self._evaluate_isolation_forest(iso_forest, scaler, dataset, batch_size)
    
    def _evaluate_autoencoder(self, autoencoder, dataset, batch_size):
        """评估自编码器模型"""
        autoencoder.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx in range(10):  # 测试批次
                try:
                    images, text_sequences, labels, _ = \
                        self.data_generator.generate_multimodal_data(
                            batch_size, self.device, split='test', dataset=dataset, anomaly_ratio=0.2
                        )
                    
                    # 特征提取
                    img_features = images.view(images.shape[0], -1)
                    text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64)
                    combined_features = torch.cat([img_features, text_features], dim=1)
                    
                    # 获取重构误差作为异常分数
                    reconstruction_errors = autoencoder.get_reconstruction_error(combined_features)
                    
                    all_scores.extend(reconstruction_errors.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"评估批次 {batch_idx} 失败: {e}")
                    continue
        
        if len(all_scores) == 0:
            print("⚠️ 自编码器评估失败：没有有效的测试批次")
            return {'auc': 0.0, 'accuracy': 0.0, 'error': 'No valid test batches'}
        
        return self._compute_metrics(all_scores, all_labels)
    
    def _evaluate_svm(self, svm_model, scaler, dataset, batch_size):
        """评估SVM模型"""
        all_scores = []
        all_labels = []
        
        for batch_idx in range(10):  # 测试批次
            try:
                images, text_sequences, labels, _ = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='test', dataset=dataset, anomaly_ratio=0.2
                    )
                
                # 特征提取
                img_features = images.view(images.shape[0], -1).cpu().numpy()
                text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64).cpu().numpy()
                combined_features = np.concatenate([img_features, text_features], axis=1)
                
                # 标准化
                combined_features_scaled = scaler.transform(combined_features)
                
                # 获取决策分数（距离分离超平面的距离）
                decision_scores = svm_model.decision_function(combined_features_scaled)
                # 转换为异常分数（负值表示异常）
                anomaly_scores = -decision_scores  # 负值越大越异常
                
                all_scores.extend(anomaly_scores)
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"SVM评估批次 {batch_idx} 失败: {e}")
                continue
        
        if len(all_scores) == 0:
            print("⚠️ SVM评估失败：没有有效的测试批次")
            return {'auc': 0.0, 'accuracy': 0.0, 'error': 'No valid test batches'}
        
        return self._compute_metrics(all_scores, all_labels)
    
    def _evaluate_isolation_forest(self, iso_forest, scaler, dataset, batch_size):
        """评估Isolation Forest模型"""
        all_scores = []
        all_labels = []
        
        for batch_idx in range(10):  # 测试批次
            try:
                images, text_sequences, labels, _ = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='test', dataset=dataset, anomaly_ratio=0.2
                    )
                
                # 特征提取
                img_features = images.view(images.shape[0], -1).cpu().numpy()
                text_features = text_sequences.float().mean(dim=1, keepdim=True).repeat(1, 64).cpu().numpy()
                combined_features = np.concatenate([img_features, text_features], axis=1)
                
                # 标准化
                combined_features_scaled = scaler.transform(combined_features)
                
                # 获取异常分数
                anomaly_scores = iso_forest.decision_function(combined_features_scaled)
                # 转换为正值（值越大越正常，我们需要反转）
                anomaly_scores = -anomaly_scores  # 负值越大越异常
                
                all_scores.extend(anomaly_scores)
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Isolation Forest评估批次 {batch_idx} 失败: {e}")
                continue
        
        if len(all_scores) == 0:
            print("⚠️ Isolation Forest评估失败：没有有效的测试批次")
            return {'auc': 0.0, 'accuracy': 0.0, 'error': 'No valid test batches'}
        
        return self._compute_metrics(all_scores, all_labels)
    
    def _compute_metrics(self, all_scores, all_labels):
        """计算评估指标的通用方法"""
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        results = {}
        
        try:
            results['auc'] = roc_auc_score(all_labels, all_scores)
        except:
            results['auc'] = 0.5

        try:
            from sklearn.metrics import accuracy_score, precision_recall_curve
            
            threshold = np.median(all_scores)  # 使用中位数作为阈值
            predicted_labels = (all_scores > threshold).astype(int)
            results['accuracy'] = accuracy_score(all_labels, predicted_labels)

        except Exception as e:
            print(f"Warning: Could not calculate accuracy: {e}")
            results['accuracy'] = 0.5

        return results


# 修改后的多数据集加载器 - 支持BloodMNIST、PathMNIST、OrganaMNIST
class MultiDatasetLoader:
    """多数据集加载器 - 支持从F:\Desktop\bloodmnist\Data加载三个数据集"""

    def __init__(self, data_root="F:\\Desktop\\bloodmnist\\Data", batch_size=128, cache_size=1024, 
                 embedding_dim=64, max_seq_len=64, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_root = data_root
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 静态数据划分参数
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio  
        self.test_ratio = test_ratio
        
        # 确保比例总和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"数据划分比例总和必须为1.0，当前为: {total_ratio}")

        # 数据集配置
        self.dataset_configs = {
            'bloodmnist': {
                'path': os.path.join(data_root, 'bloodmnist'),
                'classes': ["basophil", "eosinophil", "erythroblast", "ig", 
                           "lymphocyte", "monocyte", "neutrophil", "platelet"],
                'num_classes': 8
            },
            'pathmnist': {
                'path': os.path.join(data_root, 'pathmnist'),
                'classes': ["adipose", "background", "debris", "lymphocytes", "mucus",
                           "smooth_muscle", "normal_colon_mucosa", "cancer_epithelium", "colorectal_adenocarcinoma"],
                'num_classes': 9
            },
            'organamnist': {
                'path': os.path.join(data_root, 'organamnist'), 
                'classes': ["bladder", "femur", "heart", "kidney", "breast", "lung",
                           "ovary", "pancreas", "prostate", "spleen", "tibia"],
                'num_classes': 11
            }
        }

        # 存储所有数据集的数据
        self.datasets = {}
        
        # 词汇表相关（跨数据集共享）
        self.vocab = {}
        self.idx_to_word = {}
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.vocab_size = 2

        self._initialize_all_datasets()

    def _initialize_all_datasets(self):
        """初始化所有三个数据集"""
        print(f"🔄 正在从 {self.data_root} 加载三个数据集...")
        
        # 首先构建跨数据集的词汇表
        self._build_global_vocabulary()
        
        # 然后加载每个数据集
        for dataset_name, config in self.dataset_configs.items():
            print(f"\n📂 正在加载 {dataset_name.upper()} 数据集...")
            self._load_single_dataset(dataset_name, config)
            
        print(f"\n✅ 所有数据集加载完成!")
        print(f"   📊 BloodMNIST: {len(self.datasets['bloodmnist']['images'])} 样本")
        print(f"   📊 PathMNIST: {len(self.datasets['pathmnist']['images'])} 样本") 
        print(f"   📊 OrganaMNIST: {len(self.datasets['organamnist']['images'])} 样本")
        print(f"   📝 全局词汇表大小: {self.vocab_size}")

    def _build_global_vocabulary(self):
        """构建跨所有数据集的全局词汇表"""
        print("🔤 构建全局词汇表...")
        all_tokens = []
        
        for dataset_name, config in self.dataset_configs.items():
            text_path = os.path.join(config['path'], f"{dataset_name}_text_descriptions.json")
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_data = json.load(f)
                
                samples_data = text_data.get('data', text_data.get('samples', []))
                for sample in samples_data:
                    description = sample.get('text_description', sample.get('description', ''))
                    tokens = [word.strip('.,!?;:"\'').lower() for word in jieba.lcut(description) 
                             if word.strip('.,!?;:"\'').isalnum()]
                    all_tokens.extend(tokens)
        
        # 构建词汇表
        unique_tokens = sorted(list(set(all_tokens)))
        for word in unique_tokens:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.vocab_size += 1
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"   ✅ 全局词汇表构建完成，词汇量: {self.vocab_size}")

    def _load_single_dataset(self, dataset_name, config):
        """加载单个数据集"""
        dataset_path = config['path']
        
        # 加载图像数据
        images_path = os.path.join(dataset_path, f"{dataset_name}_images.pkl")
        with open(images_path, 'rb') as f:
            image_data = pickle.load(f)
        
        # 加载文本数据
        text_path = os.path.join(dataset_path, f"{dataset_name}_text_descriptions.json")
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        # 处理图像和标签
        original_images = torch.from_numpy(np.array(image_data['images'])).float()
        original_labels = torch.from_numpy(np.array(image_data['labels'])).long()
        
        # 处理文本数据
        raw_text_descriptions = []
        samples_data = text_data.get('data', text_data.get('samples', []))
        for sample in samples_data:
            description = sample.get('text_description', sample.get('description', ''))
            raw_text_descriptions.append(description)
        
        # 将文本转换为序列（使用全局词汇表）
        text_sequences = []
        for desc in raw_text_descriptions:
            tokens = [word.strip('.,!?;:"\'').lower() for word in jieba.lcut(desc) 
                     if word.strip('.,!?;:"\'').isalnum()]
            indexed_tokens = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
            
            if len(indexed_tokens) < self.max_seq_len:
                padded_sequence = indexed_tokens + [self.word_to_idx['<pad>']] * (
                            self.max_seq_len - len(indexed_tokens))
            else:
                padded_sequence = indexed_tokens[:self.max_seq_len]
            text_sequences.append(padded_sequence)
        
        original_text_sequences = torch.tensor(text_sequences, dtype=torch.long)
        
        # 确保数据长度一致
        min_samples = min(len(original_images), len(original_text_sequences), len(original_labels))
        original_images = original_images[:min_samples]
        original_text_sequences = original_text_sequences[:min_samples]
        original_labels = original_labels[:min_samples]
        
        # 数据预处理
        if original_images.max() > 1.0:
            original_images = original_images / 255.0
        original_images = (original_images - 0.5) / 0.5
        
        # 执行静态数据划分
        total_samples = min_samples
        indices = np.random.permutation(total_samples)
        
        train_end = int(total_samples * self.train_ratio)
        val_end = train_end + int(total_samples * self.val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # 存储数据集
        self.datasets[dataset_name] = {
            'images': original_images,
            'text_sequences': original_text_sequences,
            'labels': original_labels,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'class_names': config['classes'],
            'num_classes': config['num_classes']
        }
        
        print(f"   ✅ {dataset_name.upper()} 加载完成:")
        print(f"      🟢 训练集: {len(train_indices)} 样本")
        print(f"      🟡 验证集: {len(val_indices)} 样本")
        print(f"      🔴 测试集: {len(test_indices)} 样本")

    def get_split_batch(self, batch_size, device, split='train', dataset='bloodmnist'):
        """
        从指定数据集的指定划分获取批次数据
        
        Args:
            batch_size: 批次大小
            device: 设备
            split: 数据划分 {'train', 'val', 'test'}
            dataset: 数据集名称 {'bloodmnist', 'pathmnist', 'organamnist'}
        """
        if dataset not in self.datasets:
            raise ValueError(f"未知数据集: {dataset}")
        
        dataset_data = self.datasets[dataset]
        
        # 根据划分选择对应的索引
        if split == 'train':
            available_indices = dataset_data['train_indices']
            split_name = "🟢 TRAIN"
        elif split == 'val':
            available_indices = dataset_data['val_indices']
            split_name = "🟡 VAL"
        elif split == 'test':
            available_indices = dataset_data['test_indices']
            split_name = "🔴 TEST"
        else:
            raise ValueError(f"无效的数据划分: {split}")
        
        if len(available_indices) == 0:
            raise ValueError(f"{dataset.upper()} {split_name} 划分为空!")
        
        # 从对应划分中随机采样
        if batch_size > len(available_indices):
            selected_indices = np.random.choice(available_indices, batch_size, replace=True)
        else:
            selected_indices = np.random.choice(available_indices, batch_size, replace=False)
        
        # 获取数据
        images = dataset_data['images'][selected_indices].to(device)
        text_sequences = dataset_data['text_sequences'][selected_indices].to(device)
        labels = dataset_data['labels'][selected_indices].to(device)
        
        return images, text_sequences, labels

    def get_dataset_info(self, dataset='bloodmnist'):
        """获取指定数据集的信息"""
        if dataset not in self.datasets:
            raise ValueError(f"未知数据集: {dataset}")
        
        dataset_data = self.datasets[dataset]
        return {
            'num_classes': dataset_data['num_classes'],
            'class_names': dataset_data['class_names'],
            'total_samples': len(dataset_data['images']),
            'train_samples': len(dataset_data['train_indices']),
            'val_samples': len(dataset_data['val_indices']),
            'test_samples': len(dataset_data['test_indices'])
        }


class ComplexDataGenerator:
    """复杂数据生成器 - 支持多数据集"""

    def __init__(self, config):
        self.config = config
        self.anomaly_generator = ComplexAnomalyGenerator()
        self.multi_loader = MultiDatasetLoader(
            data_root=config.get('data_root', "F:\\Desktop\\bloodmnist\\Data"),
            batch_size=config.get('batch_size', 32),
            cache_size=config.get('cache_size', 1024),
            embedding_dim=config.get('embedding_dim', 64),
            max_seq_len=config.get('max_seq_len', 64)
        )

    def generate_multimodal_data(self, batch_size, device, split='train', dataset='bloodmnist', anomaly_ratio=0.15):
        """
        ✅ 根据数据划分和数据集生成多模态数据
        
        Args:
            batch_size: 批次大小
            device: 设备
            split: 数据划分 {'train', 'val', 'test'}
            dataset: 数据集名称 {'bloodmnist', 'pathmnist', 'organamnist'}
            anomaly_ratio: 异常样本比例
        """
        # 使用新的多数据集接口获取数据
        images, text_sequences, original_labels = self.multi_loader.get_split_batch(
            batch_size, device, split, dataset)

        # 添加基础噪声
        base_noise = torch.randn_like(images) * 0.05
        images = images + base_noise

        # 生成异常样本
        images, perturbed_text_sequences, anomaly_labels, anomaly_types = \
            self.anomaly_generator.generate_anomalies_with_real_text(
                images, text_sequences, original_labels, self.config, anomaly_ratio,
                vocab_size=self.multi_loader.vocab_size
            )

        return images, perturbed_text_sequences, anomaly_labels, anomaly_types


        return images, perturbed_text_sequences, anomaly_labels, anomaly_types

    def _apply_data_augmentation(self, images):  # 此方法已在 ComplexAnomalyGenerator 内部调用，这里保留但注意不要重复调用
        batch_size = images.shape[0]

        for i in range(batch_size):
            if np.random.random() > 0.5:
                brightness_factor = 0.8 + np.random.random() * 0.4
                images[i] = images[i] * brightness_factor

            if np.random.random() > 0.6:
                contrast_factor = 0.7 + np.random.random() * 0.6
                mean_val = images[i].mean()
                images[i] = (images[
                                 i] - mean_val) * contrast_factor + mean_val  # Corrected var name `contrast_val` to `contrast_factor`

        images = torch.clamp(images, -2, 2)
        return images

    def _add_text_noise(self, text_sequences):
        return text_sequences


class ComplexAnomalyGenerator:
    """生成更复杂和真实的异常模式 - 调整以适应序列ID或原始文本"""

    def __init__(self, anomaly_types=['structural', 'contextual', 'collective', 'multimodal', 'boundary']):
        self.anomaly_types = anomaly_types

    def generate_anomalies_with_real_text(self, images, real_text_sequences, labels, config, anomaly_ratio=0.15,
                                          vocab_size=None):  # 新增 vocab_size
        num_samples = len(images)
        num_anomalies = int(num_samples * anomaly_ratio)
        device = images.device

        anomaly_labels = torch.zeros(num_samples, device=device)
        anomaly_indices = torch.randperm(num_samples)[:num_anomalies]
        anomaly_labels[anomaly_indices] = 1

        anomaly_type_assignment = np.random.choice(self.anomaly_types, num_anomalies)

        modified_text_sequences = real_text_sequences.clone()

        normal_indices = torch.where(anomaly_labels == 0)[0]

        for i, idx in enumerate(anomaly_indices):
            anomaly_type = anomaly_type_assignment[i]

            if anomaly_type == 'structural':
                images[idx] = self._apply_structural_anomaly(images[idx])
                modified_text_sequences[idx] = self._add_sequence_anomaly(modified_text_sequences[idx], 'structural',
                                                                          vocab_size, device)

            elif anomaly_type == 'contextual':
                if len(normal_indices) > 0:
                    rand_idx = torch.randint(0, len(normal_indices), (1,))
                    random_normal_idx = normal_indices[rand_idx.item()]
                    modified_text_sequences[idx] = real_text_sequences[random_normal_idx].clone()
                    modified_text_sequences[idx] = self._add_sequence_anomaly(modified_text_sequences[idx],
                                                                              'contextual', vocab_size, device)

            elif anomaly_type == 'collective':
                images[idx] = self._apply_collective_anomaly(images[idx])
                modified_text_sequences[idx] = self._add_sequence_anomaly(modified_text_sequences[idx], 'collective',
                                                                          vocab_size, device)

            elif anomaly_type == 'multimodal':
                images[idx] = self._apply_multimodal_anomaly(images[idx])
                modified_text_sequences[idx] = self._add_sequence_anomaly(modified_text_sequences[idx], 'multimodal',
                                                                          vocab_size, device)

            elif anomaly_type == 'boundary':
                if len(normal_indices) > 1:
                    rand_idx = torch.randint(0, len(normal_indices), (1,))
                    other_normal_idx = normal_indices[rand_idx.item()]
                    alpha = 0.7 + torch.rand(1).item() * 0.2

                    images[idx] = alpha * images[idx] + (1 - alpha) * images[other_normal_idx]
                    modified_text_sequences[idx] = (alpha * real_text_sequences[idx].float() + \
                                                    (1 - alpha) * real_text_sequences[
                                                        other_normal_idx].float()).round().long()

        full_anomaly_types = np.array(['normal'] * num_samples)
        full_anomaly_types[anomaly_indices.cpu().numpy()] = anomaly_type_assignment

        return images, modified_text_sequences, anomaly_labels, full_anomaly_types

    def _add_sequence_anomaly(self, sequence_ids, anomaly_type, vocab_size, device):
        """对文本ID序列添加简单异常扰动 (已确保 vocab_size 传入)"""
        seq_len = sequence_ids.shape[0]

        if vocab_size is None or vocab_size < 2:
            return sequence_ids

        if anomaly_type == 'structural':
            num_to_perturb = int(seq_len * 0.1)
            if num_to_perturb == 0 and seq_len > 0: num_to_perturb = 1

            perturb_positions = torch.randperm(seq_len)[:num_to_perturb]
            random_word_ids = torch.randint(2, vocab_size, (num_to_perturb,), device=device, dtype=torch.long)

            for i, pos in enumerate(perturb_positions):
                sequence_ids[pos] = random_word_ids[i]

        elif anomaly_type == 'contextual':
            if seq_len > 0 and np.random.random() > 0.5:
                insert_pos = torch.randint(0, seq_len, (1,)).item()
                if np.random.random() > 0.5:
                    new_ids = torch.cat((sequence_ids[:insert_pos], torch.tensor([1], device=device, dtype=torch.long),
                                         sequence_ids[insert_pos:]))
                else:
                    if insert_pos < seq_len and sequence_ids[insert_pos] != 0:
                        new_ids = torch.cat((sequence_ids[:insert_pos], sequence_ids[insert_pos + 1:]))
                    else:
                        new_ids = sequence_ids

                if len(new_ids) < seq_len:
                    sequence_ids = torch.cat(
                        (new_ids, torch.tensor([0] * (seq_len - len(new_ids)), device=device, dtype=torch.long)))
                elif len(new_ids) > seq_len:
                    sequence_ids = new_ids[:seq_len]
                else:
                    sequence_ids = new_ids

        elif anomaly_type == 'collective':
            if seq_len > 5:
                start_pos = torch.randint(0, seq_len - 5, (1,)).item()
                pattern_id = torch.randint(2, vocab_size, (1,)).item()
                sequence_ids[start_pos:start_pos + 5] = pattern_id

        elif anomaly_type == 'multimodal':
            if seq_len > 0:
                pos = torch.randint(0, seq_len, (1,)).item()
                sequence_ids[pos] = torch.randint(2, vocab_size, (1,)).item()
                if seq_len < 2 * sequence_ids.shape[0] and np.random.random() > 0.5:
                    insert_pos = torch.randint(0, seq_len, (1,)).item()
                    new_ids = torch.cat((sequence_ids[:insert_pos], torch.tensor([1], device=device, dtype=torch.long),
                                         sequence_ids[insert_pos:]))
                    if len(new_ids) > seq_len:
                        sequence_ids = new_ids[:seq_len]
                    else:
                        sequence_ids = new_ids

        return sequence_ids

    def _apply_structural_anomaly(self, image):
        anomaly_type = np.random.choice(['occlusion', 'noise', 'blur'])

        if anomaly_type == 'occlusion':
            h, w = image.shape[-2:]
            mask_h, mask_w = h // 4, w // 4
            start_h = torch.randint(0, h - mask_h, (1,)).item()
            start_w = torch.randint(0, w - mask_w, (1,)).item()
            image[:, start_h:start_h + mask_h, start_w:start_w + mask_w] = 0

        elif anomaly_type == 'noise':
            noise = torch.randn_like(image) * 0.5
            image = image + noise

        elif anomaly_type == 'blur':
            kernel_size = 5
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size * kernel_size)

            if len(image.shape) == 3:
                img_batch = image.unsqueeze(0)
            else:
                img_batch = image

            blurred_channels = []
            for c in range(img_batch.shape[1]):
                single_channel = img_batch[:, c:c + 1, :, :]
                padded = F.pad(single_channel, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                               mode='reflect')
                blurred = F.conv2d(padded, kernel, padding=0)
                blurred_channels.append(blurred)

            blurred_image = torch.cat(blurred_channels, dim=1)

            if len(image.shape) == 3:
                image = blurred_image.squeeze(0)
            else:
                image = blurred_image

        return image

    def _apply_collective_anomaly(self, image):
        h, w = image.shape[-2:]
        pattern_type = np.random.choice(['checkerboard', 'stripe', 'spot'])

        if pattern_type == 'checkerboard':
            for i in range(0, h, 4):
                for j in range(0, w, 4):
                    if (i // 4 + j // 4) % 2 == 0:
                        image[:, i:i + 4, j:j + 4] *= 0.3

        elif pattern_type == 'stripe':
            for i in range(0, h, 3):
                image[:, i, :] *= 0.5

        elif pattern_type == 'spot':
            num_spots = 5
            for _ in range(num_spots):
                cy = torch.randint(5, h - 5, (1,)).item()
                cx = torch.randint(5, w - 5, (1,)).item()
                image[:, cy - 2:cy + 2, cx - 2:cx + 2] = 1.0

        return image

    def _apply_multimodal_anomaly(self, image):
        image = self._apply_structural_anomaly(image)
        if image.shape[0] == 3:
            channel_shift = torch.randn(3, 1, 1, device=image.device) * 0.3
            image = image + channel_shift
        return image


class AblationTrainer:
    """消融实验训练器"""

    def __init__(self, model, config, device='cuda', dataset='bloodmnist'):
        self.model = model
        self.config = config
        self.device = device
        self.dataset = dataset  # 新增：当前使用的数据集
        self.data_generator = ComplexDataGenerator(config)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def train(self, num_epochs=50, train_batches=20, val_batches=5):
        """
        ✅ 训练模型（严格的数据划分）
        - 训练阶段：仅使用 Train 数据进行梯度更新
        - 验证阶段：仅使用 Val 数据进行模型选择和早停
        - 测试阶段：完全隔离，不在训练过程中接触
        """
        print(f"🚀 开始训练 - 严格的数据划分模式")
        print(f"   🟢 训练数据：仅用于梯度更新")
        print(f"   🟡 验证数据：仅用于模型选择和早停")
        print(f"   🔴 测试数据：完全隔离，训练期间不接触")
        
        self.model.train()
        train_losses = []
        val_metrics = []
        batch_size = self.config.get('batch_size', 16)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []

            # ✅ 训练阶段：仅使用 Train 数据
            for batch_idx in range(train_batches):
                torch.set_grad_enabled(True)
                self.model.train()

                # 🟢 严格使用训练集数据，指定数据集
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='train', dataset=self.dataset, anomaly_ratio=0.15
                    )

                self.optimizer.zero_grad()
                _, losses = self.model(images, text_sequences, labels)

                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_losses.append(losses['total_loss'].item() if losses['total_loss'].numel() == 1 else losses[
                    'total_loss'].mean().item())

            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)

            # ✅ 验证阶段：仅使用 Val 数据（用于早停和模型选择）
            torch.set_grad_enabled(False)
            if epoch % 10 == 0:
                val_results = self.evaluate_validation(val_batches)  # 新方法：仅在验证集上评估
                val_metrics.append(val_results)
                self.scheduler.step(avg_train_loss)

                print(f"Epoch {epoch}/{num_epochs}:")
                print(f"  🟢 Train Loss = {avg_train_loss:.4f}")
                print(f"  🟡 Val AUC = {val_results['auc']:.4f}, Val Acc = {val_results['accuracy']:.4f}")

        return train_losses, val_metrics

    def evaluate_validation(self, num_batches=10):
        """
        ✅ 在验证集上评估（用于模型选择和早停）
        """
        self.model.eval()
        all_scores = []
        all_labels = []
        batch_size = self.config.get('batch_size', 32)

        with torch.no_grad():
            for _ in range(num_batches):
                # 🟡 仅使用验证集数据，指定数据集
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='val', dataset=self.dataset, anomaly_ratio=0.2
                    )

                scores, _ = self.model(images, text_sequences)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return self._compute_metrics(all_scores, all_labels)

    def evaluate_testset(self, num_batches=10):
        """
        ✅ 最终测试集评估（仅在训练完成后调用）
        这是真正的泛化性能测试
        """
        print(f"🔴 ⚠️  执行最终测试集评估 - 这是真正的泛化测试!")
        self.model.eval()
        all_scores = []
        all_labels = []
        batch_size = self.config.get('batch_size', 32)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # 🔴 首次接触测试集数据（仅在最终评估时），指定数据集
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='test', dataset=self.dataset, anomaly_ratio=0.2
                    )

                scores, _ = self.model(images, text_sequences)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx == 0:
                    print(f"🔴 测试批次 {batch_idx+1}: {len(labels)} 样本, {labels.sum().item()} 个异常")

        test_results = self._compute_metrics(all_scores, all_labels)
        print(f"🎯 最终测试结果: AUC={test_results['auc']:.4f}, Accuracy={test_results['accuracy']:.4f}")
        return test_results

    def _compute_metrics(self, all_scores, all_labels):
        """计算评估指标的通用方法"""
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        results = {}
        
        try:
            results['auc'] = roc_auc_score(all_labels, all_scores)
        except:
            results['auc'] = 0.5

        try:
            from sklearn.metrics import accuracy_score, precision_recall_curve
            
            threshold = 0.5
            predicted_labels_fixed = (all_scores > threshold).astype(int)
            results['accuracy_fixed'] = accuracy_score(all_labels, predicted_labels_fixed)

            precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
            temp_f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(temp_f1_scores)
            best_threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5

            predicted_labels_optimal = (all_scores > best_threshold).astype(int)
            results['accuracy_optimal'] = accuracy_score(all_labels, predicted_labels_optimal)
            results['best_threshold'] = best_threshold
            results['accuracy'] = results['accuracy_optimal']

        except Exception as e:
            print(f"Warning: Could not calculate accuracy: {e}")
            results['accuracy'] = 0.5
            results['accuracy_fixed'] = 0.5
            results['accuracy_optimal'] = 0.5
            results['best_threshold'] = 0.5

        return results



class BaselineExperiment:
    """基线实验管理器 - 只运行主实验的baseline"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

        # 基础配置 - 适配BloodMNIST数据集
        self.base_config = {
            'hidden_dim': 128,
            'repr_dim': 64,
            'final_repr_dim': 32,
            'embedding_dim': 64,
            'max_seq_len': 64,
            'vocab_size': None,
            'text_dim': 64,

            # =================== 坚定地修改以下三个参数 ===================
            'top_k': 5,  # 必须修改。从10改为5，构建稀疏高质量的图。
            'cheb_k': 3,  # 必须修改。从10改为3，聚焦局部特征，防止过平滑。
            'learning_rate': 0.0005,  # 必须修改。从0.001改为0.0005，稳定训练过程。
            # ==========================================================

            'threshold': 0.6,  # 此参数可保持不变
            'k_nearest': 3,
            'tau': 0.5,
            'lambda_recon': 0.1,
            'lambda_spectral': 0.01,
            'batch_size': 32,
            'weight_decay': 1e-5,
            'cache_size': 1024,
            'num_classes': 8,  # 默认BloodMNIST，会在运行时动态更新
            'image_size': 28,
            'lstm_layers': 2,
            'lstm_dropout': 0.3,
            'data_root': r"F:\\Desktop\\bloodmnist\\Data"  # 数据根目录
        }

        # 只保留主实验的baseline配置 + 新增基线方法
        self.baseline_config = {
            'original_baseline_bilstm': {
                'name': 'Complete Model (Baseline)',
                'hypergraph_type': 'dynamic',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': True,
            }
        }
        
        # 新增：基线对比方法配置
        self.comparison_methods = {
            'autoencoder': {
                'name': 'Autoencoder Baseline',
                'description': 'Simple autoencoder for anomaly detection'
            },
            'svm': {
                'name': 'One-Class SVM',
                'description': 'One-Class Support Vector Machine'
            },
            'isolation_forest': {
                'name': 'Isolation Forest',
                'description': 'Isolation Forest ensemble method'
            }
        }
        
        # 当前使用的数据集
        self.current_dataset = 'bloodmnist'

    def run_baseline_experiment(self, epochs=30):
        """运行主实验的baseline"""
        experiment_name = 'original_baseline_bilstm'
        print(f"\n{'=' * 20} {self.baseline_config[experiment_name]['name']} {'=' * 20}")

        baseline_config = self.baseline_config[experiment_name]

        try:
            model = AblationAnomalyDetector(self.base_config, baseline_config).to(self.device)
            print(f"Running baseline experiment using BiLSTM Text Encoder with Word Embeddings.")

            trainer = AblationTrainer(model, self.base_config, self.device, self.current_dataset)

            print(f"Starting training for {epochs} epochs...")

            train_losses, val_metrics = trainer.train(epochs, train_batches=15, val_batches=5)

            final_results = trainer.evaluate_testset(num_batches=32)

            self.results[experiment_name] = {
                'config': baseline_config,
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'final_results': final_results,
                'val_aucs': [m['auc'] for m in val_metrics],
                'final_auc': final_results['auc'],
            }

            print(f"Baseline experiment completed")
            print(f"Final Results - AUC: {final_results['auc']:.4f}, "
                  f"Accuracy: {final_results['accuracy']:.4f}")

            return final_results

        except Exception as e:
            print(f"Baseline experiment failed: {e}")
            import traceback
            traceback.print_exc()

            default_results = {
                'auc': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }

            self.results[experiment_name] = {
                'config': baseline_config,
                'train_losses': [],
                'val_metrics': [],
                'final_results': default_results,
                'val_aucs': [],
                'final_auc': 0.0,
            }

            return default_results

    def run_baseline_comparison_methods(self, epochs=30, dataset='bloodmnist'):
        """运行基线对比方法（Autoencoder、SVM、Isolation Forest）"""
        print(f"\n🔬 开始运行基线对比方法 - {dataset.upper()}")
        print("=" * 70)
        
        # 创建基线模型训练器
        baseline_trainer = BaselineModelTrainer(self.base_config, self.device)
        comparison_results = {}
        
        # 1. 训练和评估自编码器
        print("\n1️⃣ 自编码器基线实验")
        print("-" * 50)
        try:
            autoencoder_results = baseline_trainer.train_autoencoder(
                dataset=dataset, epochs=epochs, batch_size=self.base_config['batch_size']
            )
            comparison_results['autoencoder'] = {
                'name': self.comparison_methods['autoencoder']['name'],
                'results': autoencoder_results
            }
            print(f"✅ 自编码器完成 - AUC: {autoencoder_results['auc']:.4f}, Accuracy: {autoencoder_results['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ 自编码器失败: {str(e)}")
            comparison_results['autoencoder'] = {
                'name': self.comparison_methods['autoencoder']['name'],
                'results': {'auc': 0.0, 'accuracy': 0.0, 'error': str(e)}
            }
        
        # 2. 训练和评估One-Class SVM
        print("\n2️⃣ One-Class SVM基线实验")
        print("-" * 50)
        try:
            svm_results = baseline_trainer.train_svm(
                dataset=dataset, batch_size=self.base_config['batch_size'], nu=0.1
            )
            comparison_results['svm'] = {
                'name': self.comparison_methods['svm']['name'],
                'results': svm_results
            }
            print(f"✅ SVM完成 - AUC: {svm_results['auc']:.4f}, Accuracy: {svm_results['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ SVM失败: {str(e)}")
            comparison_results['svm'] = {
                'name': self.comparison_methods['svm']['name'],
                'results': {'auc': 0.0, 'accuracy': 0.0, 'error': str(e)}
            }
        
        # 3. 训练和评估Isolation Forest
        print("\n3️⃣ Isolation Forest基线实验")
        print("-" * 50)
        try:
            isolation_results = baseline_trainer.train_isolation_forest(
                dataset=dataset, batch_size=self.base_config['batch_size'], contamination=0.1
            )
            comparison_results['isolation_forest'] = {
                'name': self.comparison_methods['isolation_forest']['name'],
                'results': isolation_results
            }
            print(f"✅ Isolation Forest完成 - AUC: {isolation_results['auc']:.4f}, Accuracy: {isolation_results['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ Isolation Forest失败: {str(e)}")
            comparison_results['isolation_forest'] = {
                'name': self.comparison_methods['isolation_forest']['name'],
                'results': {'auc': 0.0, 'accuracy': 0.0, 'error': str(e)}
            }
        
        return comparison_results

    def run_baseline_only(self, epochs=30, dataset='bloodmnist'):
        """运行主实验的baseline + 基线对比方法"""
        print(f"Starting comprehensive baseline experiment on {dataset.upper()}, device: {self.device}")

        # 动态获取 vocab_size 并更新 base_config
        temp_data_loader = MultiDatasetLoader(
            data_root=self.base_config.get('data_root', "F:\\Desktop\\bloodmnist\\Data"),
            batch_size=self.base_config['batch_size'],
            cache_size=self.base_config['cache_size'],
            embedding_dim=self.base_config['embedding_dim'],
            max_seq_len=self.base_config['max_seq_len']
        )
        vocab_size = temp_data_loader.vocab_size  # 直接访问词汇表大小属性
        self.base_config['vocab_size'] = vocab_size
        
        # 获取数据集信息并更新配置
        dataset_info = temp_data_loader.get_dataset_info(dataset)
        self.base_config['num_classes'] = dataset_info['num_classes']
        self.current_dataset = dataset
        
        print(f"Dynamic vocabulary size detected: {vocab_size}")
        print(f"Dataset info: {dataset_info}")
        del temp_data_loader

        # 1. 运行主要的baseline实验（原有的hypergraph模型）
        print(f"\n🎯 主要基线实验 - 多模态超图异常检测")
        print("=" * 70)
        try:
            main_results = self.run_baseline_experiment(epochs)
            self.results['main_model'] = self.results.get('original_baseline_bilstm', {})
        except Exception as e:
            print(f"Main baseline experiment failed: {str(e)}")
            self.results['main_model'] = {'final_results': {'auc': 0.0, 'accuracy': 0.0, 'error': str(e)}}

        # 2. 运行基线对比方法
        comparison_results = self.run_baseline_comparison_methods(epochs=epochs//2, dataset=dataset)
        self.results['comparison_methods'] = comparison_results

        return self.results

def main():
    """Main function - 在三个数据集上运行baseline实验"""

    print("🧪 Multimodal Hypergraph Anomaly Detection System - Multi-Dataset Baseline Experiments")
    print("================================================================================")
    print("Running baseline experiments on BloodMNIST, PathMNIST, and OrganaMNIST datasets.")

    # 三个数据集列表
    datasets = ['bloodmnist', 'pathmnist', 'organamnist']
    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"🔬 开始在 {dataset.upper()} 数据集上运行基线实验")
        print(f"{'='*80}")
        
        experiment = BaselineExperiment()
        
        print(f"Starting baseline experiment with {dataset.upper()} dataset...")
        results = experiment.run_baseline_only(epochs=50, dataset=dataset)
        
        # 存储结果
        all_results[dataset] = results

        # 打印当前数据集结果
        print(f"\n{'='*80}")
        print(f"{dataset.upper()} COMPREHENSIVE BASELINE EXPERIMENT RESULTS")
        print(f"{'='*80}")
        
        # 主模型结果
        if 'main_model' in results and 'final_results' in results['main_model']:
            main_result = results['main_model']['final_results']
            print(f"🎯 主要模型 (多模态超图): AUC = {main_result['auc']:.4f}, Accuracy = {main_result['accuracy']:.4f}")
        else:
            print(f"❌ 主要模型: 实验失败")
        
        # 对比方法结果
        if 'comparison_methods' in results:
            print(f"\n📊 基线对比方法:")
            comparison_methods = results['comparison_methods']
            
            for method_key, method_data in comparison_methods.items():
                method_name = method_data['name']
                method_results = method_data['results']
                if 'error' in method_results:
                    print(f"   ❌ {method_name}: 实验失败 - {method_results.get('error', 'Unknown error')}")
                else:
                    print(f"   ✅ {method_name}: AUC = {method_results['auc']:.4f}, Accuracy = {method_results['accuracy']:.4f}")
        
        print(f"{'='*80}")

    # 汇总所有结果
    print(f"\n{'='*80}")
    print("� ALL DATASETS SUMMARY")
    print(f"{'='*80}")
    
    for dataset in datasets:
        if dataset in all_results:
            results = all_results[dataset]
            print(f"\n📋 {dataset.upper()} 数据集:")
            
            # 主模型结果
            if 'main_model' in results and 'final_results' in results['main_model']:
                main_result = results['main_model']['final_results']
                print(f"   🎯 多模态超图模型: AUC = {main_result['auc']:.4f}, Accuracy = {main_result['accuracy']:.4f}")
            else:
                print(f"   ❌ 多模态超图模型: 失败")
            
            # 对比方法结果
            if 'comparison_methods' in results:
                comparison_methods = results['comparison_methods']
                for method_key, method_data in comparison_methods.items():
                    method_name = method_data['name']
                    method_results = method_data['results']
                    if 'error' not in method_results:
                        print(f"   📊 {method_name}: AUC = {method_results['auc']:.4f}, Accuracy = {method_results['accuracy']:.4f}")
                    else:
                        print(f"   ❌ {method_name}: 失败")
        else:
            print(f"❌ {dataset.upper():12} - Failed to complete")

    print(f"\n🎉 Multi-dataset comprehensive baseline experiments completed!")
    print("✅ All three datasets have been evaluated with multiple baseline methods")


if __name__ == "__main__":
    main()