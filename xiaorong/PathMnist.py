import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import os
import sys
import json
import pickle
import warnings
import traceback
from datetime import datetime
import jieba  # 新增：导入 jieba 用于中文分词
import random

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入自定义模块
from models.modules.hyper_generator import HybridHyperedgeGenerator
from models.modules.wavelet_cheb_conv import WaveletChebConv
from models.modules.pruning_regularizer import SpectralCutRegularizer

warnings.filterwarnings('ignore')

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


class FixedKNNHypergraphGenerator:
    """固定k-NN超图生成器（消融实验1）"""

    def __init__(self, k_nearest=10):
        self.k_nearest = k_nearest

    def __call__(self, features_list):
        batch_size = features_list[0].shape[0]
        device = features_list[0].device

        combined_features = torch.cat(features_list, dim=1)
        combined_features = F.normalize(combined_features, dim=1)

        similarity = torch.mm(combined_features, combined_features.t())

        actual_k = min(self.k_nearest, batch_size - 1)
        if actual_k <= 0:
            actual_k = 1

        _, top_k_indices = torch.topk(similarity, actual_k + 1, dim=1)
        top_k_indices = top_k_indices[:, 1:]

        num_hyperedges = batch_size
        H = torch.zeros(batch_size, num_hyperedges, device=device)

        for i in range(batch_size):
            H[i, i] = 1.0
            for j, neighbor_idx in enumerate(top_k_indices[i]):
                if j < actual_k and neighbor_idx < batch_size:
                    H[neighbor_idx, i] = 1.0

        for i in range(num_hyperedges):
            if H[:, i].sum() == 0:
                H[i, i] = 1.0

        edge_weights = torch.ones(num_hyperedges, device=device)

        return H, edge_weights


class SimpleGraphConv(nn.Module):
    """简单图卷积层（消融实验2：替代谱小波滤波）"""

    def __init__(self, in_dim, out_dim):
        super(SimpleGraphConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, L):
        x_transformed = self.linear(x)
        return torch.mm(L, x_transformed)


class SimilarityOnlyHyperedgeGenerator:
    """仅使用相似度先验的超边生成器（消融实验4a）"""

    def __init__(self, threshold=0.6, top_k=8):
        self.threshold = threshold
        self.top_k = top_k

    def __call__(self, features_list):
        batch_size = features_list[0].shape[0]
        device = features_list[0].device

        combined_features = torch.cat(features_list, dim=1)
        combined_features = F.normalize(combined_features, dim=1)

        similarity = torch.mm(combined_features, combined_features.t())

        actual_top_k = min(self.top_k, batch_size)
        _, top_k_indices = torch.topk(similarity, actual_top_k, dim=1)

        hyperedges = []
        edge_weights = []

        for i in range(batch_size):
            neighbors = top_k_indices[i]
            if len(neighbors) > 0:
                hyperedges.append(neighbors.cpu().numpy())
                neighbor_similarities = similarity[i, neighbors]
                if neighbor_similarities.numel() > 0:
                    edge_weights.append(neighbor_similarities.mean().item())
                else:
                    edge_weights.append(1.0)

        if len(hyperedges) == 0:
            hyperedges.append(np.arange(batch_size))
            edge_weights.append(1.0)

        num_hyperedges = len(hyperedges)
        H = torch.zeros(batch_size, num_hyperedges, device=device)

        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                if node < batch_size:
                    H[node, edge_idx] = 1.0

        edge_weights = torch.tensor(edge_weights, device=device, dtype=torch.float32)

        return H, edge_weights


class AttentionOnlyHyperedgeGenerator(nn.Module):
    """仅使用模态注意力打分的超边生成器（消融实验4b）"""

    def __init__(self, input_dims, hidden_dim=128, num_heads=4, top_k=8):
        super(AttentionOnlyHyperedgeGenerator, self).__init__()
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for dim in input_dims
        ])

        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.edge_weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        device = features_list[0].device

        projected_features = []
        for features, projection in zip(features_list, self.modality_projections):
            proj_feat = projection(features)
            projected_features.append(proj_feat)

        modal_sequence = torch.stack(projected_features, dim=1)
        attended_features, _ = self.cross_modal_attention(
            modal_sequence, modal_sequence, modal_sequence
        )

        fused_features = attended_features.mean(dim=1)

        attention_similarity = torch.mm(fused_features, fused_features.t())
        attention_similarity = F.softmax(attention_similarity, dim=1)

        actual_top_k = min(self.top_k, batch_size)
        _, top_k_indices = torch.topk(attention_similarity, actual_top_k, dim=1)

        num_hyperedges = batch_size
        H = torch.zeros(batch_size, num_hyperedges, device=device)

        for i in range(batch_size):
            H[i, i] = 1.0
            for j, neighbor_idx in enumerate(top_k_indices[i]):
                if j < actual_top_k and neighbor_idx < batch_size:
                    H[neighbor_idx, i] = attention_similarity[i, neighbor_idx]

        for i in range(num_hyperedges):
            if H[:, i].sum() == 0:
                H[i, i] = 1.0

        edge_weights = self.edge_weight_predictor(fused_features).squeeze()

        if edge_weights.dim() == 0:
            edge_weights = edge_weights.unsqueeze(0).expand(num_hyperedges)
        elif edge_weights.shape[0] != num_hyperedges:
            edge_weights = torch.ones(num_hyperedges, device=device)

        edge_weights = torch.clamp(edge_weights, min=1e-8)

        return H, edge_weights


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
        elif ablation_config['hypergraph_type'] == 'fixed_knn':
            self.hypergraph_generator = FixedKNNHypergraphGenerator(
                k_nearest=config.get('k_nearest', 5)
            )
        elif ablation_config['hypergraph_type'] == 'similarity_only':
            self.hypergraph_generator = SimilarityOnlyHyperedgeGenerator(
                threshold=config.get('threshold', 0.6),
                top_k=config.get('top_k', 8)
            )
        elif ablation_config['hypergraph_type'] == 'attention_only':
            self.hypergraph_generator = AttentionOnlyHyperedgeGenerator(
                input_dims=input_dims,
                hidden_dim=config['hidden_dim'],
                top_k=config.get('top_k', 8)
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
        else:  # 'simple_graph'
            self.conv1 = SimpleGraphConv(
                in_dim=conv_input_dim,
                out_dim=config['hidden_dim']
            )
            self.conv2 = SimpleGraphConv(
                in_dim=config['hidden_dim'],
                out_dim=config['repr_dim']
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


# 修改后的 MultimodalPathMNISTDataLoader - 处理原始文本
class MultimodalPathMNISTDataLoader:
    """多模态PathMNIST数据加载器 - 从指定路径加载图像和文本数据，支持静态数据划分"""

    def __init__(self, data_path, batch_size=128, cache_size=1024, embedding_dim=64, max_seq_len=64,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_path = data_path
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

        # 存储分割后的数据缓存
        self.images_cache = None
        self.text_sequences_cache = None
        self.labels_cache = None
        
        # 存储分割索引
        self.train_indices = None
        self.val_indices = None 
        self.test_indices = None
        
        # 词汇表相关
        self.vocab = {}
        self.idx_to_word = {}
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.vocab_size = 2

        self.class_names = [
            "adipose", "background", "debris", "lymphocytes",
            "mucus", "smooth_muscle", "normal_colon_mucosa", "cancer_associated_stroma",
            "colorectal_adenocarcinoma_epithelium"
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self._initialize_loader()

    def _initialize_loader(self):
        """初始化数据加载器，包括文本分词、序列化和静态数据划分"""

        print(f"Loading multimodal PathMNIST data from {self.data_path}")

        images_path = os.path.join(self.data_path, "pathmnist_images.pkl")
        with open(images_path, 'rb') as f:
            image_data = pickle.load(f)

        text_path = os.path.join(self.data_path, "pathmnist_text_descriptions.json")
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        # 加载原始图像和标签
        original_images = torch.from_numpy(np.array(image_data['images'])).float()
        original_labels = torch.from_numpy(np.array(image_data['labels'])).long()
        print(f"Successfully loaded images and labels from pickle file")

        # 处理文本数据
        raw_text_descriptions = []
        samples_data = text_data.get('data', text_data.get('samples', []))

        for sample in samples_data:
            description = sample.get('text_description', sample.get('description', ''))
            raw_text_descriptions.append(description)

        print(f"Found {len(raw_text_descriptions)} raw text descriptions")
        print(f"Sample raw text: {raw_text_descriptions[0] if raw_text_descriptions else 'No text found'}")

        # 构建词汇表
        all_tokens = []
        for desc in raw_text_descriptions:
            tokens = [word.strip('.,!?;:"\'').lower() for word in jieba.lcut(desc) if word.strip('.,!?;:"\'').isalnum()]
            all_tokens.extend(tokens)

        unique_tokens = sorted(list(set(all_tokens)))
        for word in unique_tokens:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.vocab_size += 1
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        print(f"Vocabulary size: {self.vocab_size}")

        # 将文本转换为序列
        text_sequences = []
        for desc in raw_text_descriptions:
            tokens = [word.strip('.,!?;:"\'').lower() for word in jieba.lcut(desc) if word.strip('.,!?;:"\'').isalnum()]
            indexed_tokens = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]

            if len(indexed_tokens) < self.max_seq_len:
                padded_sequence = indexed_tokens + [self.word_to_idx['<pad>']] * (
                            self.max_seq_len - len(indexed_tokens))
            else:
                padded_sequence = indexed_tokens[:self.max_seq_len]
            text_sequences.append(padded_sequence)

        original_text_sequences = torch.tensor(text_sequences, dtype=torch.long)

        # 确保所有数据长度一致
        min_samples = min(len(original_images), len(original_text_sequences), len(original_labels))
        original_images = original_images[:min_samples]
        original_text_sequences = original_text_sequences[:min_samples]
        original_labels = original_labels[:min_samples]

        print(f"\n🔄 执行静态数据划分 (Train: {self.train_ratio:.1%} | Val: {self.val_ratio:.1%} | Test: {self.test_ratio:.1%})")
        
        total_samples = min_samples
        indices = np.random.permutation(total_samples)
        
        # 计算划分点
        train_end = int(total_samples * self.train_ratio)
        val_end = train_end + int(total_samples * self.val_ratio)
        
        # 静态索引划分
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        print(f"📊 数据划分结果:")
        print(f"   🟢 Train: {len(self.train_indices)} samples (indices: {self.train_indices[:5]}...)")
        print(f"   🟡 Val: {len(self.val_indices)} samples (indices: {self.val_indices[:5]}...)")
        print(f"   🔴 Test: {len(self.test_indices)} samples (indices: {self.test_indices[:5]}...)")

        # 图像预处理
        if original_images.max() > 1.0:
            original_images = original_images / 255.0
        original_images = (original_images - 0.5) / 0.5

        # 存储完整数据（但训练时不会接触测试集）
        self.images_cache = original_images
        self.text_sequences_cache = original_text_sequences  
        self.labels_cache = original_labels

        print(f"✅ 数据加载完成:")
        print(f"   Image shape: {self.images_cache.shape}")
        print(f"   Image range: {self.images_cache.min():.3f} to {self.images_cache.max():.3f}")
        print(f"   Text sequence shape: {self.text_sequences_cache.shape}")
        print(f"   Number of classes: {len(self.class_names)}")
        print(f"   ⚠️  测试集在训练阶段完全隔离，仅在最终评估时使用")

    def get_split_batch(self, batch_size, device, split='train'):
        """
        ✅ 新接口：按数据划分获取批次数据（确保测试集隔离）
        
        Args:
            batch_size: 批次大小
            device: 设备
            split: 数据划分 {'train', 'val', 'test'}
            
        Returns:
            images, text_sequences, labels
        """
        
        # 根据划分选择对应的索引
        if split == 'train':
            available_indices = self.train_indices
            split_name = "🟢 TRAIN"
        elif split == 'val':
            available_indices = self.val_indices 
            split_name = "🟡 VAL"
        elif split == 'test':
            available_indices = self.test_indices
            split_name = "🔴 TEST"
        else:
            raise ValueError(f"无效的数据划分: {split}，必须是 'train', 'val', 或 'test'")
        
        if len(available_indices) == 0:
            raise ValueError(f"{split_name} 划分为空!")
        
        # 从对应划分中随机采样
        if batch_size > len(available_indices):
            # 如果请求的批次大小超过可用数据，进行重复采样
            selected_indices = np.random.choice(available_indices, batch_size, replace=True)
        else:
            # 否则进行无重复采样
            selected_indices = np.random.choice(available_indices, batch_size, replace=False)
        
        # 获取数据
        images = self.images_cache[selected_indices].to(device)
        text_sequences = self.text_sequences_cache[selected_indices].to(device)
        labels = self.labels_cache[selected_indices].to(device)
        
        
        return images, text_sequences, labels

    def get_batch(self, batch_size, device):
        """
        ⚠️  兼容性接口：随机从所有数据中采样（不推荐用于严格的训练/测试分离）
        建议使用 get_split_batch() 方法
        """
        print("⚠️  Warning: 使用get_batch()可能破坏训练/测试分离，建议使用get_split_batch()")

        indices = torch.randperm(self.images_cache.shape[0])[:batch_size]
        images = self.images_cache[indices].to(device)
        text_sequences = self.text_sequences_cache[indices].to(device)
        labels = self.labels_cache[indices].to(device)

        return images, text_sequences, labels

    def get_class_info(self):
        return len(self.class_names)

    def get_text_vectorizer(self):  # 名称保留，但现在返回的是词汇表信息
        return self.word_to_idx, self.idx_to_word, self.vocab_size


class ComplexDataGenerator:
    """复杂数据生成器 - 使用多模态PathMNIST真实数据"""

    def __init__(self, config):
        self.config = config
        self.anomaly_generator = ComplexAnomalyGenerator()
        self.multimodal_loader = MultimodalPathMNISTDataLoader(
            data_path=config.get('data_path',
                                 "F:\\Desktop\\HyperBridge\\HyperBridge-main\\multimodal_medmnist_datasets\\pathmnist"),
            # 传递 data_path
            batch_size=config.get('batch_size', 32),
            cache_size=config.get('cache_size', 1024),
            embedding_dim=config.get('embedding_dim', 64),
            max_seq_len=config.get('max_seq_len', 64)
        )

    def generate_multimodal_data(self, batch_size, device, split='train', anomaly_ratio=0.15):
        """
        ✅ 根据数据划分生成多模态数据（确保测试集隔离）
        
        Args:
            batch_size: 批次大小
            device: 设备
            split: 数据划分 {'train', 'val', 'test'}
            anomaly_ratio: 异常样本比例
        """
        # 使用新的划分接口获取数据
        images, text_sequences, original_labels = self.multimodal_loader.get_split_batch(batch_size, device, split)

        # 添加基础噪声
        base_noise = torch.randn_like(images) * 0.05
        images = images + base_noise

        # 生成异常样本
        images, perturbed_text_sequences, anomaly_labels, anomaly_types = \
            self.anomaly_generator.generate_anomalies_with_real_text(
                images, text_sequences, original_labels, self.config, anomaly_ratio,
                vocab_size=self.multimodal_loader.vocab_size
            )


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

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
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

                # 🟢 严格使用训练集数据
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='train', anomaly_ratio=0.15
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
                # 🟡 仅使用验证集数据
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='val', anomaly_ratio=0.2
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
        print(f"test evaluation")
        self.model.eval()
        all_scores = []
        all_labels = []
        batch_size = self.config.get('batch_size', 32)

        with torch.no_grad():
            for _ in range(num_batches):
                #  首次接触测试集数据（仅在最终评估时）
                images, text_sequences, labels, anomaly_types = \
                    self.data_generator.generate_multimodal_data(
                        batch_size, self.device, split='test', anomaly_ratio=0.2
                    )

                scores, _ = self.model(images, text_sequences)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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



class AblationExperiment:
    """消融实验管理器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

        # 基础配置 - 适配PathMNIST数据集
        self.base_config = {
            'hidden_dim': 128,
            'repr_dim': 64,
            'final_repr_dim': 32,
            'embedding_dim': 64,
            'max_seq_len': 64,
            'vocab_size': None,
            'text_dim': 64,

            # =================== 坚定地修改以下三个参数 ===================
            'top_k': 8, 
            'cheb_k': 10,  
            'learning_rate': 0.0005,
 

            'threshold': 0.6,  # 此参数可保持不变
            'k_nearest': 3,
            'tau': 0.5,
            'lambda_recon': 0.1,
            'lambda_spectral': 0.01,
            'batch_size': 32,
            'weight_decay': 1e-5,
            'cache_size': 1024,
            'num_classes': 9,  # PathMNIST有9个类别
            'image_size': 28,
            'lstm_layers': 2,
            'lstm_dropout': 0.3,  # 推荐从0.3开始，如果过拟合再增加
            'data_path': r"F:\\Desktop\\HyperBridge\\HyperBridge-main\\multimodal_medmnist_datasets\\pathmnist"  # 确保 DataLoader 能找到文件
        }

        self.ablation_configs = {
            'original_baseline_bilstm': {
                'name': 'Complete Model (Baseline)',  # <--- 已修改为您指定的名称
                'hypergraph_type': 'dynamic',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': True,
            },
            # 'optimized_baseline_bilstm' 整个条目已被删除
            'fixed_knn': {
                'name': 'Exp1: Fixed k-NN Hyperedge Generation',  # 为了简洁，也可以一并修改这些
                'hypergraph_type': 'fixed_knn',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': True,
            },
            'simple_conv': {
                'name': 'Exp2: Simple Graph Convolution',
                'hypergraph_type': 'dynamic',
                'conv_type': 'simple_graph',
                'use_spectral_regularizer': True,
            },
            'no_spectral_reg': {
                'name': 'Exp3: No Spectral Pruning Regularization',
                'hypergraph_type': 'dynamic',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': False,
            },
            'similarity_only': {
                'name': 'Exp4a: Similarity Prior Only',
                'hypergraph_type': 'similarity_only',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': True,
            },
            'attention_only': {
                'name': 'Exp4b: Attention Scoring Only',
                'hypergraph_type': 'attention_only',
                'conv_type': 'wavelet_cheb',
                'use_spectral_regularizer': True,
            },
        }

    def run_experiment(self, experiment_name, epochs=30, num_runs=3):
        """运行单个消融实验（多次运行取平均值）"""
        print(f"\n{'=' * 20} {self.ablation_configs[experiment_name]['name']} {'=' * 20}")
        print(f"🔄 将运行 {num_runs} 次实验并计算平均结果")

        ablation_config = self.ablation_configs[experiment_name]
        
        # 存储所有运行的结果
        all_train_losses = []
        all_val_metrics = []
        all_final_results = []
        all_aucs = []
        all_accuracies = []

        successful_runs = 0
        
        for run_idx in range(num_runs):
            print(f"\n🏃‍♂️ 开始第 {run_idx + 1}/{num_runs} 次运行...")
            
            try:
                # 每次运行都创建新的模型实例
                model = AblationAnomalyDetector(self.base_config, ablation_config).to(self.device)
                trainer = AblationTrainer(model, self.base_config, self.device)

                print(f"Starting training for {epochs} epochs...")
                train_losses, val_metrics = trainer.train(epochs, train_batches=15, val_batches=5)
                final_results = trainer.evaluate_testset(num_batches=10)

                # 收集结果
                all_train_losses.append(train_losses)
                all_val_metrics.append(val_metrics)
                all_final_results.append(final_results)
                all_aucs.append(final_results['auc'])
                all_accuracies.append(final_results['accuracy'])
                
                successful_runs += 1
                
                print(f"✅ 第 {run_idx + 1} 次运行完成 - AUC: {final_results['auc']:.4f}, "
                      f"Accuracy: {final_results['accuracy']:.4f}")

            except Exception as e:
                print(f"❌ 第 {run_idx + 1} 次运行失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if successful_runs == 0:
            print(f"❌ 所有 {num_runs} 次运行都失败了")
            default_results = {
                'auc': 0.0,
                'accuracy': 0.0,
                'error': 'All runs failed'
            }

            self.results[experiment_name] = {
                'config': ablation_config,
                'train_losses': [],
                'val_metrics': [],
                'final_results': default_results,
                'val_aucs': [],
                'final_auc': 0.0,
                'run_details': {
                    'num_runs': num_runs,
                    'successful_runs': 0,
                    'individual_results': []
                }
            }
            return default_results

        # 计算平均结果
        print(f"\n📊 计算 {successful_runs} 次成功运行的平均结果...")
        
        # 计算平均最终结果
        avg_auc = np.mean(all_aucs)
        avg_accuracy = np.mean(all_accuracies)
        std_auc = np.std(all_aucs)
        std_accuracy = np.std(all_accuracies)
        
        # 计算平均训练损失（取相同长度的部分）
        min_train_length = min(len(losses) for losses in all_train_losses)
        avg_train_losses = []
        for epoch_idx in range(min_train_length):
            epoch_losses = [losses[epoch_idx] for losses in all_train_losses]
            avg_train_losses.append(np.mean(epoch_losses))
        
        # 计算平均验证指标
        min_val_length = min(len(metrics) for metrics in all_val_metrics)
        avg_val_metrics = []
        for val_idx in range(min_val_length):
            val_aucs = [metrics[val_idx]['auc'] for metrics in all_val_metrics]
            val_accs = [metrics[val_idx]['accuracy'] for metrics in all_val_metrics]
            avg_val_metrics.append({
                'auc': np.mean(val_aucs),
                'accuracy': np.mean(val_accs)
            })

        avg_final_results = {
            'auc': avg_auc,
            'accuracy': avg_accuracy,
            'auc_std': std_auc,
            'accuracy_std': std_accuracy
        }

        # 存储结果
        self.results[experiment_name] = {
            'config': ablation_config,
            'train_losses': avg_train_losses,
            'val_metrics': avg_val_metrics,
            'final_results': avg_final_results,
            'val_aucs': [m['auc'] for m in avg_val_metrics],
            'final_auc': avg_auc,
            'run_details': {
                'num_runs': num_runs,
                'successful_runs': successful_runs,
                'individual_aucs': all_aucs,
                'individual_accuracies': all_accuracies,
                'individual_results': all_final_results
            }
        }

        print(f"🎯 实验 {ablation_config['name']} 完成")
        print(f"📈 平均结果 ({successful_runs} 次运行):")
        print(f"   AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"   Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"   个体AUC: {[f'{auc:.4f}' for auc in all_aucs]}")
        print(f"   个体Accuracy: {[f'{acc:.4f}' for acc in all_accuracies]}")

        return avg_final_results

    def run_all_experiments(self, epochs=30):
        """运行所有消融实验"""
        print(f"Starting ablation experiments, device: {self.device}")

        # 动态获取 vocab_size 并更新 base_config
        # 传入 data_path 到 MultimodalPathMNISTDataLoader
        temp_data_loader = MultimodalPathMNISTDataLoader(
            data_path=self.base_config['data_path'],  # 使用配置中的data_path
            batch_size=self.base_config['batch_size'],
            cache_size=self.base_config['cache_size'],
            embedding_dim=self.base_config['embedding_dim'],
            max_seq_len=self.base_config['max_seq_len']
        )
        _, _, vocab_size = temp_data_loader.get_text_vectorizer()
        self.base_config['vocab_size'] = vocab_size  # 更新 vocab_size
        print(f"Dynamic vocabulary size detected: {vocab_size}")
        # 清理临时 loader，避免重复加载
        del temp_data_loader

        # 运行所有实验 (已取消注释)
        experiment_order = [
            'original_baseline_bilstm',
            'fixed_knn',
            'simple_conv',
            'no_spectral_reg',
            'similarity_only',
            'attention_only'
        ]

        for exp_name in experiment_order:
            if exp_name in self.ablation_configs:
                try:
                    self.run_experiment(exp_name, epochs)
                except Exception as e:
                    print(f"Experiment {exp_name} failed: {str(e)}")
                    continue
            else:
                print(f"Warning: Experiment '{exp_name}' not found in ablation_configs. Skipping.")

        return self.results

    def create_visualization(self):
        """Create comprehensive visualization results with AUC and Accuracy (display only) including error bars"""
        if not self.results:
            print("No available experimental results")
            return

        # Define a mapping for shorter, readable plot labels
        plot_label_map = {
            'Complete Model (Baseline)': 'Complete Model',
            'Exp1: Fixed k-NN Hyperedge Generation': 'Exp1: Fixed k-NN',
            'Exp2: Simple Graph Convolution': 'Exp2: Simple GCN',
            'Exp3: No Spectral Pruning Regularization': 'Exp3: No SpecReg',
            'Exp4a: Similarity Prior Only': 'Exp4a: Sim Only',
            'Exp4b: Attention Scoring Only': 'Exp4b: Att Only',
        }

        methods = []
        aucs = []
        accuracies = []
        combined_scores = []
        auc_stds = []
        accuracy_stds = []

        ordered_exp_names = [k for k in self.ablation_configs.keys() if k in self.results]

        for exp_name in ordered_exp_names:
            result = self.results[exp_name]
            config = result['config']
            methods.append(plot_label_map.get(config['name'], config['name']))
            auc = result['final_results'].get('auc', 0.0)
            accuracy = result['final_results'].get('accuracy', 0.0)
            auc_std = result['final_results'].get('auc_std', 0.0)
            accuracy_std = result['final_results'].get('accuracy_std', 0.0)
            
            aucs.append(auc)
            accuracies.append(accuracy)
            auc_stds.append(auc_std)
            accuracy_stds.append(accuracy_std)
            # 50% AUC + 50% Accuracy for combined ranking
            combined_scores.append((auc + accuracy) / 2)

        plt.figure(figsize=(18, 12))

        # Subplot 1: Performance Comparison Bar Chart with Combined Score and Error Bars
        plt.subplot(2, 3, 1)
        x = np.arange(len(methods))
        width = 0.25

        bars1 = plt.bar(x - width, aucs, width, label='AUC', color='skyblue', alpha=0.8, yerr=auc_stds, capsize=3)
        bars2 = plt.bar(x, accuracies, width, label='Accuracy', color='lightgreen', alpha=0.8, yerr=accuracy_stds, capsize=3)
        combined_stds = [(auc_std + acc_std) / 2 for auc_std, acc_std in zip(auc_stds, accuracy_stds)]
        bars3 = plt.bar(x + width, combined_scores, width, label='Combined (50%AUC+50%ACC)', color='orange', alpha=0.8, yerr=combined_stds, capsize=3)

        plt.ylabel('Score', fontsize=10)
        plt.title('Performance Comparison (with Standard Deviation)', fontsize=12)
        plt.xticks(x, methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')

        for i, (auc, acc, combined) in enumerate(zip(aucs, accuracies, combined_scores)):
            plt.text(i - width, auc + auc_stds[i] + 0.01, f'{auc:.3f}±{auc_stds[i]:.3f}', ha='center', va='bottom', fontsize=6)
            plt.text(i, acc + accuracy_stds[i] + 0.01, f'{acc:.3f}±{accuracy_stds[i]:.3f}', ha='center', va='bottom', fontsize=6)
            plt.text(i + width, combined + combined_stds[i] + 0.01, f'{combined:.3f}', ha='center', va='bottom', fontsize=6)

        # Subplot 2: AUC Comparison with Error Bars
        plt.subplot(2, 3, 2)
        bars = plt.bar(range(len(methods)), aucs, color='skyblue', alpha=0.8, yerr=auc_stds, capsize=4)
        plt.ylabel('AUC Score', fontsize=10)
        plt.title('AUC Comparison (with Std Dev)', fontsize=12)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        for i, (v, std) in enumerate(zip(aucs, auc_stds)):
            plt.text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=7)

        # Subplot 3: Accuracy Comparison with Error Bars
        plt.subplot(2, 3, 3)
        bars = plt.bar(range(len(methods)), accuracies, color='lightgreen', alpha=0.8, yerr=accuracy_stds, capsize=4)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Comparison (with Std Dev)', fontsize=12)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        for i, (v, std) in enumerate(zip(accuracies, accuracy_stds)):
            plt.text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=7)

        # Subplot 4: Training Loss Curves
        plt.subplot(2, 3, 4)
        for exp_name in ordered_exp_names:
            result = self.results[exp_name]
            train_losses = result['train_losses']
            plt.plot(train_losses, label=plot_label_map.get(result['config']['name'], result['config']['name']),
                     linewidth=2)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Training Loss', fontsize=10)
        plt.title('Training Loss Curves', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=8)
        plt.grid(True, alpha=0.3)

        # Subplot 5: Validation AUC Trends
        plt.subplot(2, 3, 5)
        for exp_name in ordered_exp_names:
            result = self.results[exp_name]
            if 'val_metrics' in result and len(result['val_metrics']) > 0:
                val_aucs = [m['auc'] for m in result['val_metrics']]
                epochs = range(0, len(result['train_losses']), 10)[:len(val_aucs)]
                plt.plot(epochs, val_aucs, 'o-',
                         label=plot_label_map.get(result['config']['name'], result['config']['name']),
                         linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Validation AUC', fontsize=10)
        plt.title('Validation AUC Trends', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=8)
        plt.grid(True, alpha=0.3)

        # Subplot 6: Performance Ranking by Combined Score (50%AUC+50%ACC)
        plt.subplot(2, 3, 6)
        sorted_indices = np.argsort(combined_scores)[::-1]
        sorted_methods = [methods[i] for i in sorted_indices]
        sorted_combined_scores = [combined_scores[i] for i in sorted_indices]

        bars = plt.barh(range(len(sorted_methods)), sorted_combined_scores, color='orange', alpha=0.8)
        plt.xlabel('Combined Score (50%AUC+50%ACC)', fontsize=10)
        plt.title('Performance Ranking', fontsize=12)
        plt.yticks(range(len(sorted_methods)), sorted_methods, fontsize=8)

        for i, v in enumerate(sorted_combined_scores):
            plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

        plt.tight_layout()
        plt.show()  # Display the plot instead of saving

        # Display performance summary in console
        self._print_performance_summary(methods, aucs, accuracies, combined_scores, auc_stds, accuracy_stds)

    def _print_performance_summary(self, methods, aucs, accuracies, combined_scores, auc_stds, accuracy_stds):
        """Print performance summary to console with combined ranking and standard deviations"""
        print("\n" + "="*90)
        print("ABLATION EXPERIMENT PERFORMANCE SUMMARY (Multiple Runs Average ± Std)")
        print("="*90)
        
        # Create a list of tuples for sorting by combined score
        results = list(zip(methods, aucs, accuracies, combined_scores, auc_stds, accuracy_stds))
        # Sort by combined score (50%AUC+50%ACC) in descending order
        results.sort(key=lambda x: x[3], reverse=True)
        
        print(f"{'Rank':<4} {'Method':<20} {'AUC (±std)':<15} {'Accuracy (±std)':<18} {'Combined':<10}")
        print("-" * 85)
        
        for rank, (method, auc, acc, combined, auc_std, acc_std) in enumerate(results, 1):
            print(f"{rank:<4} {method:<20} {auc:.4f}±{auc_std:.4f}    {acc:.4f}±{acc_std:.4f}      {combined:<10.4f}")
        
        print("\n" + "="*90)
        print("📊 RANKING CRITERIA: 50% AUC + 50% Accuracy")
        print("📈 Results are averaged over multiple runs with standard deviation shown")
        print("="*90)



    def print_analysis_report(self):
        """Print analysis report"""
        if not self.results:
            print("No experimental results available")
            return

        print(f"\n{'=' * 60}")
        print("Ablation Study Analysis Report (Multiple Runs Average)")
        print(f"{'=' * 60}")

        print(f"\nExperimental Results (Average ± Std from Multiple Runs):")
        print("-" * 70)
        # 在这里重新计算 _sorted_results_items，确保其在当前方法中可用
        _results_for_analysis = {k: v for k, v in self.results.items() if k in self.ablation_configs}
        _sorted_results_items = sorted(_results_for_analysis.items(), key=lambda x: x[1].get('final_auc', 0.0),
                                       reverse=True)

        for exp_name, result in _sorted_results_items:
            config = result['config']
            auc = result['final_auc']
            accuracy = result['final_results'].get('accuracy', 0.0)
            auc_std = result['final_results'].get('auc_std', 0.0)
            accuracy_std = result['final_results'].get('accuracy_std', 0.0)
            
            # 显示运行次数信息
            run_details = result.get('run_details', {})
            num_runs = run_details.get('successful_runs', 1)
            
            print(f"{config['name']:<50} | AUC: {auc:.4f}±{auc_std:.4f} | Acc: {accuracy:.4f}±{accuracy_std:.4f} | Runs: {num_runs}")

        print(f"\nPerformance Ranking (by AUC):")
        print("-" * 50)
        for rank, (exp_name, result) in enumerate(_sorted_results_items, 1):  # 使用重新定义的 _sorted_results_items
            config = result['config']
            auc = result['final_auc']
            auc_std = result['final_results'].get('auc_std', 0.0)
            run_details = result.get('run_details', {})
            num_runs = run_details.get('successful_runs', 1)
            print(f"{rank}. {config['name']} (AUC: {auc:.4f}±{auc_std:.4f}, {num_runs} runs)")

        # 显示个体运行结果的详细信息
        print(f"\n📊 Individual Run Details:")
        print("-" * 50)
        for exp_name, result in _sorted_results_items:
            config = result['config']
            run_details = result.get('run_details', {})
            if 'individual_aucs' in run_details and 'individual_accuracies' in run_details:
                individual_aucs = run_details['individual_aucs']
                individual_accs = run_details['individual_accuracies']
                print(f"\n{config['name']}:")
                print(f"  Individual AUCs: {[f'{auc:.4f}' for auc in individual_aucs]}")
                print(f"  Individual Accs: {[f'{acc:.4f}' for acc in individual_accs]}")

        if 'original_baseline_bilstm' in self.results:
            original_baseline_result = self.results.get('original_baseline_bilstm', {})
            original_baseline_auc = original_baseline_result.get('final_auc', 0.0)
            original_baseline_std = original_baseline_result.get('final_results', {}).get('auc_std', 0.0)

            print(f"\nKey Findings (All experiments use BiLSTM with Word Embeddings, Multiple Runs):")
            print("-" * 70)

            print(f"  - Complete Model (Baseline) AUC: {original_baseline_auc:.4f}±{original_baseline_std:.4f}")

            max_drop = 0
            most_impactful_ablation = None

            print(f"\n  - Impact of module changes (relative to Complete Model Baseline):")
            for exp_name, result in _sorted_results_items:  # 使用重新定义的 _sorted_results_items
                if exp_name not in ['original_baseline_bilstm']:
                    current_auc = result.get('final_auc', 0.0)
                    current_std = result.get('final_results', {}).get('auc_std', 0.0)
                    drop = original_baseline_auc - current_auc
                    impact_type = "Drop" if drop > 0 else "Gain" if drop < 0 else "No Change"
                    print(f"    - '{result['config']['name'].split('(')[0].strip()}': {impact_type} in AUC: {abs(drop):.4f} (±{current_std:.4f})")
                    if abs(drop) > abs(max_drop):
                        max_drop = drop
                        most_impactful_ablation = result['config']['name']

            if most_impactful_ablation:
                impact_type = "Drop" if max_drop > 0 else "Gain" if max_drop < 0 else "No Change"
                print(f"\n  - **Most impactful change (largest absolute AUC difference from Baseline):**")
                print(f"    -> '{most_impactful_ablation.split('(')[0].strip()}' ({impact_type} of {abs(max_drop):.4f})")
        else:
            print("Baseline results not available for detailed findings.")

        print(f"\n{'=' * 60}")


def main():
    """Main function"""

    print("🧪 Multimodal Hypergraph Anomaly Detection System - Ablation Study (PathMNIST)")
    print("================================================================================")
    print("🔄 Note: Each experiment will run 3 times to ensure statistical reliability")
    print("📊 Results will be averaged with standard deviation reported")

    ablation_exp = AblationExperiment()

    print("Starting ablation experiments with PathMNIST dataset...")
    print("⚠️  This may take longer due to multiple runs per experiment...")
    results = ablation_exp.run_all_experiments(epochs=80)

    print("\nGenerating comprehensive experiment visualization with error bars...")
    ablation_exp.create_visualization()

    print("\nPrinting detailed analysis report with multiple-run statistics...")
    ablation_exp.print_analysis_report()

    print("\n🎉 PathMNIST ablation experiments completed!")
    print("✅ All metrics (AUC, Accuracy) averaged over multiple runs")
    print("📈 Standard deviations provide confidence intervals for results")


if __name__ == "__main__":
    main()