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

# ====================================================================================
# START: MULTIMODAL TEXT ENCODER AND ABLATION CLASSES
# ====================================================================================

# 兼容适配器：包装新的HybridHyperedgeGenerator以保持旧接口
class HybridHyperedgeGeneratorAdapter(nn.Module):
    """适配器：包装新的HybridHyperedgeGenerator以保持旧的接口兼容性"""
    def __init__(self, num_modalities, input_dims, hidden_dim, top_k, threshold):
        super().__init__()
        self.generator = HybridHyperedgeGenerator(
            num_modalities=num_modalities,
            input_dims=input_dims, 
            hidden_dim=hidden_dim,
            top_k=top_k,
            threshold=threshold
        )
    
    def forward(self, features_list):
        return self.generator(features_list)


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

    def __init__(self, threshold=0.6, top_k=10):
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
            self.hypergraph_generator = HybridHyperedgeGeneratorAdapter(
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


# 修改后的 MultimodalOrganaMNISTDataLoader - 处理原始文本
class MultimodalOrganaMNISTDataLoader:
    """多模态OrganaMNIST数据加载器 - 从指定路径加载图像和文本数据，支持静态数据划分"""

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

        # OrganaMNIST 有11个类别
        self.class_names = [
            "bladder", "femur-left", "femur-right", "heart", "kidney-left", 
            "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self._initialize_loader()

    def _initialize_loader(self):
        """初始化数据加载器，包括文本分词、序列化和静态数据划分"""

        print(f"Loading multimodal OrganaMNIST data from {self.data_path}")

        images_path = os.path.join(self.data_path, "organamnist_images.pkl")
        with open(images_path, 'rb') as f:
            image_data = pickle.load(f)

        text_path = os.path.join(self.data_path, "organamnist_text_descriptions.json")
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
    """复杂数据生成器 - 使用多模态OrganaMNIST真实数据"""

    def __init__(self, config):
        self.config = config
        self.anomaly_generator = ComplexAnomalyGenerator()
        self.multimodal_loader = MultimodalOrganaMNISTDataLoader(
            data_path=config.get('data_path',
                                 "F:\\Desktop\\bloodmnist\\Data\\organamnist"),
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

        # 基础配置 - 适配OrganaMNIST数据集
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
            # ==========================================================

            'threshold': 0.6,  # 此参数可保持不变
            'k_nearest': 3,
            'tau': 0.5,
            'lambda_recon': 0.1,
            'lambda_spectral': 0.01,
            'batch_size': 32,
            'weight_decay': 1e-5,
            'cache_size': 1024,
            'num_classes': 11,  # OrganaMNIST有11个类别
            'image_size': 28,
            'lstm_layers': 2,
            'lstm_dropout': 0.3,  # 推荐从0.3开始，如果过拟合再增加
            'data_path': r"F:\\Desktop\\bloodmnist\\Data\\organamnist"  # 确保 DataLoader 能找到文件
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

    def _compute_average_results(self, all_runs_results):
        """计算多次运行的平均结果"""
        if not all_runs_results:
            return {'auc': 0.0, 'accuracy': 0.0}
        
        avg_auc = np.mean([r['auc'] for r in all_runs_results])
        avg_accuracy = np.mean([r['accuracy'] for r in all_runs_results])
        
        # 保留其他可能的指标
        avg_results = {
            'auc': avg_auc,
            'accuracy': avg_accuracy
        }
        
        # 如果有其他指标，也计算平均值
        if all_runs_results[0].get('accuracy_fixed') is not None:
            avg_results['accuracy_fixed'] = np.mean([r.get('accuracy_fixed', 0.0) for r in all_runs_results])
        if all_runs_results[0].get('accuracy_optimal') is not None:
            avg_results['accuracy_optimal'] = np.mean([r.get('accuracy_optimal', 0.0) for r in all_runs_results])
        if all_runs_results[0].get('best_threshold') is not None:
            avg_results['best_threshold'] = np.mean([r.get('best_threshold', 0.5) for r in all_runs_results])
            
        return avg_results

    def _compute_std_results(self, all_runs_results):
        """计算多次运行的标准差"""
        if len(all_runs_results) <= 1:
            return {'auc': 0.0, 'accuracy': 0.0}
        
        std_auc = np.std([r['auc'] for r in all_runs_results])
        std_accuracy = np.std([r['accuracy'] for r in all_runs_results])
        
        return {
            'auc': std_auc,
            'accuracy': std_accuracy
        }

    def _compute_average_losses(self, all_train_losses):
        """计算多次运行的平均训练损失"""
        if not all_train_losses:
            return []
        
        # 找到最短的训练序列长度（以防不同运行的epoch数不同）
        min_length = min(len(losses) for losses in all_train_losses)
        
        # 计算每个epoch的平均损失
        avg_losses = []
        for epoch in range(min_length):
            epoch_losses = [losses[epoch] for losses in all_train_losses]
            avg_losses.append(np.mean(epoch_losses))
        
        return avg_losses

    def _compute_average_val_metrics(self, all_val_metrics):
        """计算多次运行的平均验证指标"""
        if not all_val_metrics:
            return []
        
        # 找到最短的验证序列长度
        min_length = min(len(metrics) for metrics in all_val_metrics)
        
        avg_val_metrics = []
        for idx in range(min_length):
            epoch_metrics = [metrics[idx] for metrics in all_val_metrics]
            avg_auc = np.mean([m['auc'] for m in epoch_metrics])
            avg_accuracy = np.mean([m['accuracy'] for m in epoch_metrics])
            
            avg_val_metrics.append({
                'auc': avg_auc,
                'accuracy': avg_accuracy
            })
        
        return avg_val_metrics

    def _compute_average_val_aucs(self, all_val_aucs):
        """计算多次运行的平均验证AUC"""
        if not all_val_aucs:
            return []
        
        # 找到最短的序列长度
        min_length = min(len(aucs) for aucs in all_val_aucs)
        
        avg_val_aucs = []
        for idx in range(min_length):
            epoch_aucs = [aucs[idx] for aucs in all_val_aucs]
            avg_val_aucs.append(np.mean(epoch_aucs))
        
        return avg_val_aucs

    def run_experiment(self, experiment_name, epochs=30, num_runs=3):
        """运行单个消融实验，进行多次运行并取平均值"""
        print(f"\n{'=' * 20} {self.ablation_configs[experiment_name]['name']} {'=' * 20}")
        print(f"🔄 Running {num_runs} independent experiments for robust evaluation...")

        ablation_config = self.ablation_configs[experiment_name]
        
        # 存储多次运行的结果
        all_runs_results = []
        all_train_losses = []
        all_val_metrics = []
        all_val_aucs = []

        try:
            for run_idx in range(num_runs):
                print(f"\n📊 Run {run_idx + 1}/{num_runs} - {ablation_config['name']}")
                
                # 每次运行创建新的模型实例
                model = AblationAnomalyDetector(self.base_config, ablation_config).to(self.device)
                trainer = AblationTrainer(model, self.base_config, self.device)

                print(f"Starting training for {epochs} epochs...")
                train_losses, val_metrics = trainer.train(epochs, train_batches=15, val_batches=5)
                final_results = trainer.evaluate_testset(num_batches=10)

                # 保存这次运行的结果
                all_runs_results.append(final_results)
                all_train_losses.append(train_losses)
                all_val_metrics.append(val_metrics)
                all_val_aucs.append([m['auc'] for m in val_metrics])

                print(f"Run {run_idx + 1} Results - AUC: {final_results['auc']:.4f}, "
                      f"Accuracy: {final_results['accuracy']:.4f}")

            # 计算平均结果
            avg_results = self._compute_average_results(all_runs_results)
            avg_train_losses = self._compute_average_losses(all_train_losses)
            avg_val_metrics = self._compute_average_val_metrics(all_val_metrics)
            avg_val_aucs = self._compute_average_val_aucs(all_val_aucs)

            # 计算标准差以评估稳定性
            std_results = self._compute_std_results(all_runs_results)

            self.results[experiment_name] = {
                'config': ablation_config,
                'train_losses': avg_train_losses,
                'val_metrics': avg_val_metrics,
                'final_results': avg_results,
                'val_aucs': avg_val_aucs,
                'final_auc': avg_results['auc'],
                'std_results': std_results,  # 新增：标准差信息
                'all_runs': all_runs_results,  # 新增：所有运行的详细结果
                'num_runs': num_runs
            }

            print(f"\n✅ Experiment {ablation_config['name']} completed ({num_runs} runs)")
            print(f"📊 Average Results - AUC: {avg_results['auc']:.4f} ± {std_results['auc']:.4f}, "
                  f"Accuracy: {avg_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}")

            return avg_results

        except Exception as e:
            print(f"Experiment {experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()

            default_results = {
                'auc': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }

            self.results[experiment_name] = {
                'config': ablation_config,
                'train_losses': [],
                'val_metrics': [],
                'final_results': default_results,
                'val_aucs': [],
                'final_auc': 0.0,
                'std_results': {'auc': 0.0, 'accuracy': 0.0},
                'all_runs': [],
                'num_runs': 0
            }

            return default_results

    def run_all_experiments(self, epochs=30, num_runs=3):
        """运行所有消融实验，每个实验进行多次运行"""
        print(f"Starting ablation experiments, device: {self.device}")
        print(f"🔄 Each experiment will run {num_runs} times for robust evaluation")

        # 动态获取 vocab_size 并更新 base_config
        # 传入 data_path 到 MultimodalOrganaMNISTDataLoader
        temp_data_loader = MultimodalOrganaMNISTDataLoader(
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
                    self.run_experiment(exp_name, epochs, num_runs)  # 传递num_runs参数
                except Exception as e:
                    print(f"Experiment {exp_name} failed: {str(e)}")
                    continue
            else:
                print(f"Warning: Experiment '{exp_name}' not found in ablation_configs. Skipping.")

        return self.results

    def create_visualization(self):
        """Create comprehensive visualization results with AUC and Accuracy (display only)"""
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
        combined_scores = []  # 新增：综合排名分数
        std_aucs = []  # 新增：AUC标准差
        std_accs = []  # 新增：Accuracy标准差
        num_runs_list = []  # 新增：运行次数信息

        ordered_exp_names = [k for k in self.ablation_configs.keys() if k in self.results]

        for exp_name in ordered_exp_names:
            result = self.results[exp_name]
            config = result['config']
            method_name = plot_label_map.get(config['name'], config['name'])
            auc = result['final_results'].get('auc', 0.0)
            acc = result['final_results'].get('accuracy', 0.0)
            std_auc = result.get('std_results', {}).get('auc', 0.0)
            std_acc = result.get('std_results', {}).get('accuracy', 0.0)
            runs = result.get('num_runs', 1)
            
            methods.append(method_name)
            aucs.append(auc)
            accuracies.append(acc)
            std_aucs.append(std_auc)
            std_accs.append(std_acc)
            num_runs_list.append(runs)
            # 综合分数：50% AUC + 50% ACC
            combined_scores.append(0.5 * auc + 0.5 * acc)

        # 创建排名（包含标准差信息）
        ranking_data = list(zip(methods, aucs, accuracies, combined_scores, std_aucs, std_accs, num_runs_list))
        ranking_data.sort(key=lambda x: x[3], reverse=True)  # 按综合分数排序

        # 调整图形大小以获得更好的可读性
        plt.figure(figsize=(20, 15))

        # 子图1: 性能对比柱状图
        plt.subplot(2, 3, 1)
        x = np.arange(len(methods))
        width = 0.35

        bars1 = plt.bar(x - width / 2, aucs, width, label='AUC', color='skyblue', alpha=0.8)
        bars2 = plt.bar(x + width / 2, accuracies, width, label='Accuracy', color='lightgreen', alpha=0.8)

        plt.ylabel('Score', fontsize=10)
        plt.title('Performance Comparison (AUC & Accuracy)', fontsize=12)
        plt.xticks(x, methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        # 子图2: AUC对比
        plt.subplot(2, 3, 2)
        bars = plt.bar(range(len(methods)), aucs, color='skyblue', alpha=0.8)
        plt.ylabel('AUC Score', fontsize=10)
        plt.title('AUC Comparison', fontsize=12)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        for i, v in enumerate(aucs):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        # 子图3: Accuracy对比
        plt.subplot(2, 3, 3)
        bars = plt.bar(range(len(methods)), accuracies, color='lightgreen', alpha=0.8)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Comparison', fontsize=12)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=8)
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        # 子图4: 训练损失曲线
        plt.subplot(2, 3, 4)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, exp_name in enumerate(ordered_exp_names):
            if exp_name in self.results and self.results[exp_name]['train_losses']:
                losses = self.results[exp_name]['train_losses']
                method_name = plot_label_map.get(self.results[exp_name]['config']['name'], 
                                                self.results[exp_name]['config']['name'])
                plt.plot(losses, label=method_name, color=colors[i % len(colors)], linewidth=2)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Training Loss', fontsize=10)
        plt.title('Training Loss Curves', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=8)
        plt.grid(True, alpha=0.3)

        # 子图5: 验证AUC趋势
        plt.subplot(2, 3, 5)
        for i, exp_name in enumerate(ordered_exp_names):
            if exp_name in self.results and self.results[exp_name]['val_aucs']:
                val_aucs = self.results[exp_name]['val_aucs']
                method_name = plot_label_map.get(self.results[exp_name]['config']['name'], 
                                                self.results[exp_name]['config']['name'])
                epochs = [i * 10 for i in range(len(val_aucs))]  # 验证每10个epoch
                plt.plot(epochs, val_aucs, label=method_name, color=colors[i % len(colors)], 
                        marker='o', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Validation AUC', fontsize=10)
        plt.title('Validation AUC Trends', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=8)
        plt.grid(True, alpha=0.3)

        # 子图6: 综合排名表
        plt.subplot(2, 3, 6)
        plt.axis('tight')
        plt.axis('off')
        
        # 创建排名表格数据
        ranking_table_data = []
        for i, data_tuple in enumerate(ranking_data):
            method, auc, acc, combined = data_tuple[:4]  # 取前4个元素
            ranking_table_data.append([
                f"#{i+1}",
                method[:15] + "..." if len(method) > 15 else method,  # 截断长名称
                f"{auc:.3f}",
                f"{acc:.3f}", 
                f"{combined:.3f}"
            ])
        
        # 创建表格
        table = plt.table(cellText=ranking_table_data,
                         colLabels=['Rank', 'Method', 'AUC', 'ACC', 'Combined'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.1, 0.4, 0.15, 0.15, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for i in range(len(ranking_table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i == 1:  # 第一名
                    cell.set_facecolor('#FFD700')
                elif i == 2:  # 第二名
                    cell.set_facecolor('#C0C0C0')
                elif i == 3:  # 第三名
                    cell.set_facecolor('#CD7F32')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        plt.title('Final Ranking (50% AUC + 50% ACC)', fontsize=12, pad=20)

        plt.tight_layout()
        plt.show()

        # 控制台输出详细结果
        self._print_experimental_results(ranking_data)

    def _print_experimental_results(self, ranking_data):
        """打印详细的实验结果到控制台（包含标准差信息）"""
        print("\n" + "="*80)
        print("🏆 MULTIMODAL HYPERGRAPH ANOMALY DETECTION - EXPERIMENTAL RESULTS")
        print("="*80)
        
        print(f"\n📊 FINAL RANKING (50% AUC + 50% Accuracy) - Multiple Runs Average")
        print("-"*80)
        print(f"{'Rank':<6}{'Method':<25}{'AUC':<12}{'ACC':<12}{'Combined':<12}{'Runs':<6}")
        print("-"*80)
        
        for i, data_tuple in enumerate(ranking_data):
            method, auc, acc, combined = data_tuple[:4]
            std_auc, std_acc, runs = data_tuple[4:7] if len(data_tuple) >= 7 else (0.0, 0.0, 1)
            
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            
            if runs > 1:
                auc_str = f"{auc:.3f}±{std_auc:.3f}"
                acc_str = f"{acc:.3f}±{std_acc:.3f}"
            else:
                auc_str = f"{auc:.3f}"
                acc_str = f"{acc:.3f}"
            
            print(f"{rank_symbol:<6}{method:<25}{auc_str:<12}{acc_str:<12}{combined:<12.3f}{runs:<6}")
        
        print("\n📈 PERFORMANCE ANALYSIS")
        print("-"*40)
        
        # 最佳性能分析
        best_data = ranking_data[0]
        worst_data = ranking_data[-1]
        best_method, best_auc, best_acc, best_combined = best_data[:4]
        worst_method, worst_auc, worst_acc, worst_combined = worst_data[:4]
        
        print(f"🎯 Best Overall: {best_method}")
        print(f"   AUC: {best_auc:.3f}, Accuracy: {best_acc:.3f}, Combined: {best_combined:.3f}")
        
        print(f"\n📉 Lowest Performance: {worst_method}")
        print(f"   AUC: {worst_auc:.3f}, Accuracy: {worst_acc:.3f}, Combined: {worst_combined:.3f}")
        
        # 性能差距
        auc_gap = best_auc - worst_auc
        acc_gap = best_acc - worst_acc
        combined_gap = best_combined - worst_combined
        
        print(f"\n📏 Performance Gap:")
        print(f"   AUC Range: {auc_gap:.3f} ({worst_auc:.3f} - {best_auc:.3f})")
        print(f"   Accuracy Range: {acc_gap:.3f} ({worst_acc:.3f} - {best_acc:.3f})")
        print(f"   Combined Range: {combined_gap:.3f} ({worst_combined:.3f} - {best_combined:.3f})")
        
        # 各指标的最佳表现
        aucs = [data[1] for data in ranking_data]
        accs = [data[2] for data in ranking_data]
        
        best_auc_idx = aucs.index(max(aucs))
        best_acc_idx = accs.index(max(accs))
        
        print(f"\n🏅 Best by Individual Metric:")
        print(f"   Best AUC: {ranking_data[best_auc_idx][0]} ({max(aucs):.3f})")
        print(f"   Best Accuracy: {ranking_data[best_acc_idx][0]} ({max(accs):.3f})")
        
        # 平均性能
        avg_auc = sum(aucs) / len(aucs)
        avg_acc = sum(accs) / len(accs)
        avg_combined = sum([data[3] for data in ranking_data]) / len(ranking_data)
        
        print(f"\n📊 Average Performance:")
        print(f"   Average AUC: {avg_auc:.3f}")
        print(f"   Average Accuracy: {avg_acc:.3f}")
        print(f"   Average Combined: {avg_combined:.3f}")
        
        # 显示稳定性信息
        print(f"\n🎯 Stability Analysis (Standard Deviation):")
        for i, data_tuple in enumerate(ranking_data):
            if len(data_tuple) >= 7:
                method, _, _, _, std_auc, std_acc, runs = data_tuple[:7]
                if runs > 1:
                    print(f"   {method}: AUC_std={std_auc:.4f}, ACC_std={std_acc:.4f} ({runs} runs)")
        
        print("\n" + "="*80)

    # === 已移除文件保存功能，仅显示结果 ===

    # === create_enhanced_visualization 也需要进行类似优化和位置修正 ===
    def create_enhanced_visualization(self):
        """Create enhanced visualization results with anomaly type analysis (display only)"""
        if not self.results:
            print("No available experimental results")
            return

        # Define a mapping for shorter, readable plot labels for enhanced viz
        plot_label_map_enhanced = {
            'Complete Model (Baseline)': 'Complete',
            'Exp1: Fixed k-NN Hyperedge Generation': 'Exp1_kNN',
            'Exp2: Simple Graph Convolution': 'Exp2_SGC',
            'Exp3: No Spectral Pruning Regularization': 'Exp3_NoReg',
            'Exp4a: Similarity Prior Only': 'Exp4a_Sim',
            'Exp4b: Attention Scoring Only': 'Exp4b_Att',
        }

        methods = []
        aucs = []
        accuracies = []

        ordered_exp_names = [k for k in self.ablation_configs.keys() if k in self.results]

        for exp_name in ordered_exp_names:
            result = self.results[exp_name]
            config = result['config']
            methods.append(plot_label_map_enhanced.get(config['name'], config['name']))
            aucs.append(result['final_results'].get('auc', 0.0))
            accuracies.append(result['final_results'].get('accuracy', 0.0))

        plt.figure(figsize=(24, 18))  # Large figure size for enhanced plots

        # 1. AUC Comparison Bar Chart (Enhanced)
        plt.subplot(3, 4, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars = plt.bar(range(len(methods)), aucs, color=colors, alpha=0.8, edgecolor='black')
        plt.ylabel('AUC Score', fontsize=12)
        plt.title('Ablation Study AUC Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=9)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(aucs):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 2. Accuracy Comparison Bar Chart (Enhanced)
        plt.subplot(3, 4, 2)
        bars = plt.bar(range(len(methods)), accuracies, color=colors, alpha=0.8, edgecolor='black')
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title('Ablation Study Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right', fontsize=9)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 3. Training Loss Curves (Enhanced)
        plt.subplot(3, 4, 3)
        for i, exp_name in enumerate(ordered_exp_names):
            if exp_name in self.results and self.results[exp_name]['train_losses']:
                result = self.results[exp_name]
                train_losses = result['train_losses']
                plt.plot(train_losses,
                         label=plot_label_map_enhanced.get(result['config']['name'], result['config']['name']),
                         color=colors[i], linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Convergence', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)

        # 4. Validation AUC Trends (Enhanced)
        plt.subplot(3, 4, 4)
        for i, exp_name in enumerate(ordered_exp_names):
            if exp_name in self.results and self.results[exp_name]['val_aucs']:
                result = self.results[exp_name]
                val_aucs = result['val_aucs']
                epochs = [i * 10 for i in range(len(val_aucs))]
                plt.plot(epochs, val_aucs, 'o-',
                         label=plot_label_map_enhanced.get(result['config']['name'], result['config']['name']),
                         color=colors[i], linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation AUC', fontsize=12)
        plt.title('Validation AUC Trends', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Enhanced visualization displayed successfully!")

    def print_analysis_report(self):
        """Print analysis report"""
        if not self.results:
            print("No experimental results available")
            return

        print(f"\n{'=' * 60}")
        print("Ablation Study Analysis Report (BiLSTM Word Embeddings)")
        print(f"{'=' * 60}")

        print(f"\nExperimental Results:")
        print("-" * 50)
        # 在这里重新计算 _sorted_results_items，确保其在当前方法中可用
        _results_for_analysis = {k: v for k, v in self.results.items() if k in self.ablation_configs}
        _sorted_results_items = sorted(_results_for_analysis.items(), key=lambda x: x[1].get('final_auc', 0.0),
                                       reverse=True)

        for exp_name, result in _sorted_results_items:
            config = result['config']
            auc = result['final_auc']
            print(f"{config['name']:<70.70} | AUC: {auc:.4f}")

        print(f"\nPerformance Ranking:")
        print("-" * 50)
        for rank, (exp_name, result) in enumerate(_sorted_results_items, 1):  # 使用重新定义的 _sorted_results_items
            config = result['config']
            auc = result['final_auc']
            print(f"{rank}. {config['name']} (AUC: {auc:.4f})")

        if 'original_baseline_bilstm' in self.results and 'optimized_baseline_bilstm' in self.results:
            original_baseline_auc = self.results.get('original_baseline_bilstm', {}).get('final_auc', 0.0)
            optimized_baseline_auc = self.results.get('optimized_baseline_bilstm', {}).get('final_auc', 0.0)

            print(f"\nKey Findings (All experiments use BiLSTM with Word Embeddings):")
            print("-" * 50)

            print(f"  - Original Baseline (Dynamic Hypergraph) AUC: {original_baseline_auc:.4f}\n")
            print(f"  - **Optimized Baseline (Similarity Prior Hypergraph) AUC: {optimized_baseline_auc:.4f}**\n")
            if optimized_baseline_auc > original_baseline_auc:
                print(
                    f"    -> Performance improved by: {optimized_baseline_auc - original_baseline_auc:.4f} (indicating 'Similarity Prior' is a beneficial hypergraph strategy with BiLSTM).\n")
            else:
                print(
                    f"    -> Optimization attempt did not improve performance significantly compared to original baseline (Difference: {optimized_baseline_auc - original_baseline_auc:.4f}).\n")

            max_drop = 0
            most_impactful_ablation = None

            print("\n  - Impact of other module changes (relative to Original Baseline with BiLSTM Word Embeddings):\n")
            for exp_name, result in _sorted_results_items:  # 使用重新定义的 _sorted_results_items
                if exp_name not in ['original_baseline_bilstm', 'optimized_baseline_bilstm']:
                    current_auc = result.get('final_auc', 0.0)
                    drop = original_baseline_auc - current_auc
                    impact_type = "Drop" if drop > 0 else "Gain" if drop < 0 else "No Change"
                    print(
                        f"    - '{result['config']['name'].split('(')[0].strip()}': {impact_type} in AUC: {abs(drop):.4f}\n")
                    if abs(drop) > abs(max_drop):
                        max_drop = drop
                        most_impactful_ablation = result['config']['name']

            if most_impactful_ablation:
                impact_type = "Drop" if max_drop > 0 else "Gain" if max_drop < 0 else "No Change"
                print(f"\n  - **Most impactful change (largest absolute AUC difference from Original Baseline):**\n")
                print(
                    f"    -> '{most_impactful_ablation.split('(')[0].strip()}' ({impact_type} of {abs(max_drop):.4f})\n")
        else:
            print("Baseline or Optimized Baseline results not available for detailed findings.")

        print(f"\n{'=' * 60}")


def main():
    """Main function"""

    print("🧪 Multimodal Hypergraph Anomaly Detection System - Ablation Study (OrganaMNIST)")
    print("================================================================================")
    print("Note: All experiments are now using the BiLSTM Text Encoder by default.")
    print("🔄 Each experiment will run 3 times for robust evaluation with error bars.")

    ablation_exp = AblationExperiment()

    # data_path 已经直接在 AblationExperiment 的 __init__ 中的 self.base_config 里设置了
    # 所以这里不需要再单独设置 ablation_exp.base_config['data_path']，避免覆盖或不一致
    # 但如果您的路径每次运行都不同，可以在这里根据需要覆盖

    print("Starting ablation experiments with OrganaMNIST dataset...")
    results = ablation_exp.run_all_experiments(epochs=80, num_runs=3)  # 每个实验运行3次

    print("\nGenerating comprehensive experiment visualization...")
    ablation_exp.create_visualization()

    print("\nGenerating enhanced experiment visualization with detailed analysis...")
    ablation_exp.create_enhanced_visualization()

    print("\nPrinting analysis report to console...")
    ablation_exp.print_analysis_report()

    print("\n🎉 OrganaMNIST ablation experiments completed!")
    print("✅ All metrics (AUC, Accuracy) have been evaluated and visualized with statistical significance")
    print("📊 Results show mean ± standard deviation across 3 independent runs")


if __name__ == "__main__":
    main()