import torch
import torch.nn.functional as F
from torch import nn
from .feature_encoder import ModalityEncoder
from .similarity_builder import build_candidate_edges
from .attention_selector import score_and_filter_edges

class HyperedgeGenerator(nn.Module):
    def __init__(self, modality_dims, hidden_dim=64, top_k=10, threshold=0.5):
        super().__init__()
        self.encoder = ModalityEncoder(modality_dims, hidden_dim)
        self.top_k = top_k
        self.threshold = threshold

    def forward(self, x_modalities):
        """
        x_modalities: dict 模态名 -> tensor(batch_size, feat_dim)
        """
        # Step 1: 模态嵌入
        fused_embeddings = self.encoder(x_modalities)   # z_i ∈ R^{N×d}

        # Step 2: 相似度先验候选边
        candidates = build_candidate_edges(fused_embeddings, top_k=self.top_k)

        # Step 3: 注意力权重打分并筛选
        hyperedges = score_and_filter_edges(candidates, x_modalities, threshold=self.threshold)

        return hyperedges
    