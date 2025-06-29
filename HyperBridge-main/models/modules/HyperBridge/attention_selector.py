import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHyperedgeSelector(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, threshold=0.5):
        """
        input_dims: dict，键为模态名，值为每个模态的输入维度（如 {'image': 32, 'text': 64}）
        hidden_dim: MLP 中间维度
        threshold: 用于保留超边的分数阈值
        """
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.threshold = threshold

        # 每个模态一个 MLP projector
        self.projectors = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dims[m], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # 输出一个评分
            ) for m in self.modalities
        })

        # 可学习的模态注意力权重
        self.alpha = nn.Parameter(torch.ones(len(self.modalities)))  # [M]
    
    def forward(self, hyperedges, features_dict):
        """
        参数：
            hyperedges: list of list，每个超边是若干节点索引
            features_dict: dict，模态名 -> 特征张量 [N, d_m]
        返回：
            selected_hyperedges: list of list，筛选后的超边集合
        """
        scores = []
        for e in hyperedges:
            e_score = 0.0
            for i, m in enumerate(self.modalities):
                modal_feats = features_dict[m][e]  # [K, d_m]
                pooled = modal_feats.mean(dim=0)   # [d_m]
                score = self.projectors[m](pooled) # [1]
                e_score += F.softmax(self.alpha, dim=0)[i] * score
            scores.append(torch.sigmoid(e_score).item())

        # 筛选
        selected_hyperedges = [e for e, s in zip(hyperedges, scores) if s > self.threshold]
        return selected_hyperedges, scores
