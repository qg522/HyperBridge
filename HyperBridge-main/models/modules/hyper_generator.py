import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class HybridHyperedgeGenerator(nn.Module):
    def __init__(self, num_modalities, input_dims, hidden_dim, top_k=10, threshold=0.5):
        super(HybridHyperedgeGenerator, self).__init__()
        self.num_modalities = num_modalities
        self.top_k = top_k
        self.threshold = threshold

        self.modal_mlps = ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for input_dim in input_dims
        ])

        self.attn_weights = nn.Parameter(torch.ones(num_modalities))
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)  # 修复：输入维度应该是hidden_dim而不是num_modalities * hidden_dim

    def forward(self, x_list):
        """
        x_list: List of [N, D_m] tensors, one per modality
        """
        modality_features = []
        for i in range(self.num_modalities):
            modality_features.append(self.modal_mlps[i](x_list[i]))  # [N, H]

        # 方法1: 加权求和融合
        modality_stack = torch.stack(modality_features, dim=0)  # [M, N, H]
        attn_scores = F.softmax(self.attn_weights, dim=0).view(-1, 1, 1)
        fused_features = torch.sum(attn_scores * modality_stack, dim=0)  # [N, H]
        
        # 方法2: 连接融合（注释掉的备选方案）
        # fused_features = torch.cat(modality_features, dim=1)  # [N, M*H]
        
        fused_features = self.fusion_proj(fused_features)  # [N, H]
        normed_feat = F.normalize(fused_features, p=2, dim=1)
        similarity = torch.matmul(normed_feat, normed_feat.T)  # [N, N]

        edge_list = []
        edge_weights = []
        N = fused_features.size(0)

        for i in range(N):
            sim_row = similarity[i]
            neighbors = torch.topk(sim_row, self.top_k + 1, largest=True).indices  # +1 for self
            neighbors = neighbors[neighbors != i][:self.top_k]  # remove self
            edge = [i] + neighbors.tolist()
            edge_list.append(edge)

            # 改进的超边打分机制 - 更贴近公式 w_e = σ(∑_m α_m · MLP_m(·))
            edge_feat_per_modality = []
            for m in range(self.num_modalities):
                # 对每个模态，取超边上节点的特征并通过对应的MLP
                edge_nodes_feat = x_list[m][edge]  # [K+1, D_m]
                edge_modal_feat = self.modal_mlps[m](edge_nodes_feat)  # [K+1, H]
                edge_feat_per_modality.append(edge_modal_feat)
            
            # 堆叠所有模态特征 [M, K+1, H]
            stacked_edge_feats = torch.stack(edge_feat_per_modality, dim=0)
            
            # 应用注意力权重进行模态融合
            attn_scores = F.softmax(self.attn_weights, dim=0).view(-1, 1, 1)  # [M, 1, 1]
            weighted_edge_feats = torch.sum(attn_scores * stacked_edge_feats, dim=0)  # [K+1, H]
            
            # 聚合超边上所有节点的特征 (平均池化)
            aggregated_edge_feat = weighted_edge_feats.mean(dim=0)  # [H]
            
            # 通过融合投影层和sigmoid得到最终权重
            weight = torch.sigmoid(self.fusion_proj(aggregated_edge_feat).mean())  # 标量权重
            edge_weights.append(weight)

        H = torch.zeros(N, len(edge_list)).to(fused_features.device)
        final_weights = []
        for e_idx, (edge, w) in enumerate(zip(edge_list, edge_weights)):
            if w > self.threshold:
                for node in edge:
                    H[node, e_idx] = 1
                final_weights.append(w)

        return H, torch.tensor(final_weights).to(fused_features.device)