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

            edge_feat = fused_features[edge]  # [K+1, H]
            weight = torch.sigmoid(edge_feat.mean())
            edge_weights.append(weight)

        H = torch.zeros(N, len(edge_list)).to(fused_features.device)
        final_weights = []
        for e_idx, (edge, w) in enumerate(zip(edge_list, edge_weights)):
            if w > self.threshold:
                for node in edge:
                    H[node, e_idx] = 1
                final_weights.append(w)

        return H, torch.tensor(final_weights).to(fused_features.device)


if __name__ == '__main__':
    # 模拟3模态输入
    x1 = torch.rand(100, 64)  # 图像
    x2 = torch.rand(100, 32)  # 时间序列
    x3 = torch.rand(100, 16)  # 文本
    model = HybridHyperedgeGenerator(num_modalities=3, input_dims=[64, 32, 16], hidden_dim=64)
    H, edge_weights = model([x1, x2, x3])
    print("H shape:", H.shape)
    print("Selected edges:", edge_weights.shape)
