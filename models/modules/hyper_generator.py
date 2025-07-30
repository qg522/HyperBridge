import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class HybridHyperedgeGenerator(nn.Module):
    """
    混合超边生成器 - 基于多模态特征相似性和注意力机制生成动态超图
    
    该模块实现了论文中的核心超图构建算法：
    1. 多模态特征编码：为每个模态学习独立的特征表示
    2. 注意力融合：通过可学习权重融合不同模态信息
    3. 相似性计算：计算节点间的余弦相似度
    4. 超边生成：基于top-k邻居和阈值筛选构建超边
    5. 超边打分：根据模态融合特征计算超边权重
    """
    
    def __init__(self, num_modalities, input_dims, hidden_dim, top_k=8, threshold=0.5):
        """
        初始化混合超边生成器
        
        Args:
            num_modalities (int): 模态数量（如图像+文本=2）
            input_dims (list): 每个模态的输入维度列表，如[2048, 100]
            hidden_dim (int): 隐藏层维度，所有模态映射到统一的表示空间
            top_k (int): 每个节点选择的最近邻数量，用于构建超边
            threshold (float): 超边权重阈值，低于此值的超边会被过滤
        """
        super(HybridHyperedgeGenerator, self).__init__()
        self.num_modalities = num_modalities  # 模态数量
        self.top_k = top_k                   # 近邻数量
        self.threshold = threshold           # 权重阈值
        
        # 为每个模态构建独立的MLP编码器
        # 将不同维度的模态特征映射到统一的hidden_dim空间
        self.modal_mlps = ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),  # 第一层：维度对齐
                nn.ReLU(),                         # 激活函数
                nn.Linear(hidden_dim, hidden_dim)  # 第二层：特征提取
            ) for input_dim in input_dims  # 为每个模态创建一个MLP
        ])
        
        # 可学习的模态注意力权重，用于融合不同模态
        # 初始化为全1，训练过程中学习每个模态的重要性
        self.attn_weights = nn.Parameter(torch.ones(num_modalities))
        
        # 融合投影层：将多模态融合特征进一步变换
        # 用于最终的超边权重计算
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_list):
        """
        前向传播：根据多模态输入生成超图结构
        
        整体流程：
        1. 多模态特征编码 - 将每个模态通过对应MLP编码
        2. 注意力融合 - 使用可学习权重融合多模态特征
        3. 相似度计算 - 计算节点间余弦相似度矩阵
        4. 超边构建 - 为每个节点找top-k邻居构成超边
        5. 超边打分 - 计算每条超边的重要性权重
        6. 阈值过滤 - 移除权重过低的超边
        
        Args:
            x_list (list): 长度为num_modalities的张量列表
                         每个张量形状为[N, D_m]，N是节点数，D_m是第m个模态的特征维度
                         例如：[图像特征[N,2048], 文本特征[N,100]]
                         
        Returns:
            H (torch.Tensor): 超图关联矩阵，形状[N, E]
                            H[i,j]=1表示节点i属于超边j，否则为0
            edge_weights (torch.Tensor): 超边权重向量，形状[E]
                                       每个值表示对应超边的重要性
        """
        # ===================== 步骤1: 多模态特征编码 =====================
        modality_features = []
        for i in range(self.num_modalities):
            # 将第i个模态的特征通过对应的MLP编码到统一空间
            encoded_feat = self.modal_mlps[i](x_list[i])  # [N, hidden_dim]
            modality_features.append(encoded_feat)
        
        # ===================== 步骤2: 注意力融合机制 =====================
        # 将所有模态特征堆叠：[M, N, H] (M=模态数, N=节点数, H=隐藏维度)
        modality_stack = torch.stack(modality_features, dim=0)  
        
        # 计算模态注意力权重：将原始权重通过softmax归一化
        attn_scores = F.softmax(self.attn_weights, dim=0).view(-1, 1, 1)  # [M, 1, 1]
        
        # 加权融合：每个模态按其注意力权重进行加权求和
        fused_features = torch.sum(attn_scores * modality_stack, dim=0)  # [N, H]
        
        # ===================== 步骤3: 节点相似度计算 =====================
        # 通过融合投影层进一步变换特征
        fused_features = self.fusion_proj(fused_features)  # [N, H]
        
        # L2归一化：使特征向量长度为1，便于计算余弦相似度
        normed_feat = F.normalize(fused_features, p=2, dim=1)  # [N, H]
        
        # 计算余弦相似度矩阵：S[i,j] = cos(v_i, v_j)
        similarity = torch.matmul(normed_feat, normed_feat.T)  # [N, N]

        # ===================== 步骤4: 超边构建 =====================
        edge_list = []      # 存储所有超边的节点列表
        edge_weights = []   # 存储所有超边的权重
        N = fused_features.size(0)  # 节点总数

        # 为每个节点构建以其为中心的超边
        for i in range(N):
            # 获取节点i与所有其他节点的相似度
            sim_row = similarity[i]  # [N]
            
            # 找到相似度最高的top_k+1个节点（+1是因为包含自己）
            neighbors = torch.topk(sim_row, self.top_k + 1, largest=True).indices
            
            # 移除自己，保留top_k个最相似的邻居
            neighbors = neighbors[neighbors != i][:self.top_k]
            
            # 构建超边：中心节点 + 其top_k邻居
            edge = [i] + neighbors.tolist()  # 超边包含的所有节点
            edge_list.append(edge)

            # ===================== 步骤5: 超边权重计算 =====================
            # 实现论文公式：w_e = σ(∑_m α_m · MLP_m(x_e^m))
            # 其中x_e^m表示超边e中所有节点在第m个模态的特征
            
            edge_feat_per_modality = []  # 存储每个模态在当前超边上的特征
            
            # 对每个模态分别处理
            for m in range(self.num_modalities):
                # 提取超边上所有节点在第m个模态的原始特征
                edge_nodes_feat = x_list[m][edge]  # [K+1, D_m] K=top_k
                
                # 通过第m个模态的MLP编码这些特征
                edge_modal_feat = self.modal_mlps[m](edge_nodes_feat)  # [K+1, H]
                edge_feat_per_modality.append(edge_modal_feat)
            
            # 将所有模态的特征堆叠：[M, K+1, H]
            stacked_edge_feats = torch.stack(edge_feat_per_modality, dim=0)
            
            # 应用模态注意力权重：α_m · x_e^m
            attn_scores = F.softmax(self.attn_weights, dim=0).view(-1, 1, 1)  # [M, 1, 1]
            weighted_edge_feats = torch.sum(attn_scores * stacked_edge_feats, dim=0)  # [K+1, H]
            
            # 聚合超边上所有节点的特征（平均池化）
            aggregated_edge_feat = weighted_edge_feats.mean(dim=0)  # [H]
            
            # 通过融合投影层和sigmoid激活得到最终权重
            # σ(MLP(∑_m α_m · x_e^m))
            weight = torch.sigmoid(self.fusion_proj(aggregated_edge_feat).mean())  # 标量权重
            edge_weights.append(weight)

        # ===================== 步骤6: 构建最终超图 =====================
        # 初始化超图关联矩阵H，H[i,j]=1表示节点i属于超边j
        H = torch.zeros(N, len(edge_list)).to(fused_features.device)  # [N, E]
        final_weights = []  # 过滤后的超边权重
        
        # 遍历所有超边，应用阈值过滤
        for e_idx, (edge, w) in enumerate(zip(edge_list, edge_weights)):
            # 只保留权重大于阈值的超边
            if w > self.threshold:
                # 在关联矩阵中标记该超边包含的所有节点
                for node in edge:
                    H[node, e_idx] = 1
                final_weights.append(w)

        # 返回超图关联矩阵和对应的权重向量
        return H, torch.tensor(final_weights).to(fused_features.device)