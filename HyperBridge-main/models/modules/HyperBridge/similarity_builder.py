import torch
import torch.nn.functional as F

class SimilarityHyperedgeBuilder:
    def __init__(self, top_k=10):
        """
        top_k: 每个节点选取的最相似邻居数量，用于构造候选超边
        """
        self.top_k = top_k

    def build(self, embeddings):
        """
        参数：
        embeddings: [N, d] 融合后的节点嵌入
        返回：
        candidate_edges: list of list，每个元素是某个节点构成的超边（由自身及 top-k 相似节点组成）
        """
        N = embeddings.size(0)
        # 对所有节点做归一化
        normed = F.normalize(embeddings, dim=1)  # [N, d]
        # 相似度矩阵 [N, N]
        sim_matrix = torch.matmul(normed, normed.T)

        candidate_edges = []
        for i in range(N):
            sim_scores = sim_matrix[i]  # 与所有节点的相似度
            sim_scores[i] = -1e9  # 不包括自己
            topk_indices = torch.topk(sim_scores, self.top_k).indices  # [top_k]

            hyperedge = [i] + topk_indices.tolist()
            candidate_edges.append(hyperedge)

        return candidate_edges
