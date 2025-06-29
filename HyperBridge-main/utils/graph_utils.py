import torch

def construct_incidence_matrix_sparse(num_nodes, hyperedges):
    """
    构建稀疏邻接矩阵 H (num_nodes × num_hyperedges)
    Args:
        num_nodes (int): 节点数量
        hyperedges (List[List[int]]): 每条超边所包含的节点索引
    Returns:
        H (torch.sparse.FloatTensor): 稀疏邻接矩阵
    """
    row_idx = []
    col_idx = []
    for j, edge in enumerate(hyperedges):
        for i in edge:
            row_idx.append(i)
            col_idx.append(j)
    
    indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
    values = torch.ones(len(row_idx), dtype=torch.float32)
    H = torch.sparse_coo_tensor(indices, values, size=(num_nodes, len(hyperedges)))
    return H.coalesce()

def compute_Dv_De_sparse(H):
    """
    计算稀疏超图的 Dv 和 De（对角线向量形式）
    Args:
        H (sparse tensor): 稀疏邻接矩阵
    Returns:
        Dv (Tensor): [N]
        De (Tensor): [E]
    """
    De = torch.sparse.sum(H, dim=0).to_dense()  # 每列求和：超边度
    Dv = torch.sparse.sum(H, dim=1).to_dense()  # 每行求和：顶点度
    return Dv, De

def compute_normalized_laplacian_sparse(H, Dv, De, epsilon=1e-8):
    """
    计算稀疏归一化超图拉普拉斯矩阵：
    L = Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}
    Returns:
        L (Tensor): 稠密形式的对称拉普拉斯矩阵（可用于 Chebyshev）
    """
    N, E = H.shape
    Dv_inv_sqrt = torch.diag(1.0 / (Dv + epsilon).sqrt())
    De_inv = torch.diag(1.0 / (De + epsilon))

    H_dense = H.to_dense()
    HT_dense = H_dense.T

    L = Dv_inv_sqrt @ H_dense @ De_inv @ HT_dense @ Dv_inv_sqrt
    return L
