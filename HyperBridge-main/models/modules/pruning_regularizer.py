# models/modules/pruning_regularizer.py
import torch
import torch.nn as nn

class SpectralCutRegularizer(nn.Module):
    def __init__(self, use_rayleigh=False, reduction='mean'):
        """
        :param use_rayleigh: 是否使用 Rayleigh 商归一化
        :param reduction: 对 loss 是否取 mean（或 sum）
        """
        super(SpectralCutRegularizer, self).__init__()
        self.use_rayleigh = use_rayleigh
        self.reduction = reduction

    def forward(self, F, H, Dv, De):
        """
        :param F: 节点嵌入矩阵 [N, d]
        :param H: 超图关联矩阵 [N, E]
        :param Dv: 节点度矩阵，对角阵 [N, N]
        :param De: 超边度矩阵，对角阵 [E, E]
        :return: 谱剪切正则项损失值
        """
        device = F.device

        # 确保所有张量在同一设备上
        H = H.to(device)
        Dv = Dv.to(device)
        De = De.to(device)

        # Dv^{-1/2}
        Dv_inv_sqrt = torch.diag(torch.pow(torch.diag(Dv) + 1e-8, -0.5)).to(device)
        # De^{-1}
        De_inv = torch.diag(1.0 / (torch.diag(De) + 1e-8)).to(device)

        # 构造超图拉普拉斯算子
        L = torch.eye(Dv.shape[0]).to(device) - Dv_inv_sqrt @ H @ De_inv @ H.t() @ Dv_inv_sqrt

        # 计算 Tr(F^T L F)
        trace_term = torch.trace(F.t() @ L @ F)

        if self.use_rayleigh:
            denominator = torch.trace(F.t() @ F)
            loss = trace_term / (denominator + 1e-8)
        else:
            loss = trace_term

        if self.reduction == 'mean':
            loss = loss / F.shape[0]

        return loss
