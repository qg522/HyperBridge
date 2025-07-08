# 文件名建议保存为：wavelet_cheb_conv.py
import torch
import torch.nn as nn

class WaveletChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=5, tau=0.5):
        """
        小波核 + 切比雪夫图卷积模块

        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            K: 切比雪夫多项式阶数
            tau: 小波核参数，用于 g(λ) = e^{-tau * λ}
        """
        super(WaveletChebConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.tau = tau

        # 卷积参数：每一阶对应一个参数矩阵
        self.theta = nn.Parameter(torch.Tensor(K, in_dim, out_dim))
        nn.init.xavier_uniform_(self.theta)

        # 计算 Chebyshev 系数 (用于逼近小波核)
        self.cheb_coeff = self._compute_chebyshev_coeff()

    def _compute_chebyshev_coeff(self):
        def wavelet_kernel(x):
            return torch.exp(-self.tau * x)  # 小波核 g(λ) = e^{-τλ}

        # 切比雪夫-高斯积分法
        coeffs = []
        for k in range(self.K):
            # 修复：确保输入是张量
            x_k = torch.cos(torch.tensor(torch.pi * (k + 0.5) / self.K))
            c_k = (2 / self.K) * wavelet_kernel(x_k)
            coeffs.append(c_k)
        return torch.stack(coeffs).float().view(-1, 1, 1)  # (K, 1, 1)

    def forward(self, x, laplacian):
        """
        x: 节点特征, shape = (N, in_dim)
        laplacian: 拉普拉斯矩阵 (稀疏 or dense), shape = (N, N)
        """
        # 确保切比雪夫系数在正确设备上
        cheb_coeff = self.cheb_coeff.to(x.device)
        
        Tx_0 = x  # T_0(x)
        Tx_1 = torch.matmul(laplacian, x)  # T_1(x)
        Tx_list = [Tx_0, Tx_1]

        for k in range(2, self.K):
            Tx_k = 2 * torch.matmul(laplacian, Tx_list[-1]) - Tx_list[-2]
            Tx_list.append(Tx_k)

        Tx_stack = torch.stack(Tx_list, dim=0)  # (K, N, in_dim)

        # 融合小波核系数
        output = torch.einsum("kni,kio,k->no", Tx_stack, self.theta, cheb_coeff.squeeze())
        return output
