import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.image_encoder import CNNImageEncoder
from .modules.text_encoder import BiLSTMTextEncoder
from .modules.hyper_generator import HybridHyperedgeGenerator
from .modules.wavelet_cheb_conv import WaveletChebConv
from .modules.pruning_regularizer import SpectralCutRegularizer

class HyperBridge(nn.Module):
    def __init__(self, config):
        super(HyperBridge, self).__init__()
        
        # 根据实际的编码器类名和参数进行修正
        self.image_encoder = CNNImageEncoder(
            in_channels=config.get('img_channels', 3), 
            hidden_dim=config['hidden'], 
            out_dim=config['hidden']
        )
        self.text_encoder = BiLSTMTextEncoder(
            vocab_size=config['vocab_size'], 
            embed_dim=config.get('embed_dim', 128),
            hidden_dim=config.get('text_hidden', 64),
            output_dim=config['hidden']
        )
        self.signal_encoder = nn.Sequential(
            nn.Linear(config['sig_in'], config['hidden']),
            nn.ReLU(),
            nn.Linear(config['hidden'], config['hidden'])
        )

        self.hyperedge_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[config['hidden'], config['hidden'], config['sig_in']],  # 修正：这些应该是编码后的维度
            hidden_dim=config['hidden'],
            top_k=config.get('top_k', 10),
            threshold=config.get('thresh', 0.5)
        )

        # 修正WaveletChebConv的参数名
        self.gnn_layer = WaveletChebConv(
            in_dim=config['hidden'], 
            out_dim=config['hidden'], 
            K=config.get('K', 5),
            tau=config.get('tau', 0.5)
        )

        self.classifier = nn.Linear(config['hidden'], config['n_class'])
        self.regularizer = SpectralCutRegularizer()
        
        # 添加特征融合层
        self.feature_fusion = nn.Linear(config['hidden'] * 3, config['hidden'])

    def forward(self, x_img, x_txt, x_sig):
        # 各模态编码
        z_img = self.image_encoder(x_img)      # [B, hidden]
        z_txt = self.text_encoder(x_txt)       # [B, hidden]  
        z_sig = self.signal_encoder(x_sig)     # [B, hidden]

        # 生成超图结构
        H, edge_weights = self.hyperedge_generator([z_img, z_txt, z_sig])

        # 特征融合
        fused = torch.cat([z_img, z_txt, z_sig], dim=-1)  # [B, hidden*3]
        fused = self.feature_fusion(fused)  # [B, hidden]

        # 构建拉普拉斯矩阵用于WaveletChebConv
        # H是关联矩阵 [N, E]，需要转换为拉普拉斯矩阵
        laplacian = self._compute_hypergraph_laplacian(H)
        
        # GNN层处理
        out = self.gnn_layer(fused, laplacian)  # [B, hidden]
        logits = self.classifier(out)           # [B, n_class]

        # 计算正则化损失 - 需要节点度和超边度矩阵
        Dv, De = self._compute_degree_matrices(H)
        reg_loss = self.regularizer(out, H, Dv, De)
        
        return logits, reg_loss
    
    def _compute_hypergraph_laplacian(self, H):
        """
        从关联矩阵H计算超图拉普拉斯矩阵
        H: [N, E] 关联矩阵
        返回: [N, N] 拉普拉斯矩阵
        """
        device = H.device
        N, E = H.shape
        
        # 计算节点度和超边度
        node_degrees = H.sum(dim=1)  # [N]
        edge_degrees = H.sum(dim=0)  # [E]
        
        # 避免除零
        node_degrees = torch.clamp(node_degrees, min=1e-8)
        edge_degrees = torch.clamp(edge_degrees, min=1e-8)
        
        # 构建度矩阵的逆
        Dv_inv_sqrt = torch.diag(torch.pow(node_degrees, -0.5))  # [N, N]
        De_inv = torch.diag(1.0 / edge_degrees)  # [E, E]
        
        # 计算归一化的邻接矩阵: A = H * De^(-1) * H^T
        A = H @ De_inv @ H.t()  # [N, N]
        
        # 归一化: D^(-1/2) * A * D^(-1/2)
        A_norm = Dv_inv_sqrt @ A @ Dv_inv_sqrt
        
        # 拉普拉斯矩阵: L = I - A_norm
        I = torch.eye(N, device=device)
        L = I - A_norm
        
        return L
    
    def _compute_degree_matrices(self, H):
        """
        计算节点度矩阵和超边度矩阵
        H: [N, E] 关联矩阵
        返回: (Dv, De) 度矩阵
        """
        device = H.device
        
        # 节点度和超边度
        node_degrees = H.sum(dim=1)  # [N]
        edge_degrees = H.sum(dim=0)  # [E]
        
        # 构建对角矩阵
        Dv = torch.diag(node_degrees).to(device)  # [N, N]
        De = torch.diag(edge_degrees).to(device)  # [E, E]
        
        return Dv, De
