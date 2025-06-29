import torch
import torch.nn as nn

class ModalityEncoder(nn.Module):
    def __init__(self, modality_dims, hidden_dim=128):
        """
        参数：
        modality_dims: dict，键为模态名，值为输入维度，如 {'image': 128, 'text': 64}
        hidden_dim: 输出的共享嵌入维度
        """
        super(ModalityEncoder, self).__init__()
        self.encoders = nn.ModuleDict()

        for modality, input_dim in modality_dims.items():
            self.encoders[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, x_dict):
        """
        参数：
        x_dict: dict，键为模态名，值为该模态的特征张量，形状为 [N, D_in]
        返回：
        fused_embedding: [N, hidden_dim * M] 融合后的特征
        """
        z_list = []

        for modality, encoder in self.encoders.items():
            z = encoder(x_dict[modality])  # [N, hidden_dim]
            z_list.append(z)

        fused_embedding = torch.cat(z_list, dim=-1)  # [N, hidden_dim * M]
        return fused_embedding