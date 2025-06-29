import torch
import torch.nn as nn

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=64, num_layers=1):
        super(BiLSTMTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        x: [B, T]，输入为词索引序列
        """
        embedded = self.embedding(x)       # [B, T, embed_dim]
        lstm_out, _ = self.lstm(embedded)  # [B, T, 2*hidden_dim]
        pooled = torch.mean(lstm_out, dim=1)  # mean pooling over time
        return self.fc(pooled)             # [B, output_dim]
