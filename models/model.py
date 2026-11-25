import torch.nn as nn
import torch.nn.functional as F

class LSTMContrastiveClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dims=[128, 64], projection_dim=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden_dims[0], hidden_dims[1], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj_head = nn.Sequential(nn.Linear(2 * hidden_dims[1], 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, projection_dim))
        self.classifier = nn.Linear(projection_dim, 1)

    def forward(self, x, return_embedding=False):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        out = self.global_pool(out).squeeze(-1)
        embeddings = self.proj_head(out)
        embeddings = F.normalize(embeddings, dim=1)
        logits = self.classifier(embeddings).squeeze(1)
        if return_embedding:
            return (embeddings, logits)
        return logits
