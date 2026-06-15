import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMContrastiveClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], projection_dim=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden_dims[0], hidden_dims[1], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj_head = nn.Sequential(
            nn.Linear(2 * hidden_dims[1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, projection_dim)
        )
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
            return embeddings, logits
        return logits

class FocalContrastiveLoss(nn.Module):
    def __init__(self, gamma=2.0, temp=0.1):
        super().__init__()
        self.gamma = gamma
        self.temp = temp
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, a, p, n):
        pos_sim = F.cosine_similarity(a, p) / self.temp
        neg_sim = F.cosine_similarity(a, n) / self.temp
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.ce(logits, labels)
        pt = torch.exp(-loss)
        return ((1 - pt) ** self.gamma * loss).mean()

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Fully supervised contrastive loss (InfoNCE-style)
    """
    device = embeddings.device
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # positive mask
    
    sim = torch.matmul(embeddings, embeddings.T) / temperature  
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - logits_max.detach()  # numerical stability

    exp_sim = torch.exp(sim) * (1 - torch.eye(labels.size(0), device=device))  # remove self-sim
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) - 1 + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss