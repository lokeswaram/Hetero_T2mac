import torch
import torch.nn as nn

class TrustEstimator(nn.Module):
    def __init__(self, role_emb_dim, history_len=5, hidden_dim=32):
        super().__init__()
        self.history_len = history_len
        self.fc = nn.Sequential(
            nn.Linear(role_emb_dim * 2 + history_len + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, sender_role_emb, receiver_role_emb, comm_history, message_usefulness):
        """
        sender_role_emb: [batch_size, n_agents, n_agents, role_emb_dim]
        receiver_role_emb: [batch_size, n_agents, n_agents, role_emb_dim]
        comm_history: [batch_size, n_agents, n_agents, history_len]
        message_usefulness: [batch_size, n_agents, n_agents, 1]
        
        Returns: trust_scores [batch_size, n_agents, n_agents]
        """
        x = torch.cat([sender_role_emb, receiver_role_emb, comm_history, message_usefulness], dim=-1)
        trust_scores = self.fc(x).squeeze(-1)
        return trust_scores
