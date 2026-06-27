import torch
import torch.nn as nn
import torch.nn.functional as F

class RoleAwareAttention(nn.Module):
    def __init__(self, query_dim, key_dim, role_emb_dim, args):
        super().__init__()
        self.args = args
        self.fc = nn.Linear(query_dim + key_dim + role_emb_dim * 2, 1)

    def forward(self, queries, keys, role_embs):
        """
        queries: [batch_size, n_agents, query_dim]
        keys: [batch_size, n_agents, key_dim]
        role_embs: [batch_size, n_agents, role_emb_dim]
        Returns: attn_weights of shape [batch_size, n_agents, n_agents]
        """
        bs, n_agents, q_dim = queries.shape
        _, _, k_dim = keys.shape
        _, _, r_dim = role_embs.shape

        q_exp = queries.unsqueeze(2).expand(-1, -1, n_agents, -1)
        k_exp = keys.unsqueeze(1).expand(-1, n_agents, -1, -1)
        r_i_exp = role_embs.unsqueeze(2).expand(-1, -1, n_agents, -1)
        r_j_exp = role_embs.unsqueeze(1).expand(-1, n_agents, -1, -1)

        concat_features = torch.cat([q_exp, k_exp, r_i_exp, r_j_exp], dim=-1)
        scores = self.fc(concat_features).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights
