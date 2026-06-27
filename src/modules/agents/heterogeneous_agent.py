import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterogeneousAgent(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.role_names = args.role_names
        self.num_roles = len(self.role_names)
        
        # Learnable Role Embeddings
        self.role_embeddings = nn.Embedding(self.num_roles, args.role_emb_dim)
        
        # Banks
        from modules.agents.role_encoder import EncoderBank
        from modules.agents.role_gru import GRUBank
        from modules.agents.role_policy import PolicyHeadBank
        
        self.encoder_bank = EncoderBank(input_dim, args.roles, self.role_names)
        self.gru_bank = GRUBank(args.roles, self.role_names)
        self.policy_bank = PolicyHeadBank(args.n_actions, args.roles, self.role_names)
        
        # Projections to communication dimension
        self.key_projs = nn.ModuleDict()
        self.query_projs = nn.ModuleDict()
        self.value_projs = nn.ModuleDict()
        for name in self.role_names:
            h_dim = args.roles[name].get("hidden_dim", 64)
            self.key_projs[name] = nn.Linear(h_dim, args.n_key)
            self.query_projs[name] = nn.Linear(h_dim, args.n_query)
            self.value_projs[name] = nn.Linear(h_dim, args.n_value)
            
        # Role-Aware Attention
        from modules.role_aware_attention import RoleAwareAttention
        self.role_attn = RoleAwareAttention(args.n_query, args.n_key, args.role_emb_dim, args)
        
        # Combined projection back to hidden_dim for each role
        self.combine_projs = nn.ModuleDict()
        for name in self.role_names:
            h_dim = args.roles[name].get("hidden_dim", 64)
            self.combine_projs[name] = nn.Linear(args.n_value + h_dim, h_dim)
            
        # Sender/Gate probability estimation per role
        self.msg_proj1 = nn.ModuleDict()
        self.msg_proj2 = nn.ModuleDict()
        for name in self.role_names:
            h_dim = args.roles[name].get("hidden_dim", 64)
            self.msg_proj1[name] = nn.Linear(input_dim, h_dim)
            self.msg_proj2[name] = nn.Linear(h_dim, self.n_agents)
            
        # Trust Estimator
        from modules.trust_estimator import TrustEstimator
        self.trust_estimator = TrustEstimator(args.role_emb_dim, getattr(args, "history_len", 5))

    def forward(self, x, hidden_states, roles):
        """
        x: [bs * n_agents, input_dim]
        hidden_states: list of length n_agents, each shape [bs, hidden_dim_i]
        roles: list/tensor of size n_agents
        """
        bs = int(x.shape[0] / self.n_agents)
        new_hidden_states = []
        
        for agent_id in range(self.n_agents):
            role_idx = roles[agent_id].item() if hasattr(roles[agent_id], "item") else roles[agent_id]
            role_name = self.role_names[role_idx]
            
            # Agent obs
            agent_obs = x[agent_id * bs : (agent_id + 1) * bs]
            
            # Encoder
            enc_out = self.encoder_bank(agent_obs, role_name)
            
            # GRU
            h_in = hidden_states[agent_id]
            h_out = self.gru_bank(enc_out, h_in, role_name)
            new_hidden_states.append(h_out)
            
        return new_hidden_states

    def q_without_communication(self, hidden_states, roles):
        qs = []
        for agent_id in range(self.n_agents):
            role_idx = roles[agent_id].item() if hasattr(roles[agent_id], "item") else roles[agent_id]
            role_name = self.role_names[role_idx]
            h = hidden_states[agent_id]
            q = self.policy_bank(h, role_name)
            qs.append(q.unsqueeze(1))
        return torch.cat(qs, dim=1) # [bs, n_agents, n_actions]

    def communicate(self, hidden_states, roles):
        keys = []
        values = []
        queries = []
        for agent_id in range(self.n_agents):
            role_idx = roles[agent_id].item() if hasattr(roles[agent_id], "item") else roles[agent_id]
            role_name = self.role_names[role_idx]
            h = hidden_states[agent_id]
            
            k = self.key_projs[role_name](h).unsqueeze(1)
            v = self.value_projs[role_name](h).unsqueeze(1)
            q = self.query_projs[role_name](h).unsqueeze(1)
            
            keys.append(k)
            values.append(v)
            queries.append(q)
            
        return torch.cat(keys, dim=1), torch.cat(values, dim=1), torch.cat(queries, dim=1)

    def generate_send_prob(self, obs, roles):
        bs = int(obs.shape[0] / self.n_agents)
        probs = []
        for agent_id in range(self.n_agents):
            role_idx = roles[agent_id].item() if hasattr(roles[agent_id], "item") else roles[agent_id]
            role_name = self.role_names[role_idx]
            agent_obs = obs[agent_id * bs : (agent_id + 1) * bs]
            
            f = F.relu(self.msg_proj1[role_name](agent_obs))
            s = self.msg_proj2[role_name](f)
            prob = F.softmax(s, dim=-1).unsqueeze(1)
            probs.append(prob)
        return torch.cat(probs, dim=1) # [bs, n_agents, n_agents]

    def aggregate(self, query, key, value, hidden_states, send_target, roles, comm_history=None, past_usefulness=None):
        bs = query.shape[0]
        
        # 1. Role Embeddings lookup
        roles_tensor = torch.tensor(roles, device=query.device) # [n_agents]
        role_embs = self.role_embeddings(roles_tensor).unsqueeze(0).repeat(bs, 1, 1) # [bs, n_agents, role_emb_dim]
        
        # 2. Role-Aware Attention
        attn_weights = self.role_attn(query, key, role_embs) # [bs, n_agents, n_agents]
        
        # 3. Trust Estimator
        r_i_exp = role_embs.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        r_j_exp = role_embs.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        
        if comm_history is None:
            comm_history = torch.zeros(bs, self.n_agents, self.n_agents, self.args.history_len, device=query.device)
        if past_usefulness is None:
            past_usefulness = torch.zeros(bs, self.n_agents, self.n_agents, 1, device=query.device)
            
        trust_scores = self.trust_estimator(r_j_exp, r_i_exp, comm_history, past_usefulness) # [bs, n_agents, n_agents]
        
        gated_attn = attn_weights * send_target.float() * trust_scores
        
        attn_applied = torch.bmm(gated_attn, value) # [bs, n_agents, n_value]
        
        final_qs = []
        for agent_id in range(self.n_agents):
            role_idx = roles[agent_id].item() if hasattr(roles[agent_id], "item") else roles[agent_id]
            role_name = self.role_names[role_idx]
            
            h = hidden_states[agent_id]
            msg_rec = attn_applied[:, agent_id]
            
            combined = torch.cat([msg_rec, h], dim=-1)
            combined_projected = F.relu(self.combine_projs[role_name](combined))
            
            q = self.policy_bank(combined_projected, role_name)
            final_qs.append(q.unsqueeze(1))
            
        final_q = torch.cat(final_qs, dim=1)
        
        evidence = torch.clamp(final_q, 0, torch.inf)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        cmb_u = self.args.n_actions / S
        
        return final_q, cmb_u, attn_applied, trust_scores

    def init_hidden(self):
        # Fallback method, not used directly in Heterogeneous mode
        # as it requires roles info. We initialize hidden states in the controller.
        return torch.zeros(1, self.args.roles[self.role_names[0]].get("hidden_dim", 64))
