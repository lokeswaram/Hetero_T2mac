import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.role_encoder import EncoderBank
from modules.agents.role_gru import GRUBank
from modules.agents.action_head import ActionHeadBank
from modules.agents.communication import Communication

class HeteroAgent(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.role_names = args.role_names
        self.num_roles = len(self.role_names)
        
        # Learnable Role Embeddings
        self.role_embeddings = nn.Embedding(self.num_roles, args.role_emb_dim)
        
        # Banks
        self.encoder_bank = EncoderBank(input_dim, args.roles, self.role_names)
        self.gru_bank = GRUBank(args.roles, self.role_names)
        self.policy_bank = ActionHeadBank(args.n_actions, args.roles, self.role_names)
        
        # Communication module
        self.communication = Communication(input_dim, args)

    @property
    def msg_proj1(self):
        return self.communication.msg_proj1

    @property
    def msg_proj2(self):
        return self.communication.msg_proj2

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
        return self.communication.communicate(hidden_states, roles)

    def generate_send_prob(self, obs, roles):
        return self.communication.generate_send_prob(obs, roles)

    def aggregate(self, query, key, value, hidden_states, send_target, roles, comm_history=None, past_usefulness=None):
        return self.communication.aggregate(
            query, key, value, hidden_states, send_target, roles,
            role_embeddings=self.role_embeddings,
            action_head_bank=self.policy_bank,
            comm_history=comm_history,
            past_usefulness=past_usefulness
        )

    def init_hidden(self):
        return torch.zeros(1, self.args.roles[self.role_names[0]].get("hidden_dim", 64))
