import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RoleAwareCritic(nn.Module):
    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.num_roles = len(args.role_names)
        self.output_type = "q"

        # Learnable role embedding specifically for the critic
        self.role_embedding = nn.Embedding(self.num_roles, args.role_emb_dim)

        # Calculate dynamic input shape:
        # 1. global state
        input_shape = scheme["state"]["vshape"]
        # 2. agent hidden states (max_h_dim)
        max_h_dim = max(args.roles[name].get("hidden_dim", 64) for name in args.role_names)
        input_shape += max_h_dim
        # 3. role embedding
        input_shape += args.role_emb_dim
        # 4. communication embedding (incoming messages, size n_value)
        input_shape += args.n_value
        # 5. trust embedding (trust scores, size n_agents)
        input_shape += args.n_agents

        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.n_actions)

    def forward(self, batch, t=None, hidden_states=None, roles=None, comm_embeddings=None, trust_embeddings=None):
        inputs = self._build_inputs(batch, t, hidden_states, roles, comm_embeddings, trust_embeddings)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None, hidden_states=None, roles=None, comm_embeddings=None, trust_embeddings=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        
        # 1. State
        state = batch["state"][:, ts] # [bs, max_t, state_dim]
        state_rep = state.unsqueeze(2).repeat(1, 1, self.n_agents, 1) # [bs, max_t, n_agents, state_dim]
        inputs.append(state_rep)
        
        # 2. Hidden states
        if hidden_states is not None:
            if t is not None:
                h_input = hidden_states.unsqueeze(1) # [bs, 1, n_agents, max_h_dim]
            else:
                h_input = hidden_states # [bs, max_t, n_agents, max_h_dim]
            inputs.append(h_input)
        else:
            # Fallback mock/zeros
            max_h_dim = max(self.args.roles[name].get("hidden_dim", 64) for name in self.args.role_names)
            inputs.append(th.zeros(bs, max_t, self.n_agents, max_h_dim, device=batch.device))
            
        # 3. Role embeddings
        if roles is not None:
            if t is not None:
                r_input = roles.unsqueeze(1) # [bs, 1, n_agents]
            else:
                r_input = roles # [bs, max_t, n_agents]
            role_embs = self.role_embedding(r_input.long()) # [bs, max_t, n_agents, role_emb_dim]
            inputs.append(role_embs)
        else:
            inputs.append(th.zeros(bs, max_t, self.n_agents, self.args.role_emb_dim, device=batch.device))
            
        # 4. Communication embeddings
        if comm_embeddings is not None:
            if t is not None:
                c_input = comm_embeddings.unsqueeze(1)
            else:
                c_input = comm_embeddings
            inputs.append(c_input)
        else:
            inputs.append(th.zeros(bs, max_t, self.n_agents, self.args.n_value, device=batch.device))
            
        # 5. Trust embeddings
        if trust_embeddings is not None:
            if t is not None:
                tr_input = trust_embeddings.unsqueeze(1)
            else:
                tr_input = trust_embeddings
            inputs.append(tr_input)
        else:
            inputs.append(th.zeros(bs, max_t, self.n_agents, self.n_agents, device=batch.device))
            
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
