import os
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace as SN

# Add src to the path so nested modules can import cleanly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# 1. Imports from the codebase
from modules.agents.hetero_agent import HeteroAgent
from modules.critics.hetero_critic import HeteroCritic
from smac_hetero.role_mapper import SMACRoleMapper

def test_heterogeneous_pipeline():
    print("Initializing mock configuration...")
    args = SN(
        n_agents=3,
        n_actions=6,
        role_emb_dim=16,
        history_len=5,
        n_key=32,
        n_query=32,
        n_value=32,
        device="cpu",
        role_names=["scout", "fighter", "medic", "tank", "support"],
        roles={
            "scout": {"hidden_dim": 64},
            "fighter": {"hidden_dim": 128},
            "medic": {"hidden_dim": 96},
            "tank": {"hidden_dim": 128},
            "support": {"hidden_dim": 96}
        }
    )

    # Mock environment
    class MockUnit:
        def __init__(self, unit_type):
            self.unit_type = unit_type

    class MockEnv:
        def __init__(self):
            self.n_agents = 3
            self.marine_id = 1
            self.medivac_id = 2
            self.stalker_id = 3
            self.agents = {
                0: MockUnit(1), # fighter (Marine)
                1: MockUnit(2), # medic (Medivac)
                2: MockUnit(3)  # support (Stalker)
            }

    env = MockEnv()
    
    print("Testing SMACRoleMapper...")
    mapper = SMACRoleMapper(args.role_names)
    roles = mapper.get_roles(env, method="unit_type")
    print(f"Assigned roles: {roles}") # Expect [1, 2, 4] for fighter, medic, support
    assert roles == [1, 2, 4], f"Unexpected mapped roles: {roles}"

    input_dim = 40
    print("Initializing HeteroAgent...")
    agent = HeteroAgent(input_dim, args)

    print("Initializing hidden states...")
    hidden_states = []
    for agent_id in range(args.n_agents):
        role_name = args.role_names[roles[agent_id]]
        h_dim = args.roles[role_name]["hidden_dim"]
        hidden_states.append(torch.zeros(2, h_dim)) # batch_size=2

    print("Testing recurrent forward pass...")
    # Inputs: [batch_size * n_agents, input_dim]
    x = torch.randn(2 * args.n_agents, input_dim)
    new_hidden_states = agent(x, hidden_states, roles)
    for idx, h in enumerate(new_hidden_states):
        role_name = args.role_names[roles[idx]]
        expected_dim = args.roles[role_name]["hidden_dim"]
        print(f"Agent {idx} ({role_name}) hidden state shape: {list(h.shape)}")
        assert list(h.shape) == [2, expected_dim], f"Unexpected shape for agent {idx}"

    print("Testing communicate projections...")
    keys, values, queries = agent.communicate(new_hidden_states, roles)
    print(f"Keys shape: {list(keys.shape)}")   # Expect [2, 3, 32]
    print(f"Values shape: {list(values.shape)}") # Expect [2, 3, 32]
    print(f"Queries shape: {list(queries.shape)}") # Expect [2, 3, 32]
    assert list(keys.shape) == [2, 3, 32]
    assert list(values.shape) == [2, 3, 32]
    assert list(queries.shape) == [2, 3, 32]

    print("Testing aggregate attention and trust score gating...")
    send_prob = agent.generate_send_prob(x, roles)
    print(f"Send probability shape: {list(send_prob.shape)}") # Expect [2, 3, 3]
    assert list(send_prob.shape) == [2, 3, 3]

    send_target = torch.ones(2, 3, 3).int() # All-to-all communication
    final_q, cmb_u, incoming_msgs, trust_scores = agent.aggregate(
        queries, keys, values, new_hidden_states, send_target, roles
    )
    print(f"Final Q values shape: {list(final_q.shape)}") # Expect [2, 3, 6]
    print(f"Trust scores shape: {list(trust_scores.shape)}") # Expect [2, 3, 3]
    assert list(final_q.shape) == [2, 3, 6]
    assert list(trust_scores.shape) == [2, 3, 3]

    print("Testing HeteroCritic...")
    scheme = {
        "state": {"vshape": 50},
        "obs": {"vshape": 40}
    }
    critic = HeteroCritic(scheme, args)
    
    # Mock Batch
    class MockBatch:
        def __init__(self):
            self.batch_size = 2
            self.max_seq_length = 1
            self.device = "cpu"
        def __getitem__(self, key):
            if key == "state":
                return torch.randn(2, 1, 50)
            return None

    batch = MockBatch()
    
    # Pad hidden states
    max_h_dim = max(args.roles[name]["hidden_dim"] for name in args.role_names)
    padded_h = []
    for h in new_hidden_states:
        if h.shape[-1] < max_h_dim:
            pad_size = max_h_dim - h.shape[-1]
            padded_h.append(torch.nn.functional.pad(h, (0, pad_size)).unsqueeze(1))
        else:
            padded_h.append(h.unsqueeze(1))
    hidden_states_tensor = torch.cat(padded_h, dim=1) # [2, 3, 128]
    
    roles_tensor = torch.tensor(roles).unsqueeze(0).expand(2, -1) # [2, 3]

    q_eval = critic(
        batch, 
        t=0, 
        hidden_states=hidden_states_tensor, 
        roles=roles_tensor, 
        comm_embeddings=incoming_msgs, 
        trust_embeddings=trust_scores
    )
    print(f"Critic Q-eval shape: {list(q_eval.shape)}") # Expect [2, 1, 3, 6]
    assert list(q_eval.shape) == [2, 1, 3, 6]

    print("All heterogeneous checks passed successfully!")

if __name__ == "__main__":
    test_heterogeneous_pipeline()
