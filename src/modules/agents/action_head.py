import torch.nn as nn

class RoleActionHead(nn.Module):
    def __init__(self, hidden_dim, n_actions):
        super().__init__()
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        return self.fc2(x)

class ActionHeadBank(nn.Module):
    def __init__(self, n_actions, roles_config, role_names):
        super().__init__()
        self.policies = nn.ModuleDict()
        for name in role_names:
            hidden_dim = roles_config[name].get("hidden_dim", 64)
            self.policies[name] = RoleActionHead(hidden_dim, n_actions)

    def forward(self, x, role_name):
        return self.policies[role_name](x)
