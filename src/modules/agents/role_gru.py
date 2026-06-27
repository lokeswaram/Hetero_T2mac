import torch.nn as nn

class RoleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, hidden):
        return self.rnn(x, hidden)

class GRUBank(nn.Module):
    def __init__(self, roles_config, role_names):
        super().__init__()
        self.grus = nn.ModuleDict()
        for name in role_names:
            hidden_dim = roles_config[name].get("hidden_dim", 64)
            self.grus[name] = RoleGRU(hidden_dim, hidden_dim)

    def forward(self, x, hidden, role_name):
        return self.grus[role_name](x, hidden)
