import torch.nn as nn
import torch.nn.functional as F

class RoleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc1(x))

class EncoderBank(nn.Module):
    def __init__(self, input_dim, roles_config, role_names):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for name in role_names:
            out_dim = roles_config[name].get("hidden_dim", 64)
            self.encoders[name] = RoleEncoder(input_dim, out_dim)

    def forward(self, x, role_name):
        return self.encoders[role_name](x)
