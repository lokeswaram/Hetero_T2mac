import numpy as np

class SMACRoleMapper:
    def __init__(self, role_names):
        self.role_names = role_names
        self.role_to_idx = {name: idx for idx, name in enumerate(role_names)}

    def get_roles(self, env, method="unit_type", fixed_roles=None):
        """
        Assigns roles to agents in the environment.
        method: "fixed", "random", "unit_type"
        """
        n_agents = env.n_agents
        roles = []
        
        if method == "fixed":
            if fixed_roles is not None:
                for r in fixed_roles:
                    if isinstance(r, int):
                        roles.append(r)
                    else:
                        roles.append(self.role_to_idx.get(r, 0))
            else:
                roles = [0] * n_agents
        elif method == "random":
            roles = list(np.random.randint(0, len(self.role_names), size=n_agents))
        elif method == "unit_type":
            # Map unit types using the env agents list
            # StarCraft2Env exposes self.agents containing PySC2 unit objects with .unit_type
            for agent_id in range(n_agents):
                unit = getattr(env, "agents", {}).get(agent_id, None)
                if unit is None:
                    # Fallback to scout
                    roles.append(self.role_to_idx.get("scout", 0))
                    continue
                
                utype = unit.unit_type
                
                # Compare unit_type with the ids stored in the env
                if hasattr(env, "marine_id") and utype == env.marine_id:
                    role_name = "fighter"
                elif hasattr(env, "marauder_id") and utype == env.marauder_id:
                    role_name = "fighter"
                elif hasattr(env, "medivac_id") and utype == env.medivac_id:
                    role_name = "medic"
                elif hasattr(env, "zealot_id") and utype == env.zealot_id:
                    role_name = "tank"
                elif hasattr(env, "stalker_id") and utype == env.stalker_id:
                    role_name = "support"
                else:
                    role_name = "scout"
                
                roles.append(self.role_to_idx.get(role_name, 0))
        else:
            raise ValueError(f"Unknown role assignment method: {method}")
            
        return roles
