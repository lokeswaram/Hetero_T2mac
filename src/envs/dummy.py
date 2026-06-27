from envs.multiagentenv import MultiAgentEnv
import numpy as np

class DummyEnv(MultiAgentEnv):
    def __init__(self, n_agents=3, n_actions=6, state_shape=20, obs_shape=15, episode_limit=10, **kwargs):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.episode_limit = episode_limit
        
        # Mock units to support StarCraft2 unit-type role mapping
        class MockUnit:
            def __init__(self, unit_type):
                self.unit_type = unit_type
        
        self.marine_id = 1
        self.medivac_id = 2
        self.stalker_id = 3
        
        # Grant units to mapper
        self.agents = {
            0: MockUnit(1),  # Marine -> maps to fighter
            1: MockUnit(2),  # Medivac -> maps to medic
            2: MockUnit(3)   # Stalker -> maps to support
        }
        
        self._steps = 0
        
    def step(self, actions):
        self._steps += 1
        reward = float(np.random.randn())
        terminated = self._steps >= self.episode_limit
        info = {}
        return reward, terminated, info
        
    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]
        
    def get_obs_agent(self, agent_id):
        return np.random.randn(self.obs_shape).astype(np.float32)
        
    def get_obs_size(self):
        return self.obs_shape
        
    def get_state(self):
        return np.random.randn(self.state_shape).astype(np.float32)
        
    def get_state_size(self):
        return self.state_shape
        
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
        
    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions
        
    def get_total_actions(self):
        return self.n_actions
        
    def reset(self):
        self._steps = 0
        
    def render(self):
        pass
        
    def close(self):
        pass
        
    def seed(self):
        return 0
        
    def save_replay(self):
        pass
