# Changelog: Heterogeneous T2MAC (H-T2MAC) Extension

All notable architectural adjustments and codebase upgrades for the heterogeneous multi-agent communication framework are recorded below.

## [1.0.0-hetero] - 2026-06-23

### Added
- **`src/config/algs/heterogeneous.yaml`**: Configuration variables defining role properties, hidden states, learnable embedding sizes, mapper configs, and toggles.
- **`src/modules/agents/role_encoder.py`**: Introduced `EncoderBank` containing role-specific observation encoders.
- **`src/modules/agents/role_gru.py`**: Introduced `GRUBank` containing role-specific recurrent layers (`nn.GRUCell`), allowing diverse memory spaces.
- **`src/modules/agents/role_policy.py`**: Introduced `PolicyHeadBank` containing role-specific policy heads projecting hidden dimensions to discrete actions.
- **`src/modules/agents/heterogeneous_agent.py`**: Core modular network mapping forward, communicate, and aggregate passes to role-specific banks.
- **`src/modules/role_aware_attention.py`**: `RoleAwareAttention` module integrating sender/receiver learnable role embeddings into target-triggered attention.
- **`src/modules/trust_estimator.py`**: `TrustEstimator` neural network calculating dynamic trust matrices based on usefulness, history, and roles.
- **`src/smac/role_mapper.py`**: SMAC unit-type mapper translating SC2 IDs (Marine, Marauder, Medic, Tank, Zealot, Stalker) into Scout/Fighter/Medic/Tank/Support roles.
- **`role_metrics.py`**: Metrics engine writing csv records of role-wise reward and trust levels to `logs/` and plots to `plots/`.

### Modified
- **`src/modules/agents/__init__.py`**: Registered `HeterogeneousAgent` under the key `'heterogeneous_agent'`.
- **`src/modules/critics/__init__.py`**: Registered `RoleAwareCritic`.
- **`src/controllers/basic_controller.py`**: Added dynamic role mapping, list-based agent hidden initialization, and heterogeneous forward pass.
- **`src/controllers/tmac_p2p_comm_controller.py`**: Updated `VffacMAC` to handle heterogeneous aggregates, key/query/value projects, and communication history rolling.
- **`src/learners/coma_learner.py`**: Integrated `RoleAwareCritic` and implemented target/actor sequence hidden states gathering and alignment.
- **`src/run.py`**: Expanded the default replay buffer schema to store `agent_role`, `role_embedding`, `incoming_messages`, `outgoing_messages`, and `trust_scores` when `args.heterogeneous` is set to `True`.
- **`src/runners/episode_runner.py`**: Modified transition loop to perform role mapping during reset, extract communications statistics from MAC, and pass updates to buffer schemas and metrics logger.
