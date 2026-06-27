# H-T2MAC Training Commands Reference

This reference documents the terminal commands required to run H-T2MAC in various execution modes, maps, and hardware configurations.

All commands assume your virtual environment is active and `SC2PATH` is configured.

---

## 1. Core Hardware Modes

### Single GPU Training (CUDA)
Run training on the active CUDA GPU device.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z use_cuda=True
```

### CPU Training
Force training on the CPU (useful for testing or if no GPU is available).
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z use_cuda=False
```

---

## 2. Framework Execution Modes

### Heterogeneous Mode (H-T2MAC)
Launches the full role-specific encoder, GRU, policy bank, and role-attention/trust architectures.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z
```

---

## 3. Different SMAC Maps

### Homogeneous Maps (e.g. 8m)
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=8m
```

### Heterogeneous Maps (e.g. 2s3z)
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=2s3z
```

### Heterogeneous Mixed Maps (e.g. MMM - Marines, Marauders, Medivacs)
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=MMM
```

---

## 4. Evaluation & Testing

### Evaluate a Checkpoint (Test Mode)
Runs evaluation over 32 episodes using a trained model directory.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" evaluate=True test_nepisode=32
```

### Run Single Episode Inference
Run a single test episode for debugging or step logs.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" evaluate=True test_nepisode=1
```

---

## 5. Checkpoint Operations

### Resume Training from Timestep
Loads the saved optimizer, weights, and timesteps to resume learning.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" load_step=20000
```

---

## 6. Custom Config Overrides

You can pass parameter overrides directly on the command line using Sacred's `with` syntax:

*   **Change learning rate**: `with lr=0.0005`
*   **Disable Trust module**: `with trust_enabled=False`
*   **Disable Role Attention**: `with role_attention_enabled=False`
*   **Change training timesteps**: `with t_max=1000000`

### Complete Override Example
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z lr=0.0005 trust_enabled=False t_max=500000
```
