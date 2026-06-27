# Migration Guide: Moving from T2MAC to Heterogeneous T2MAC (H-T2MAC)

This guide assists researchers in migrating their experiments from the original homogeneous T2MAC framework to the new Heterogeneous T2MAC version.

---

## 1. Backwards Compatibility
All modifications are fully backward compatible. If you run your training scripts with your original config files (e.g. `tmac_p2p_comm.yaml`, `qmix.yaml`), the code defaults to `heterogeneous: False` and uses the standard homogeneous components.

---

## 2. Configuration Setup
To configure a heterogeneous experiment:
1. Create or modify your algorithm config file to include the `heterogeneous: True` flag.
2. Define your role categories and their recurrent hidden dimensions:
   ```yaml
   heterogeneous: True
   num_roles: 3
   role_names: ["scout", "fighter", "medic"]
   roles:
     scout:
       hidden_dim: 64
     fighter:
       hidden_dim: 128
     medic:
       hidden_dim: 96
   role_emb_dim: 16
   role_assignment_method: "unit_type" # Options: fixed, random, unit_type
   ```

---

## 3. Custom SMAC Role Mapping
If you define custom unit types or custom StarCraft II maps, update the `SMACRoleMapper` class in `src/smac_hetero/role_mapper.py` to map your custom unit type IDs to the correct role names.
For instance:
```python
if hasattr(env, "custom_unit_id") and utype == env.custom_unit_id:
    role_name = "tank"
```

---

## 4. Replay Buffer Schema Alignment
H-T2MAC stores role and communication information at each timestep. If loading models trained on old checkpoints:
> [!WARNING]
> Old checkpoints do not contain role information or metrics in their model files. You must use the matching config flags when restoring training or evaluating from checkpoints.

When loading an old homogeneous checkpoint to run in evaluation, ensure you do **not** set `heterogeneous: True`, otherwise the model loader will throw a missing parameter dictionary error for the Banks and Embeddings.

---

## 5. Metrics Extraction
Logged role metrics are stored under `logs/` in CSV files. You can fetch and plot them at any time during training.
- `avg_trust.csv`: Mean trust score across all agents.
- `trust_[role_name].csv`: Mean trust received by units of `[role_name]`.
- `role_reward.csv`: Episode returns mapped across timesteps.

Plots are exported to the `/plots` folder as PNG images for quick visual analysis.
