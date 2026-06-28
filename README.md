# Heterogeneous T2MAC (Hetero_T2MAC)

Heterogeneous Target-Triggered Multi-Agent Communication (Hetero_T2MAC) is a state-of-the-art Multi-Agent Reinforcement Learning (MARL) framework designed specifically for heterogeneous agents. In complex environments like the StarCraft Multi-Agent Challenge (SMAC), agents belong to different unit types (e.g., Scout, Fighter, Medic, Tank, Support) and require distinct observation capacities, policy dimensions, recurrent states, and communication behaviors. 

Unlike homogeneous MARL systems that force all agents to share identical network parameters, Hetero_T2MAC factorizes agent architectures by using specialized, role-based neural network banks and learnable role embeddings.

---

## Key Features & Architecture

```
                       Observation
                            │
                            ▼
                      [Role Encoder]       (Role-specific observation mapping)
                            │
                            ▼
                     [Hetero Agent]        (Central agent model coordinator)
                            │
                            ▼
                  [Communication Module]   (Generates queries, keys, and values)
                            │
                            ▼
                    [Role Attention]       (Attention weights conditioned on role embs)
                            │
                            ▼
                    [Trust Estimator]      (Gating via communication history & utility)
                            │
                            ▼
                      [GRU Memory]         (Recurrent hidden state preservation)
                            │
                            ▼
                     [Action Head]         (Role-specific action Q-value outputs)
```

1. **Role Encoder (`role_encoder.py`)**: Map raw observation features to role-specific hidden spaces using an `EncoderBank` to handle distinct state shapes.
2. **Hetero Agent (`hetero_agent.py`)**: Orchestrates the forward pass, recurrent updates, and communication aggregation across different agent roles.
3. **Communication Module (`communication.py`)**: Computes message keys, queries, and values, and evaluates sending probabilities based on uncertainty dynamics.
4. **Role Attention (`attention.py`)**: Computes dot-product communication attention conditioned on sender/receiver learnable role embeddings.
5. **Trust Estimator (`trust_estimator.py`)**: Calculates a pairwise trust score ($T_{ij}$) using the history of communication actions and the local usefulness of messages.
6. **GRU Memory (`role_gru.py`)**: Uses a `GRUBank` of role-specific `nn.GRUCell` layers, allowing different memory capacities for scouts, tanks, and healers.
7. **Action Head (`action_head.py`)**: Projects hidden states to role-specific action output layers.

---

## Installation

### Linux / Windows Setup
1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd T2MAC_Base-main
   ```

2. **Conda Setup**:
   Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate t2mac
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install SMAC**:
   Install the StarCraft Multi-Agent Challenge library:
   ```bash
   pip install -e ./smac_source
   ```

5. **Cleanup Command**:
   Run the cleanup script to remove deprecated homogeneous components:
   ```bash
   clean_repo.bat
   ```

---

## Training and Evaluation

### Training Command
To train Hetero_T2MAC on StarCraft II scenario `3s5z`:
```bash
python src/main.py --config=hetero_t2mac --env-config=sc2 with env_args.map_name=3s5z use_cuda=True
```

### Evaluation Command
To evaluate a trained model using saved checkpoints:
```bash
python src/main.py --config=hetero_t2mac --env-config=sc2 with env_args.map_name=3s5z evaluate=True checkpoint_path="results/models/<model_id>"
```

### Google Colab Integration
You can run training directly inside Google Colab using the following cells:
```python
# Clone the repository
!git clone <repo_url>
%cd T2MAC_Base-main

# Install dependencies
!pip install -r requirements.txt

# Run verification test
!python verify_heterogeneous.py

# Launch training on dummy environment
!python src/main.py --config=hetero_t2mac --env-config=dummy
```

---

## Results and Metrics

Role-wise metric logging is automatically handled during training:
- **Telemetry Logs**: Saved in the `logs/` directory as CSV files.
- **Plots**: Real-time plots for rewards, average trust, and role-wise trust are saved in the `plots/` directory (e.g., `role_reward.png`, `avg_trust.png`).
- **TensorBoard**: Visualizations can be launched via:
  ```bash
  tensorboard --logdir=results/tb_logs
  ```