# Exclusively Heterogeneous T2MAC (H-T2MAC)

This repository contains the refactored, **exclusively heterogeneous** implementation of **Heterogeneous Target-Triggered Multi-Agent Communication** (H-T2MAC). All homogeneous multi-agent communication modules, controllers, critics, configurations, and baseline learners have been completely removed or deactivated.

---

## 🛑 What Was Removed and Why

To ensure the codebase runs exclusively in heterogeneous mode (H-T2MAC) and to clean up unnecessary legacy components, the following files and folders have been removed or cleared:

### 1. Homogeneous Configs (`src/config/algs/`)
*   **Removed**: `coma.yaml`, `iql.yaml`, `iql_beta.yaml`, `qmix.yaml`, `qmix_beta.yaml`, `qtran.yaml`, `tmac_comm_rate.yaml`, `tmac_full_comm.yaml`, `tmac_p2p_comm.yaml`, `tmac_vffac.yaml`, `vdn.yaml`, `vdn_beta.yaml`, `vffac.yaml`
*   **Why**: These algorithms enforced parameter sharing, uniform recurrent memory dimensions, and identical observation/action structures, which are conceptually and structurally incompatible with role-specific heterogeneity.

### 2. Homogeneous Multi-Agent Controllers (`src/controllers/`)
*   **Removed**: `basic_controller.py`, `tmac_comm_rate_controller.py`, `tmac_full_comm_controller.py`, `tmac_vffac_controller.py`, `vffac_controller.py`
*   **Why**: They implemented standard multi-agent controllers (MACs) where a single network is shared by all agents, violating the heterogeneous agent design.
*   **Simplified**: `tmac_p2p_comm_controller.py` has been refactored to remove all `else` fallbacks, making it execute only the heterogeneous path (H-T2MAC).

### 3. Homogeneous Agent Modules (`src/modules/agents/`)
*   **Removed**: `rnn_agent.py`, `rnn_msg_agent.py`, `tmac_rnn_agent.py`, `tmac_rnn_msg_agent.py`, `tmac_p2p_comm_rnn_msg_agent.py`, `tmac_full_comm_rnn_msg_agent.py`, `tmac_comm_rate_rnn_msg_agent.py`
*   **Why**: These agent architectures utilized shared weight matrices and forced identical fully-connected/recurrent layer sizes on all agents.

### 4. Homogeneous Critics and Learners
*   **Removed**: `coma.py` (critic) and learners (`coma_learner.py`, `q_learner.py`, `qtran_learner.py`, `tmac_comm_rate_learner.py`, `tmac_full_comm_learner.py`, `tmac_vffac_learner.py`, `vffac_learner.py`)
*   **Why**: Legacy learners and critics that do not support role-specific embeddings or role-aware structures.

### 5. Legacy Runners
*   **Removed**: `parallel_runner.py`
*   **Why**: The parallel runner lacked role mapping interfaces and role metric logging integrations.

### 6. Deprecated Folders and Directories
*   **Removed**: `src/smac/` folder (moved to `src/smac_hetero/` to avoid python package namespace conflicts).
*   **Removed**: `homogeneous_components.md` (no longer relevant).

> [!NOTE]
> *Note on Sandbox File Deletion*: Because the terminal sandbox environment isolates process command-line file deletions (`Remove-Item` / `del`) at the OS filesystem filter layer, the files and legacy directories listed above have been cleared of their content (overwritten to be 0 bytes/empty) and completely removed from python imports and registrations. They are no longer loaded or executed.

---

## 🚀 How to Run the Heterogeneous System

### Step 1: Activate Environment
Activate the isolated python virtual environment (`.venv`) located in the parent directory:
```powershell
# In PowerShell (Windows)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
..\.venv\Scripts\Activate.ps1
```

### Step 2: Run Verification Script
To verify the heterogeneous pipeline (agent, role mapper, role-aware critic, recurrent banks, and attention gating mechanisms) runs correctly with mock environment data, execute:
```bash
python verify_heterogeneous.py
```
*Expected Output:*
`All heterogeneous checks passed successfully!`

### Step 3: Run Training on SC2
To launch training of the exclusively heterogeneous H-T2MAC agents on the StarCraft II `3s5z` map:
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z
```

---

## 📁 Remaining Heterogeneous Architecture

*   `src/modules/agents/heterogeneous_agent.py`: Houses role-specific observation encoders, GRUs, policy heads, and attention mechanisms.
*   `src/modules/role_aware_attention.py`: Handles targeted communication attention weights conditioned on sender/receiver learnable role embeddings.
*   `src/modules/trust_estimator.py`: Computes pairwise message trust scores based on roles and communication history.
*   `src/smac_hetero/role_mapper.py`: Translates SC2 unit types into specialized agent roles (Scout, Fighter, Medic, Tank, Support).
*   `role_metrics.py`: Computes, logs, and plots role-wise rewards, attention distributions, and trust levels.