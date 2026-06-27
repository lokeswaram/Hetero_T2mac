# H-T2MAC Quick Start Guide

This is a one-page reference for running Heterogeneous T2MAC (H-T2MAC).

---

## 1. Environment Setup

Configure your Python environment and dependencies locally:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install local SMAC package
pip install -e ./smac_source

# 4. Configure StarCraft II Client path
export SC2PATH=$HOME/StarCraftII  # Linux
# Windows (PowerShell): $env:SC2PATH = "C:\Program Files (x86)\StarCraft II"
```

---

## 2. Core Execution Commands

### Verify Environment Configuration
```bash
python verify_heterogeneous.py
```

### Start H-T2MAC Training (Heterogeneous Mode)
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z
```

### Start Standard Baseline Training (Homogeneous Mode)
```bash
python src/main.py --config=tmac_p2p_comm --env-config=sc2 with env_args.map_name=3s5z
```

### Evaluate Checkpoint (Testing)
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/<run_token>" evaluate=True test_nepisode=32
```

### Resume Training from Checkpoint
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/<run_token>" load_step=10000
```

---

## 3. Telemetry Visualizer
Plot role reward curves and trust matrix charts from the CSV files in `logs/`:
```bash
python -c "from role_metrics import RoleMetricsLogger; logger = RoleMetricsLogger(); logger.plot_metrics()"
```
*   **Raw CSV records**: `logs/`
*   **Exported plots**: `plots/`
