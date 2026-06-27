# Dependency Updates: H-T2MAC

The following library versions are recommended to run the Heterogeneous T2MAC extension:

## 1. Required Packages
We rely strictly on standard PyTorch and NumPy modules:
- `torch>=1.9.0`
- `numpy>=1.19.0`

## 2. SMAC Environment Integration
H-T2MAC maps SC2 unit types to agent roles using the base StarCraft2Env attributes. Ensure you have the SMAC library installed from a local source distribution:
```bash
# Locate your local SMAC source package, navigate to it, and install:
cd smac_source
pip install -e .
```

## 3. Matplotlib for Metrics
To enable metrics plotting and rendering (`role_metrics.py`), make sure `matplotlib` is installed:
```bash
pip install matplotlib
```
If running on a headless server, matplotlib will automatically fall back to non-interactive backends (e.g. `Agg`) to prevent errors when generating plots.
