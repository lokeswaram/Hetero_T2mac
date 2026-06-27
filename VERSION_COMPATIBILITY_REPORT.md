# Version Compatibility Report: H-T2MAC

This report verifies version compatibility for Heterogeneous T2MAC (H-T2MAC).

## 1. Python Environment Requirements
- **Python**: `>=3.8` (recommended 3.9)
- **PyTorch**: `>=1.9.0` (compatible with `CUDA 11.x` and `12.x`)
- **NumPy**: `>=1.19.0`
- **PySC2**: `>=3.0.0`
- **SMAC**: StarCraft Multi-Agent Challenge (SMAC) library (standard version)

## 2. Backward Compatibility Matrix
The following matrix details the compatibility of configuration settings and model loading:

| Active Config | Saved Model | Action | Compatibility Status |
| :--- | :--- | :--- | :--- |
| `heterogeneous: False` | Homogeneous | Train / Eval | **Fully Compatible** (standard baseline behaviour) |
| `heterogeneous: True` | Heterogeneous | Train / Eval | **Fully Compatible** (H-T2MAC behaviour) |
| `heterogeneous: True` | Homogeneous | Load / Eval | **Incompatible** (structure mismatch: missing Role Banks & Embeddings) |
| `heterogeneous: False` | Heterogeneous | Load / Eval | **Incompatible** (structure mismatch: model has extra parameters) |

## 3. SMAC Map Compatibility
H-T2MAC unit mappings support standard homogeneous and heterogeneous maps:

- **Homogeneous Maps (e.g. 3m, 8m)**:
  All agents will automatically map to the `fighter` role and run correctly.
- **Heterogeneous Maps (e.g. 2s3z, 3s5z, MMM)**:
  Agents will be mapped based on unit types:
  - Marines -> `fighter`
  - Marauders -> `fighter`
  - Medivacs -> `medic`
  - Zealots -> `tank`
  - Stalkers -> `support`
  - Unknown units default to `scout`.
