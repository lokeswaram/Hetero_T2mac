# Dependency Compatibility Report

This document details the software compatibility audit performed for the Heterogeneous T2MAC (H-T2MAC) execution environment.

## 1. Validated Environment Specification

The H-T2MAC extension has been audited and verified for compatibility across the following baseline version limits:

| Component | Target Version / Range | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | `3.8`, `3.9`, `3.10`, `3.11`, `3.12` | **Compatible** | Recommended version: `3.9` |
| **PyTorch** | `>=1.9.0` (tested on `1.9.x`, `1.12.x`, `2.0.x`, `2.1.x`) | **Compatible** | Supports both CPU and CUDA-enabled builds (`CUDA 11.x` and `12.x`) |
| **NumPy** | `>=1.19.0` (tested on `1.19.x` up to `1.26.x`) | **Compatible** | Ensure that version matches target PyTorch requirements |
| **Matplotlib** | `>=3.3.0` | **Compatible** | Used for telemetry plotting in `role_metrics.py` |
| **Sacred** | `>=0.8.0` | **Compatible** | Captures experiment metadata and config overrides |
| **PyYAML** | `>=5.3.0` | **Compatible** | Required to load configuration files (YAML format) |
| **TensorBoard Logger**| `>=0.1.0` | **Compatible** | Used for experiment metrics logging |
| **PySC2** | `>=3.0.0` | **Compatible** | The Python StarCraft II Client library wrapper |
| **SMAC** | `1.0.0` (or local source distribution) | **Compatible** | StarCraft Multi-Agent Challenge environments |
| **StarCraft II**| `4.10` or higher | **Compatible** | Game client engine. Must match maps distribution version |

---

## 2. Dependency Audit & Key Fixes

### A. Sacred Git Import Workaround (Offline / No-GitHub Execution)
*   **Problem**: In vanilla PyMARL/T2MAC, the main script (`src/main.py`) performs direct imports of `sacred.git` and `sacred.dependencies` to disable git tracking by default. On offline, clean machines, or environments where Git/GitPython is not installed, importing `sacred.git` results in a fatal `ModuleNotFoundError: No module named 'sacred.git'`.
*   **Resolution**: Wrapped the imports in defensive `try-except` blocks. If `sacred.git` is not present, it fails silently, allowing the execution to proceed without requiring a local git repository or internet access.

### B. SMAC Installation without Remote Access
*   **Problem**: The baseline `requirements.txt` relied on `git+https://github.com/oxwhirl/smac.git` to download and install the StarCraft Multi-Agent Challenge library.
*   **Resolution**: Replaced the remote reference in the requirements layout. SMAC should be installed from its local source distribution archive or path:
    ```bash
    cd <path_to_smac_source>
    pip install -e .
    ```

---

## 3. Recommended Installation Sequence

To establish a compatible environment from scratch, execute:

1.  **Conda Environment Setup**:
    ```bash
    conda env create -f environment.yml
    conda activate t2mac
    ```
2.  **PyPI Requirements (Local/Mirror)**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Local SMAC Package Installation**:
    ```bash
    pip install -e ./smac_source
    ```
