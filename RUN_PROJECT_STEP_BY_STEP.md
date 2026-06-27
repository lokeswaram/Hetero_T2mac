# Step-by-Step Run Guide: H-T2MAC

This guide assumes you are starting from a completely clean machine with zero pre-installed libraries. Follow these exact steps to set up, train, and evaluate the Heterogeneous T2MAC (H-T2MAC) framework.

---

### STEP 1: Install Python
Download and install Python 3.9 (recommended) for your operating system.
*   **Windows**: Download the installer from the official website and make sure to check the box that says **"Add Python to PATH"** during installation.
*   **Linux / Ubuntu**:
    ```bash
    sudo apt update
    sudo apt install python3.9 python3.9-venv python3.9-dev -y
    ```

---

### STEP 2: Create Environment
Navigate to the root directory of the project and create an isolated virtual environment (`.venv`) to avoid conflicts with global libraries.
*   **PowerShell (Windows)**:
    ```powershell
    python -m venv .venv
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
    .\.venv\Scripts\Activate.ps1
    ```
*   **Linux / macOS**:
    ```bash
    python3.9 -m venv .venv
    source .venv/bin/activate
    ```

---

### STEP 3: Install Dependencies
With the environment activated, upgrade your package manager and install the core requirements.
*   **Windows / Linux**:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

### STEP 4: Install SMAC
StarCraft Multi-Agent Challenge (SMAC) must be installed from your local source directory.
1.  Locate the directory containing the SMAC source code.
2.  Install it in editable mode inside your virtual environment:
    ```bash
    # Replace './smac_source' with the actual path to your SMAC folder if different
    pip install -e ./smac_source
    ```

---

### STEP 5: Install StarCraft II
T2MAC requires the StarCraft II game client engine installed on your system.
*   **Windows**: Download and install StarCraft II through Battle.net. By default, it installs to `C:\Program Files (x86)\StarCraft II`.
*   **Linux**: Download the headless StarCraft II Linux client:
    ```bash
    # Extract to home folder
    wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip SC2.4.10.zip -d $HOME/
    # When prompted for a password, enter: iagreetotheeula
    ```
*   **Configure PATH**: Set the `SC2PATH` environment variable so Python can find the client:
    *   **Windows (PowerShell)**:
        ```powershell
        $env:SC2PATH = "C:\Program Files (x86)\StarCraft II"
        [Environment]::SetEnvironmentVariable("SC2PATH", "C:\Program Files (x86)\StarCraft II", "User")
        ```
    *   **Linux**:
        ```bash
        export SC2PATH=$HOME/StarCraftII
        echo "export SC2PATH=$HOME/StarCraftII" >> ~/.bashrc
        ```

---

### STEP 6: Download Maps
Download the SMAC map packs and place them in the game client's directory.
1.  Locate the map files (e.g. `3s5z.SC2Map`, `6h_vs_8z.SC2Map`).
2.  Move them into the maps directory of your StarCraft II installation:
    *   **Windows**: Copy map files to `C:\Program Files (x86)\StarCraft II\Maps\SMAC\`
    *   **Linux**:
        ```bash
        mkdir -p $SC2PATH/Maps
        cp -r smac_maps/ $SC2PATH/Maps/
        ```

---

### STEP 7: Verify Installation
Verify that the codebase and components load correctly by running the pre-packaged validation script:
```bash
python verify_heterogeneous.py
```
*Expected Output*: "All heterogeneous checks passed successfully!"

---

### STEP 8: Run Training
Start H-T2MAC heterogeneous training on a specific map (e.g., `3s5z`).
*   **Heterogeneous Mode (H-T2MAC)**:
    ```bash
    python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z
    ```

---

### STEP 9: Monitor Logs
H-T2MAC writes raw metric summaries and training performance stats locally.
*   **Training metrics**: Look inside the `results/` folder for Sacred runs and Tensorboard summaries.
*   **Role metrics**: Look inside the `logs/` directory for csv logs (e.g., `avg_trust.csv`).

---

### STEP 10: Resume Training
To resume training from a saved checkpoint, use the `checkpoint_path` and `load_step` config parameters.
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" load_step=10000
```

---

### STEP 11: Run Evaluation
To evaluate a trained model over multiple episodes without performing parameter updates:
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" evaluate=True test_nepisode=32
```

---

### STEP 12: Load Saved Model
The system loads saved checkpoints automatically when you specify the `checkpoint_path`. The matching network configurations (roles, sizes) must be present in the configuration parameters or `heterogeneous.yaml` config file.

---

### STEP 13: Run Inference
Inference runs inside evaluation mode. Set `test_nepisode=1` to run a single episode visualization or testing step:
```bash
python src/main.py --config=heterogeneous --env-config=sc2 with env_args.map_name=3s5z checkpoint_path="results/models/heterogeneous__2026-06-23_22-17-00" evaluate=True test_nepisode=1
```

---

### STEP 14: Visualize Results
Run the plotting routine to visualize role-wise learning curves, attention mappings, and trust telemetry:
```bash
python -c "from role_metrics import RoleMetricsLogger; logger = RoleMetricsLogger(); logger.plot_metrics()"
```
This exports PNG figures into the `plots/` folder.

---

### STEP 15: Troubleshooting

*   **Error: ModuleNotFoundError: No module named 'sacred.git'**
    *   *Cause*: The sacred git extra is missing.
    *   *Fix*: The code has been updated to bypass this import error. Simply ensure your virtual environment contains the requirements in `requirements.txt`.
*   **Error: StarCraft2Env could not find StarCraft II**
    *   *Cause*: `SC2PATH` is unset or points to an invalid directory.
    *   *Fix*: Export your `SC2PATH` environment variable as described in STEP 5.
*   **Error: No such file or directory: Maps/SMAC/3s5z.SC2Map**
    *   *Cause*: The SC2 map packs are missing or located in the wrong subfolder.
    *   *Fix*: Download the maps and place them inside the `Maps` folder of your StarCraft II installation folder.
