# Parasitic Backdoor Attack Research Framework

This repository contains the official research implementation, experiment notebooks, and benchmarking export pipelines for **Parasitic Backdoor Attacks** on Deep Neural Networks (e.g., ResNet-18 on CIFAR-10).

The parasitic backdoor attack is an advanced machine learning security paradigm where:
1. **TracIn Host Selection**: High-influence training samples ("hosts") are selected using gradient influence tracing (TracIn across checkpoint trajectories).
2. **Trigger Co-adaptation**: The backdoor trigger pattern is co-optimized alongside host sample perturbations during training.
3. **Dormant Backdoor Activation**: The backdoor remains dormant (low ASR, high clean accuracy) under normal training, but activates (high ASR) when host samples are unlearned or removed during model repair/fine-tuning.

---

## 📁 Repository Overview

```
parasitic-backdoor-attack/
├── BackdoorBench/                  # Submodule / benchmarking workspace for backdoor defenses
├── bb_export/                      # Export directory for BackdoorBench-compatible artifacts
├── logs/                           # SLURM and python execution logs
├── BadNet Replication.ipynb        # Standard BadNet baseline comparison notebook
├── Influence Function.ipynb        # Initial TracIn baseline notebook (v1)
├── Influence Function v2.ipynb - v39.ipynb # Iterative research and evaluation notebooks
├── run_parasitic_attack.py        # Standalone Python script for parasitic attack & BB export
└── run_parasitic_attack.sh        # SLURM job submission script for GPU clusters
```

---

## 📓 Notebook Version Log (`v1` – `v39`)

The repository contains 40+ Jupyter notebooks documenting the iterative development, hyperparameter search, and evaluation of parasitic backdoor attacks.

### Milestone 1: Initial TracIn & Base Pipeline (`v1` – `v5`)
* **`Influence Function.ipynb` (v1) / `v2` / `v3` / `v4`**:
  * Implemented standard CIFAR-10 ResNet-18 training pipeline.
  * Added first-order TracIn checkpoint loss tracing to score sample influence across training epochs.
  * Identified top $K$ high-influence host samples for targeting.
* **`v5`**:
  * Added hardware and DataLoader performance optimizations (Automatic Mixed Precision `amp`, `pin_memory=True`, multi-worker data loading).

### Milestone 2: Dormant Trigger & Unlearning Dynamics (`v6` – `v7`)
* **`v6` – `v7`**:
  * Introduced the **Dormant Trigger** paradigm: evaluating model state where trigger activation is suppressed during clean training.
  * Implemented preliminary host dataset unlearning simulation to measure ASR jumps upon host sample deletion.

### Milestone 3: Alternating Co-adaptation & Parameter Tuning (`v8` – `v11`)
* **`v8` – `v9`**:
  * Modularized pipeline into 4 structured phases: *1. Setup*, *2. Alternating Co-adaptation*, *3. Dormancy Check*, and *4. Unlearning Simulation*.
* **`v10`**:
  * Introduced **Fast TracIn** approximation for accelerated host selection without full trajectory backpropagation.
  * Evaluated exact host unlearning vs. baseline random sample removal.
* **`v11`**:
  * Hyperparameter tuning for stealthiness: quiet trigger constraints ($\epsilon = 8/255$, $\alpha_{\text{trigger}} = 2/255$), reduced host set ($K_{\text{hosts}} = 500$), and poison budget ($250$).

### Milestone 4: Unified Per-Batch Optimization & Scale-Up (`v12` – `v21`)
* **`v12` – `v13`** (`v12-Copy1.ipynb`):
  * Replaced epoch-level alternating optimization with **Unified Per-Batch Co-adaptation** during host training.
  * Scaled host anchor count and poison budget to $1000$ ($1:1$ host-to-poison ratio).
* **`v14` – `v15`** (`v15-Copy1.ipynb`):
  * Tested extra-quiet triggers ($\epsilon = 4/255$, $\alpha = 1/255$) to prevent premature shortcut learning by the model.
* **`v16` – `v21`**:
  * Systematic sweep over perturbation bounds ($\epsilon \in \{8/255, 12/255, 16/255\}$, $\alpha = 4/255$).
  * Achieved low dormant ASR ($<5\%$) combined with rapid ASR jump ($>85\%$) upon target host unlearning.

### Milestone 5: BackdoorBench Integration (`v22` – `v26`)
* **`v22` – `v25`**:
  * Integrated evaluation metrics compatible with standard backdoor defense benchmarks.
  * Added defense transferability probing across fine-tuning regimes.
* **`v26 BackdoorBench.ipynb`**:
  * Standalone, BackdoorBench-ready version of `v25`.
  * Formatted output data structures (`attack_result.pt`) containing clean test sets, poisoned test sets, poison index lists, and trained weights.

### Milestone 6: Equilibrium, Repair Scaling & Transferability (`v27` – `v39`)
* **`v27` – `v29`**:
  * Evaluated host-to-poison ratio dynamics ($2:1$ host advantage vs. $1:1$ equilibrium) to prevent trigger leakage before unlearning.
* **`v30` – `v34`**:
  * Standardized repair fine-tuning ($LR = 0.01$, repair duration scaled from 10 to 15 epochs) to allow poisons time to embed during unlearning.
  * Added checkpoint caching (`[INFO] Found existing checkpoint! Bypassing Phase 1 & 2...`) for rapid repair evaluation.
* **`v35` – `v38`**:
  * Conducted extensive **Approximate Unlearning Server-Side Transferability** sweeps.
  * Evaluated defense resistance across varying repair learning rates ($LR \in \{0.001, 0.01, 0.1\}$) and epoch durations ($10, 15, 30$ epochs).
* **`v39`**:
  * Final consolidated experimental notebook validating:
    * **Clean Data Accuracy (CDA)**: $>92\%$
    * **Dormant ASR**: $<5\%$
    * **Active Post-Unlearning ASR**: $>90\%$

### Auxiliary Notebooks
* **`BadNet Replication.ipynb`**: Replicates standard BadNet attack baseline on CIFAR-10 for direct comparison against parasitic attacks.

---

## 🛠️ BackdoorBench Export & Pipeline Usage

The repository provides a standalone Python script ([run_parasitic_attack.py](file:///scratch/ss17886/parasitic-backdoor-attack/run_parasitic_attack.py)) and a SLURM launch script ([run_parasitic_attack.sh](file:///scratch/ss17886/parasitic-backdoor-attack/run_parasitic_attack.sh)) to execute the full pipeline and generate artifacts compatible with **BackdoorBench**.

### 1. Running via Python Script

Execute [run_parasitic_attack.py](file:///scratch/ss17886/parasitic-backdoor-attack/run_parasitic_attack.py) directly:

```bash
python run_parasitic_attack.py \
  --seed 0 \
  --target-class 3 \
  --k-hosts 1000 \
  --poison-budget 1000 \
  --epsilon 0.062745 \
  --alpha-trigger 0.015686 \
  --batch-size 1024 \
  --lr 0.08 \
  --epochs-base 100 \
  --epochs-coadapt 10 \
  --data-dir ./BackdoorBench/data \
  --bb-root ./BackdoorBench \
  --export-dir ./bb_export/parasitic_attack
```

#### Command-Line Arguments:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--seed` | `0` | Random seed for reproducibility |
| `--target-class` | `3` | Target class index for backdoor trigger |
| `--k-hosts` | `1000` | Number of high-influence host samples selected via TracIn |
| `--poison-budget` | `1000` | Number of poisoned samples introduced |
| `--epsilon` | `16/255` (`0.0627`) | Maximum perturbation norm for trigger pattern |
| `--alpha-trigger` | `4/255` (`0.0157`) | Optimization step size per trigger update |
| `--batch-size` | `1024` | Batch size for training and co-adaptation |
| `--lr` | `0.08` | Base SGD learning rate |
| `--epochs-base` | `100` | Number of epochs for Phase 1 base training |
| `--epochs-coadapt` | `10` | Number of epochs for Phase 2 co-adaptation |
| `--data-dir` | `./BackdoorBench/data` | Path to dataset directory |
| `--bb-root` | `./BackdoorBench` | Root directory of BackdoorBench framework |
| `--export-dir` | `./bb_export/parasitic_attack` | Output folder for generated artifacts |

---

### 2. Running via SLURM Cluster Batch Job

To launch a GPU batch job on an HPC cluster managed by SLURM, use [run_parasitic_attack.sh](file:///scratch/ss17886/parasitic-backdoor-attack/run_parasitic_attack.sh):

```bash
sbatch run_parasitic_attack.sh
```

---

### 3. Generated Export Artifacts

Upon completion, the export pipeline writes BackdoorBench-compatible bundle files to `bb_export/parasitic_attack/`:

* **`attack_result.pt`**: PyTorch dictionary containing:
  * `model`: Model state dictionary post-coadaptation.
  * `clean_test`: Standard clean test dataset.
  * `poison_test`: Test dataset with backdoor trigger applied.
  * `poison_indices`: List of indices corresponding to poisoned samples in the training set.
  * `trigger`: Optimized trigger pattern tensor.

---

### 4. Evaluating Against BackdoorBench Defenses

Once `attack_result.pt` is generated, you can run standardized BackdoorBench defenses (such as Fine-pruning, Neural Cleanse, STRIP, ABL, NAD, Spectral Signature) against the parasitic attack:

```bash
cd BackdoorBench

# Example: Run Fine-pruning defense against parasitic attack export
python defense/fp.py \
  --result_file ../bb_export/parasitic_attack/attack_result.pt \
  --yaml_path ./config/defense/fp/cifar10.yaml

# Or execute batch defense script
sbatch run_defenses.sh
```

---

## 📊 Summary of Results & Visualizations

Key plots generated by the notebooks in the workspace:
* **[asr_jump.png](file:///scratch/ss17886/parasitic-backdoor-attack/asr_jump.png)**: Demonstrates the sharp increase in Attack Success Rate (ASR) after host sample unlearning.
* **[poisoned_samples.png](file:///scratch/ss17886/parasitic-backdoor-attack/poisoned_samples.png)** & **[poisoned_samples_comparison.png](file:///scratch/ss17886/parasitic-backdoor-attack/poisoned_samples_comparison.png)**: Visual comparison of original host images vs. co-adapted poisoned images.
* **[class_accuracies.png](file:///scratch/ss17886/parasitic-backdoor-attack/class_accuracies.png)**: Per-class Clean Data Accuracy across attack phases.
