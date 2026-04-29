# HAR Benchmark Comparison -- Complete Guide

## Overview

This project benchmarks **3 Human Activity Recognition (HAR) models** on the **HHAR dataset**, comparing supervised learning, self-supervised pretraining, and cross-device transfer across smartphones and smartwatches.

All training uses **LOSO (Leave-One-Subject-Out)** cross-validation: each fold holds out one user as the test set and trains on all others.

---

## Models

| Model | Architecture | Framework | Sensors | Pretrained Source |
|---|---|---|---|---|
| **HART** | Lightweight Transformer | TensorFlow/Keras | Accel + Gyro (6ch) or Accel only (3ch) | N/A (supervised only) |
| **LIMU-BERT** | BERT-style Transformer | PyTorch | Accel + Gyro (6ch) | Self-supervised on IMU data |
| **ssl-wearables (HARNet/ResNet)** | CNN / ResNet | PyTorch | Accel only (3ch) or Accel + Gyro (6ch) | Self-supervised on UK Biobank |
| **ResNet-Baseline** | Basic 1D ResNet-18 | PyTorch | Accel only (3ch) or Accel + Gyro (6ch) | N/A (supervised only) |

## Dataset

**HHAR (Heterogeneity Human Activity Recognition)** -- 9 users, 6 devices, 5 activities (bike excluded).

### Activities

| Index | Activity | In "no-bike" mode |
|---|---|---|
| 0 | sit | Yes |
| 1 | stand | Yes |
| 2 | walk | Yes |
| 3 | stairsup | Yes |
| 4 | stairsdown | Yes |
| 5 | bike | **Excluded** |

### Devices

| Device | Type | Native Rate | Window | Downsample | Final Length |
|---|---|---|---|---|---|
| nexus4 | Phone | ~200 Hz | 512 | 4 | 128 |
| s3 | Phone | ~150 Hz | 384 | 3 | 128 |
| s3mini | Phone | ~100 Hz | 256 | 2 | 128 |
| samsungold | Phone | ~50 Hz | 128 | 1 | 128 |
| lgwatch | Watch | ~200 Hz | 512 | 4 | 128 |
| gear | Watch | ~100 Hz | 256 | 2 | 128 |

All 6 activities are collected on **both** smartphones and smartwatches.

### Data Preprocessing per Model

Each model requires a **different** input format. `prepare_hhar_data.py` handles all three automatically.

| | HART | LIMU-BERT | ssl-wearables |
|---|---|---|---|
| **Strategy** | Per-device windowing + lowpass downsample | Fixed 50 ms time-slot averaging | Linear interpolation resample |
| **Target Rate** | ~50 Hz (128 pts / ~2.56 s) | 20 Hz (one point per 50 ms) | 30 Hz |
| **Window Length** | 128 time steps | 120 time steps (6 s) | 300 time steps (10 s) |
| **Normalization** | Per-device z-score | None (divided by 9.8 at training) | / 9.8 + clip to [-3, 3] |
| **Channels** | 6ch (acc + gyro) | 6ch (acc + gyro) | 3ch (acc only) **and** 6ch (acc + gyro)---dont load pretrained weights |
| **Device-aware** | Yes (per-device window & downsample) | No (purely time-driven) | No (uniform resample) |

> **Note on ssl-wearables pretrained weights:**
> The original HARNet / ssl-wearables pretrained weights (from UK Biobank) were trained on **3-channel accelerometer-only** data.
> They can **only** be loaded for `--channels 3` runs.
> When running with `--channels 6` (acc + gyro), you **must** train from scratch -- pretrained weights are incompatible with the 6-channel input.

---

## File Structure

| File | Purpose |
|---|---|
| `prepare_hhar_data.py` | Reads raw HHAR CSVs, generates processed data for all models |
| `prepare_pretrain_data.py` | Prepares **unlabeled** external data for SSL pretraining (Exp 2/3) |
| `prepare_wisdm_data.py` | Prepares labeled WISDM data for cross-dataset eval (Exp 5) |
| `train_loso.py` | Unified LOSO training script for all models |
| `pretrain_limu.py` | LIMU-BERT-X self-supervised pretraining (masked reconstruction) |
| `pretrain_ssl.py` | HARNet/SSL self-supervised pretraining (multi-task) |
| `eval_cross_dataset.py` | Cross-dataset training (HHAR) + eval (WISDM) for Exp 5 |
| `run_experiments.ps1` | PowerShell automation: runs data prep + all experiments |
| `code/resnet-baseline/resnet1d_baseline.py` | Basic 1D ResNet-18 model (CNN baseline) |

---

## Step-by-Step: How to Train

### Step 1 -- Prepare Data

Before any training, you must process the raw HHAR CSV files into the format each model expects.

```powershell
# All models, phone + watch, 5 classes (no bike)
python prepare_hhar_data.py --no-bike
```

**Selective preparation:**

```powershell
# Only prepare for a specific model
python prepare_hhar_data.py --hart --no-bike          # HART only
python prepare_hhar_data.py --limu --no-bike          # LIMU-BERT only
python prepare_hhar_data.py --ssl  --no-bike          # ssl-wearables only

# Only prepare for a specific device type
python prepare_hhar_data.py --phone-only --no-bike    # Smartphone data only
python prepare_hhar_data.py --watch-only --no-bike    # Smartwatch data only

# Combine flags
python prepare_hhar_data.py --ssl --watch-only --no-bike
```

**Output directories (6-class / 5-class):**

| Model | 6-class directory | 5-class directory (--no-bike) |
|---|---|---|
| HART (all) | `datasetStandardized/HHAR/` | `datasetStandardized/HHAR_nobike/` |
| HART (phone) | `datasetStandardized/HHAR_phone/` | `datasetStandardized/HHAR_phone_nobike/` |
| HART (watch) | `datasetStandardized/HHAR_watch/` | `datasetStandardized/HHAR_watch_nobike/` |
| LIMU-BERT (phone) | `dataset/hhar/` | `dataset/hhar_nobike/` |
| LIMU-BERT (watch) | `dataset/hhar_watch/` | `dataset/hhar_watch_nobike/` |
| ssl-wearables (phone) | `data/downstream/hhar/` | `data/downstream/hhar_nobike/` |
| ssl-wearables (watch) | `data/downstream/hhar_watch/` | `data/downstream/hhar_watch_nobike/` |

> 6-class and 5-class data are stored in **separate directories** and never overwrite each other.

---

### Step 2 -- Train Individual Models

Use `train_loso.py` directly to train any single configuration.

#### Arguments

| Argument | Values | Default | Description |
|---|---|---|---|
| `--model` | `hart`, `limu-bert`, `ssl-wearables`, `resnet-baseline` | (required) | Which model to train |
| `--device-type` | `phone`, `watch`, `all` | `all` | Which HHAR device subset |
| `--channels` | `3`, `6` | `6` | 3 = accel only, 6 = accel + gyro |
| `--no-bike` | flag | off | Use 5-class data (exclude bike) |
| `--pretrained` | path | None | Path to pretrained weights |
| `--data-fraction` | 0.0 -- 1.0 | `1.0` | Fraction of training data to use |
| `--epochs` | int | model default | Override training epochs |
| `--batch-size` | int | model default | Override batch size |
| `--gpu` | `"0"`, `"1"`, `"-1"` | `"0"` | GPU id (`-1` = CPU only) |
| `--limu-seq-len` | `20`, `120` | `120` | LIMU-BERT only: 120 = LIMU-BERT (full seq), 20 = LIMU-BERT-X (random crop) |
| `--experiment-tag` | string | None | Tag for result file naming |
| `--output-dir` | path | `loso_results/` | Where to save results |

#### Example Commands

```powershell
# --- HART ---
python train_loso.py --model hart --device-type phone --channels 6 --no-bike
python train_loso.py --model hart --device-type watch --channels 3 --no-bike
python train_loso.py --model hart --device-type all --channels 6 --no-bike --epochs 100

# --- LIMU-BERT ---
# From scratch (LIMU-BERT, seq_len=120, full sequence)
python train_loso.py --model limu-bert --device-type phone --no-bike
# From scratch (LIMU-BERT-X, seq_len=20, random 20-step crop from 120-step data)
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --no-bike
# With pretrained weights (MUST use --limu-seq-len 20 to match limu_bert_x.pt)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --no-bike
# Pretrained + half data
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.5 --no-bike

# --- ssl-wearables (ResNet / HARNet) ---
# Accel only (3ch)
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --no-bike
# Accel + Gyro (6ch)
python train_loso.py --model ssl-wearables --device-type phone --channels 6 --no-bike
# With pretrained HARNet weights
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.5 --no-bike

# --- ResNet-Baseline (basic 1D ResNet, no SSL pretraining) ---
# Uses same data as ssl-wearables; fair CNN baseline comparison
python train_loso.py --model resnet-baseline --device-type all --channels 6 --no-bike --experiment-tag exp0r_resnet_baseline
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --no-bike
```

Results are saved as JSON files in `loso_results/`.

Per-experiment summary CSVs are auto-generated in `loso_results/experiment_tables/`
when results are saved (`train_loso.py`, `eval_cross_dataset.py`,
`eval_custom_data.py`, `eval_unknown_activity.py`).

Manual summary commands:

```powershell
# Summarize one experiment (example: EXP8/EXP2....)
python summarize_experiment_results.py --exp exp8

# EXP number without prefix also works
python summarize_experiment_results.py --exp 8

# Summarize all experiments
python summarize_experiment_results.py --all
```

---

 ### Step 3 (Optional) -- Run Full Experiment Suite

The PowerShell script `run_experiments.ps1` automates everything (data prep + all 5 experiments).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `-Step` | string | `"all"` | Which step to run (see below) |
| `-GPU` | string | `"0"` | GPU id (`"-1"` for CPU) |
| `-Epochs` | int | 0 (=default) | Override epoch count for all runs |
| `-NoBike` | switch | off | Exclude bike activity (5 classes) |

#### Step Values

| Step | What it runs |
|---|---|
| `all` | Data preparation + Experiments 1--5 |
| `prepare` | Data preparation only |
| `exp1` | Experiment 1 only |
| `exp2` | Experiment 2 only |
| `exp3` | Experiment 3 only |
| `exp4` | Experiment 4 only |
| `exp5` | Experiment 5 only |
| `baselines` | All models from scratch (phone + watch, accel + gyro) |

#### Example Commands

```powershell
# Run everything (5 classes, no bike)
.\run_experiments.ps1 -NoBike

# Data preparation only
.\run_experiments.ps1 -Step prepare -NoBike

# Single experiment
.\run_experiments.ps1 -Step exp1 -NoBike
.\run_experiments.ps1 -Step exp3 -NoBike

# All baselines
.\run_experiments.ps1 -Step baselines -NoBike

# CPU only, custom epochs
.\run_experiments.ps1 -Step exp1 -GPU "-1" -Epochs 50 -NoBike

# Combine flags
.\run_experiments.ps1 -Step exp4 -GPU "0" -Epochs 100 -NoBike
```

---

## Experiment Definitions

### Experiment 0 -- Baseline Structural Comparison (From Scratch)

> **All 3 models**, trained from scratch, **no pretrained weights**, accel + gyro (6ch), LOSO, **5 classes (no bike)**.
>
> **Goal:** Compare the inherent generalization ability of different model architectures under identical conditions -- same data, same subjects, same sensor channels, same leave-one-subject-out protocol.

**Commands** (examples below use `--seed 42 --repeat-seeds 3` so you get repeated-run mean/std):

```powershell
# Step 1: Prepare data (generates phone / watch / all combined for each model)
python prepare_hhar_data.py --no-bike

# Step 2: Train all baselines
# --- HART (6ch only) ---
python train_loso.py --model hart --device-type phone --channels 6 --no-bike --experiment-tag exp0a_hart_phone
python train_loso.py --model hart --device-type watch --channels 6 --no-bike --experiment-tag exp0a_hart_watch
python train_loso.py --model hart --device-type all   --channels 6 --no-bike --experiment-tag exp0a_hart_all

# --- LIMU-BERT (6ch only) ---
python train_loso.py --model limu-bert --device-type phone --no-bike --experiment-tag exp0b_limubert_phone
python train_loso.py --model limu-bert --device-type watch --no-bike --experiment-tag exp0b_limubert_watch
python train_loso.py --model limu-bert --device-type all   --no-bike --experiment-tag exp0b_limubert_all

# --- LIMU-BERTX (6ch only) ---
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --no-bike --experiment-tag exp0c_limuX_scratch_phone
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --no-bike --experiment-tag exp0c_limuX_scratch_watch
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --no-bike --experiment-tag exp0c_limuX_scratch_all

# --- ssl-wearables 6ch ---
python train_loso.py --model ssl-wearables --device-type phone --channels 6 --no-bike --experiment-tag exp0d_ssl_phone_6ch
python train_loso.py --model ssl-wearables --device-type watch --channels 6 --no-bike --experiment-tag exp0d_ssl_watch_6ch
python train_loso.py --model ssl-wearables --device-type all   --channels 6 --no-bike --experiment-tag exp0d_ssl_all_6ch

# --- ssl-wearables 3ch (baseline for pretrained-weight experiments) ---
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --no-bike --experiment-tag exp0e_ssl_phone_3ch
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --no-bike --experiment-tag exp0e_ssl_watch_3ch
python train_loso.py --model ssl-wearables --device-type all   --channels 3 --no-bike --experiment-tag exp0e_ssl_all_3ch

# --- ResNet-Baseline 6ch (basic 1D ResNet, no SSL pretraining) ---
python train_loso.py --model resnet-baseline --device-type phone --channels 6 --no-bike --experiment-tag exp0f_resbase_phone_6ch
python train_loso.py --model resnet-baseline --device-type watch --channels 6 --no-bike --experiment-tag exp0f_resbase_watch_6ch
python train_loso.py --model resnet-baseline --device-type all   --channels 6 --no-bike --experiment-tag exp0f_resbase_all_6ch

# --- ResNet-Baseline 3ch ---
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --no-bike --experiment-tag exp0g_resbase_phone_3ch
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --no-bike --experiment-tag exp0g_resbase_watch_3ch
python train_loso.py --model resnet-baseline --device-type all   --channels 3 --no-bike --experiment-tag exp0g_resbase_all_3ch

# --- MobileHART (6ch only, uses same HART data pipeline) ---
python train_loso.py --model mobilehart --device-type phone --channels 6 --no-bike --experiment-tag exp0h_mobilehart_phone
python train_loso.py --model mobilehart --device-type watch --channels 6 --no-bike --experiment-tag exp0h_mobilehart_watch
python train_loso.py --model mobilehart --device-type all   --channels 6 --no-bike --experiment-tag exp0h_mobilehart_all
```

**Resume & Fold Control** (works for all models, `--resume` and `--start-fold` can be combined):

```powershell
# fold index: 0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g, 7=h, 8=i

# --- HART ---
python train_loso.py --model hart --device-type all --channels 6 --no-bike --experiment-tag exp0c_hart_all --resume
python train_loso.py --model hart --device-type all --channels 6 --no-bike --experiment-tag exp0c_hart_all --start-fold 3
python train_loso.py --model hart --device-type all --channels 6 --no-bike --experiment-tag exp0c_hart_all --resume --start-fold 2

# --- LIMU-BERT ---
python train_loso.py --model limu-bert --device-type all --no-bike --experiment-tag exp0f_limubert_all --resume
python train_loso.py --model limu-bert --device-type all --no-bike --experiment-tag exp0f_limubert_all --start-fold 3
python train_loso.py --model limu-bert --device-type all --no-bike --experiment-tag exp0f_limubert_all --resume --start-fold 2

# --- LIMU-BERT-X ---
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --no-bike --experiment-tag exp0c_limuX_scratch_all --resume
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --no-bike --experiment-tag exp0c_limuX_scratch_all --start-fold 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --no-bike --experiment-tag exp0c_limuX_scratch_all --resume --start-fold 2

# --- ssl-wearables ---
python train_loso.py --model ssl-wearables --device-type all --channels 6 --no-bike --experiment-tag exp0i_ssl_all_6ch --resume
python train_loso.py --model ssl-wearables --device-type all --channels 6 --no-bike --experiment-tag exp0i_ssl_all_6ch --start-fold 3
python train_loso.py --model ssl-wearables --device-type all --channels 6 --no-bike --experiment-tag exp0i_ssl_all_6ch --resume --start-fold 2

# --- ResNet-Baseline ---
python train_loso.py --model resnet-baseline --device-type all --channels 6 --no-bike --experiment-tag exp0o_resbase_all_6ch --resume
python train_loso.py --model resnet-baseline --device-type all --channels 6 --no-bike --experiment-tag exp0o_resbase_all_6ch --start-fold 3

# --- MobileHART ---
python train_loso.py --model mobilehart --device-type all --channels 6 --no-bike --experiment-tag exp0h_mobilehart_all --resume
python train_loso.py --model mobilehart --device-type all --channels 6 --no-bike --experiment-tag exp0h_mobilehart_all --start-fold 3
```

> **Or run all at once:** `.\run_experiments.ps1 -Step baselines -NoBike`

> **Note:** ssl-wearables 0J/0K/0L (3ch) are included because the original pretrained weights only support 3ch -- these 3ch scratch baselines serve as the reference for Experiments 4 and 5.

---

### Experiment 1 -- Can Pretraining Overcome Insufficient Supervised Learning?

> **5 classes (no bike)**
>
> **Goal:** When labelled data is scarce (half), can pretrained models match or exceed fully-supervised baselines?

**Part A -- Transformer: phone, 6ch (acc + gyro)**

| ID | Method | Data | Question |
|---|---|---|---|
| **1A** | LIMU-BERT pretrained, finetune | **Half** phone data | Does BERT pretraining compensate for limited labels? |
| **1B** | HART supervised (baseline) | Full phone data | Transformer supervised upper bound |
| **1C** | LIMU-BERT from scratch | Full phone data | BERT architecture without pretraining |

**Part B -- CNN: watch, 3ch (acc only -- pretrained weights are 3ch only)**

| ID | Method | Data | Question |
|---|---|---|---|
| **1D** | HARNet pretrained (UK Biobank), finetune | **Half** watch data | Does UK Biobank SSL pretraining compensate for limited labels? |
| **1E** | ResNet-Baseline supervised | Full watch data | CNN supervised upper bound |
| **1F** | HARNet from scratch | Full watch data | HARNet architecture without pretraining |

> **Note:** Exp 1A requires LIMU-BERT pretrained weights. Exp 1D requires pretrained `.mdl` weights in `ssl-wearables-main/model_check_point/` (3ch only).

**Commands:**

```powershell
python prepare_hhar_data.py --no-bike

# --- Part A: Transformer (phone, 6ch) ---
# 1A: LIMU-BERT-X pretrained, phone data fractions ---0.05/0.1/0.25/0.5/0.75/1.0
#     (--limu-seq-len 20 required to match limu_bert_x.pt pretrained weights)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.05 --no-bike --experiment-tag exp1a_limu_pretrained_05 --seed 42 --repeat-seeds 3

python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.1 --no-bike --experiment-tag exp1a_limu_pretrained_10 --seed 42 --repeat-seeds 3

python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.25 --no-bike --experiment-tag exp1a_limu_pretrained_25 --seed 42 --repeat-seeds 3

python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.5 --no-bike --experiment-tag exp1a_limu_pretrained_5 --seed 42 --repeat-seeds 3

python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 0.75 --no-bike --experiment-tag exp1a_limu_pretrained_75 --seed 42 --repeat-seeds 3

python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --data-fraction 1.0 --no-bike --experiment-tag exp1a_limu_pretrained_1 --seed 42 --repeat-seeds 3

# 1B: HART supervised, full phone data
python train_loso.py --model hart --device-type phone --channels 6 --no-bike --experiment-tag exp1b_hart_phone

# 1C: LIMU-BERT-X from scratch, full phone data (seq_len=20 for fair comparison with 1A)
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --no-bike --experiment-tag exp1c_limu_scratch_phone

# --- Part B: CNN (watch, 3ch) ---
# 1D: HARNet pretrained (UK Biobank), watch data fractions ---0.05/0.1/0.25/0.5/0.75/1.0
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.05 --no-bike --experiment-tag exp1d_ssl_pretrained_05 --seed 42 --repeat-seeds 3

python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.1 --no-bike --experiment-tag exp1d_ssl_pretrained_10 --seed 42 --repeat-seeds 3

python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.25 --no-bike --experiment-tag exp1d_ssl_pretrained_25 --seed 42 --repeat-seeds 3

python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.5 --no-bike --experiment-tag exp1d_ssl_pretrained_5 --seed 42 --repeat-seeds 3

python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.75 --no-bike --experiment-tag exp1d_ssl_pretrained_75 --seed 42 --repeat-seeds 3

python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp1d_ssl_pretrained_1 --seed 42 --repeat-seeds 3

# 1E: ResNet-Baseline supervised, full watch data
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --no-bike --experiment-tag exp1e_resbase_watch

# 1F: HARNet from scratch, full watch data
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --no-bike --experiment-tag exp1f_ssl_scratch_watch
```

### Experiment 2 -- Can BERT Be Suited for Smartwatch?

> Transformer models, **watch** data, accel + gyro (6ch), **5 classes (no bike)**
>
> **Goal:** LIMU-BERT-X was pretrained on phone IMU data. Can it adapt to smartwatch via self-supervised pretraining on external watch data (PAMAP2 wrist + WISDM watch)?  Compare against supervised-only baselines.

| ID | Step | Method | HHAR eval | Question |
|---|---|---|---|---|
| **2A** | Supervised LOSO (baseline) | LIMU-BERT-X from scratch | watch | Can BERT learn watch patterns with no pretraining? |
| **2B** | Supervised LOSO (baseline) | HART supervised | watch | Supervised upper bound |
| **2C-pt** | SSL pretrain | Finetune `limu_bert_x.pt` on external watch data | -- | SSL finetune existing phone weights on watch domain |
| **2C-w** | Supervised LOSO | 2C-pt weights | watch | Transfer watch-pretrained to HHAR watch |
| **2C-p** | Supervised LOSO | 2C-pt weights | phone | Transfer watch-pretrained to HHAR phone |
| **2C-a** | Supervised LOSO | 2C-pt weights | all | Transfer watch-pretrained to HHAR all devices |
| **2D-pt** | SSL pretrain from scratch | Pretrain LIMU-BERT-X from scratch on external watch data | -- | Learn representations with no prior |
| **2D** | Supervised LOSO | 2D-pt weights | watch | Pure watch-pretrained on HHAR watch |
| **2E-pt** | SSL pretrain | Finetune `limu_bert_x.pt` on WISDM all (watch+phone) | -- | Cross-device SSL finetune on WISDM |
| **2E-w** | Supervised LOSO | 2E-pt weights | watch | WISDM-all finetuned → HHAR watch |
| **2E-p** | Supervised LOSO | 2E-pt weights | phone | WISDM-all finetuned → HHAR phone |
| **2E-a** | Supervised LOSO | 2E-pt weights | all | WISDM-all finetuned → HHAR all |
| **2F-pt** | SSL pretrain from scratch | Pretrain LIMU-BERT-X from scratch on WISDM all (watch+phone) | -- | Cross-device representations, no prior |
| **2F-w** | Supervised LOSO | 2F-pt weights | watch | WISDM-all scratch → HHAR watch |
| **2F-p** | Supervised LOSO | 2F-pt weights | phone | WISDM-all scratch → HHAR phone |
| **2F-a** | Supervised LOSO | 2F-pt weights | all | WISDM-all scratch → HHAR all |

**Prerequisites -- Prepare external pretraining data:**

External datasets used (unlabeled, no HHAR):
- **PAMAP2 wrist**: body-worn IMU on dominant-arm wrist, accel + gyro (6ch), 100 Hz, 9 subjects
- **WISDM watch**: smartwatch accel + gyro (6ch), 20 Hz, 51 subjects
- **WISDM phone**: smartphone accel + gyro (6ch), 20 Hz, 51 subjects

```powershell
# Prepare unlabeled pretraining data (PAMAP2 wrist + WISDM watch → 20 Hz, 6ch)
python prepare_pretrain_data.py --limu
# Output: pretrain_data/limu_watch/data_20_120.npy  shape (N, 120, 6)

# Prepare WISDM-all pretraining data (WISDM watch + WISDM phone → 20 Hz, 6ch)
python prepare_pretrain_data.py --limu-wisdm-all
# Output: pretrain_data/limu_wisdm_all/data_20_120.npy  shape (N, 120, 6)
```

**Step 1 -- Self-supervised pretraining:**

```powershell
# 2C-pt: Finetune limu_bert_x.pt on external watch data
python pretrain_limu.py --data pretrain_data/limu_watch/data_20_120.npy --load-weights weights/limu_bert_x.pt --output pretrain_data/limu_watch_finetune.pt

# 2D-pt: Pretrain from scratch on external watch data
python pretrain_limu.py --data pretrain_data/limu_watch/data_20_120.npy --output pretrain_data/limu_watch_scratch.pt

# 2E-pt: Finetune limu_bert_x.pt on WISDM all (watch+phone)
python pretrain_limu.py --data pretrain_data/limu_wisdm_all/data_20_120.npy --load-weights weights/limu_bert_x.pt --output pretrain_data/limu_wisdm_all_finetune.pt

# 2F-pt: Pretrain from scratch on WISDM all (watch+phone)
python pretrain_limu.py --data pretrain_data/limu_wisdm_all/data_20_120.npy --output pretrain_data/limu_wisdm_all_scratch.pt
```

**Step 2 -- Supervised LOSO on HHAR (baselines + pretrained):**

```powershell
# 2A: LIMU-BERT-X from scratch (no pretraining, supervised only)
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --no-bike --experiment-tag exp2a_limu_scratch_watch

# 2B: HART supervised
python train_loso.py --model hart --device-type watch --channels 6 --no-bike --experiment-tag exp2b_hart_watch

# 2C: Watch-finetuned limu_bert_x → LOSO on watch / phone / full
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_watch_finetune.pt --device-type watch --no-bike --experiment-tag exp2c_limu_watchft_watch
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_watch_finetune.pt --device-type phone --no-bike --experiment-tag exp2c_limu_watchft_phone
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_watch_finetune.pt --device-type all --no-bike --experiment-tag exp2c_limu_watchft_all

# 2D: From-scratch watch pretrain → LOSO on watch
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_watch_scratch.pt --device-type watch --no-bike --experiment-tag exp2d_limu_watchscratch_watch

# 2E: WISDM-all finetuned → LOSO on watch / phone / all
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_finetune.pt --device-type watch --no-bike --experiment-tag exp2e_limu_wisdmft_watch
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_finetune.pt --device-type phone --no-bike --experiment-tag exp2e_limu_wisdmft_phone
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_finetune.pt --device-type all --no-bike --experiment-tag exp2e_limu_wisdmft_all

# 2F: WISDM-all from-scratch pretrain → LOSO on watch / phone / all
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_scratch.pt --device-type watch --no-bike --experiment-tag exp2f_limu_wisdmscratch_watch
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_scratch.pt --device-type phone --no-bike --experiment-tag exp2f_limu_wisdmscratch_phone
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_scratch.pt --device-type all --no-bike --experiment-tag exp2f_limu_wisdmscratch_all
```

### Experiment 3 -- Can HARNet Be Suited for Smartphone?

> CNN/ResNet models, **phone** data, **5 classes (no bike)**
>
> **Goal:** HARNet was pretrained on UK Biobank wrist-worn accelerometer data (3ch). Can it transfer to smartphone via self-supervised pretraining on external phone data (SBHAR + WISDM phone)?

| ID | Step | Method | Ch | HHAR eval | Question |
|---|---|---|---|---|---|
| **3A** | Supervised LOSO (baseline) | HARNet from scratch | 3 | phone | Can HARNet learn phone patterns with no pretraining? |
| **3B** | Supervised LOSO (baseline) | ResNet-Baseline supervised | 3 | phone | CNN supervised upper bound |
| **3C-pt** | SSL pretrain | Finetune `mtl_best.mdl` on external phone data | 3 | -- | SSL finetune existing watch weights on phone domain |
| **3C-w** | Supervised LOSO | 3C-pt weights | 3 | watch | Transfer phone-pretrained to HHAR watch |
| **3C-p** | Supervised LOSO | 3C-pt weights | 3 | phone | Transfer phone-pretrained to HHAR phone |
| **3C-a** | Supervised LOSO | 3C-pt weights | 3 | all | Transfer phone-pretrained to HHAR all devices |
| **3D-pt-3ch** | SSL pretrain from scratch | Pretrain HARNet from scratch (3ch) | 3 | -- | Learn phone representations, accel only |
| **3D-3ch** | Supervised LOSO | 3D-pt-3ch weights | 3 | phone | Pure phone-pretrained (3ch) on HHAR phone |
| **3D-pt-6ch** | SSL pretrain from scratch | Pretrain HARNet from scratch (6ch) | 6 | -- | Learn phone representations, accel+gyro |
| **3D-6ch** | Supervised LOSO | 3D-pt-6ch weights | 6 | phone | Pure phone-pretrained (6ch) on HHAR phone |
| **3E-pt** | SSL pretrain | Finetune `mtl_best.mdl` on WISDM all (watch+phone, 3ch) | 3 | -- | Cross-device SSL finetune on WISDM |
| **3E-w** | Supervised LOSO | 3E-pt weights | 3 | watch | WISDM-all finetuned → HHAR watch |
| **3E-p** | Supervised LOSO | 3E-pt weights | 3 | phone | WISDM-all finetuned → HHAR phone |
| **3E-a** | Supervised LOSO | 3E-pt weights | 3 | all | WISDM-all finetuned → HHAR all |
| **3F-pt-3ch** | SSL pretrain from scratch | Pretrain HARNet from scratch on WISDM all (3ch) | 3 | -- | Cross-device representations, accel only |
| **3F-3ch** | Supervised LOSO | 3F-pt-3ch weights | 3 | phone | WISDM-all scratch (3ch) → HHAR phone |
| **3F-pt-6ch** | SSL pretrain from scratch | Pretrain HARNet from scratch on WISDM all (6ch) | 6 | -- | Cross-device representations, accel+gyro |
| **3F-6ch** | Supervised LOSO | 3F-pt-6ch weights | 6 | phone | WISDM-all scratch (6ch) → HHAR phone |

**Prerequisites -- Prepare external pretraining data:**

External datasets used (unlabeled, no HHAR):
- **SBHAR**: smartphone (Samsung Galaxy SII), accel + gyro, 50 Hz, 30 subjects
- **WISDM phone**: smartphone accel + gyro, 20 Hz, 51 subjects
- **WISDM watch**: smartwatch accel + gyro, 20 Hz, 51 subjects

```powershell
# Prepare unlabeled pretraining data (SBHAR + WISDM phone → 30 Hz)
python prepare_pretrain_data.py --ssl          # 3ch (accel only)
python prepare_pretrain_data.py --ssl-6ch      # 6ch (accel + gyro)
# Output: pretrain_data/ssl_phone_3ch/  and  pretrain_data/ssl_phone_6ch/

# Prepare WISDM-all pretraining data (WISDM watch + WISDM phone → 30 Hz)
python prepare_pretrain_data.py --ssl-wisdm-all       # 3ch
python prepare_pretrain_data.py --ssl-wisdm-all-6ch   # 6ch
# Output: pretrain_data/ssl_wisdm_all_3ch/  and  pretrain_data/ssl_wisdm_all_6ch/
```

**Step 1 -- Self-supervised pretraining:**

```powershell
# 3C-pt: Finetune mtl_best.mdl on external phone data (3ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_phone_3ch --load-weights model_check_point/mtl_best.mdl --channels 3 --output pretrain_data/ssl_phone_finetune_3ch.mdl

# 3D-pt-3ch: Pretrain from scratch (3ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_phone_3ch --channels 3 --output pretrain_data/ssl_phone_scratch_3ch.mdl

# 3D-pt-6ch: Pretrain from scratch (6ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_phone_6ch --channels 6 --output pretrain_data/ssl_phone_scratch_6ch.mdl

# 3E-pt: Finetune mtl_best.mdl on WISDM all (watch+phone, 3ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_wisdm_all_3ch --load-weights model_check_point/mtl_best.mdl --channels 3 --output pretrain_data/ssl_wisdm_all_finetune_3ch.mdl

# 3F-pt-3ch: Pretrain from scratch on WISDM all (3ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_wisdm_all_3ch --channels 3 --output pretrain_data/ssl_wisdm_all_scratch_3ch.mdl

# 3F-pt-6ch: Pretrain from scratch on WISDM all (6ch)
python pretrain_ssl.py --data-dir pretrain_data/ssl_wisdm_all_6ch --channels 6 --output pretrain_data/ssl_wisdm_all_scratch_6ch.mdl
```

**Step 2 -- Supervised LOSO on HHAR (baselines + pretrained):**

```powershell
# 3A: HARNet from scratch (3ch)
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --no-bike --experiment-tag exp3a_ssl_scratch_phone

# 3B: ResNet-Baseline supervised (3ch)
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --no-bike --experiment-tag exp3b_resbase_phone

# 3C: Phone-finetuned mtl_best → LOSO on watch / phone / full (3ch)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_finetune_3ch.mdl --device-type watch --channels 3 --no-bike --experiment-tag exp3c_ssl_phoneft_watch
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_finetune_3ch.mdl --device-type phone --channels 3 --no-bike --experiment-tag exp3c_ssl_phoneft_phone
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_finetune_3ch.mdl --device-type all --channels 3 --no-bike --experiment-tag exp3c_ssl_phoneft_all

# 3D-3ch: From-scratch phone pretrain (3ch) → LOSO on phone
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_scratch_3ch.mdl --device-type phone --channels 3 --no-bike --experiment-tag exp3d_ssl_phonescratch3_phone

# 3D-6ch: From-scratch phone pretrain (6ch) → LOSO on phone
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_scratch_6ch.mdl --device-type phone --channels 6 --no-bike --experiment-tag exp3d_ssl_phonescratch6_phone

# 3E: WISDM-all finetuned (3ch) → LOSO on watch / phone / all
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_finetune_3ch.mdl --device-type watch --channels 3 --no-bike --experiment-tag exp3e_ssl_wisdmft_watch
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_finetune_3ch.mdl --device-type phone --channels 3 --no-bike --experiment-tag exp3e_ssl_wisdmft_phone
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_finetune_3ch.mdl --device-type all --channels 3 --no-bike --experiment-tag exp3e_ssl_wisdmft_all

# 3F-3ch: WISDM-all from-scratch (3ch) → LOSO on watch / phone / all
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_3ch.mdl --device-type watch --channels 3 --no-bike --experiment-tag exp3f_ssl_wisdmscratch3_watch --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_3ch.mdl --device-type phone --channels 3 --no-bike --experiment-tag exp3f_ssl_wisdmscratch3_phone --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_3ch.mdl --device-type all --channels 3 --no-bike --experiment-tag exp3f_ssl_wisdmscratch3_all --seed 42 --repeat-seeds 3

# 3F-6ch: WISDM-all from-scratch (6ch) → LOSO on watch / phone / all
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_6ch.mdl --device-type watch --channels 6 --no-bike --experiment-tag exp3f_ssl_wisdmscratch6_watch --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_6ch.mdl --device-type phone --channels 6 --no-bike --experiment-tag exp3f_ssl_wisdmscratch6_phone --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_6ch.mdl --device-type all --channels 6 --no-bike --experiment-tag exp3f_ssl_wisdmscratch6_all --seed 42 --repeat-seeds 3
```

### Experiment 4 -- Sensor Self-Comparison

>
> **Goal:** How much does adding gyroscope data improve HART's performance over accelerometer alone?

**Commands:**

```powershell
python prepare_hhar_data.py --hart --no-bike

# --- ssl-wearables 6ch ---
python train_loso.py --model ssl-wearables --device-type phone --channels 6 --no-bike --experiment-tag exp0g_ssl_phone_6ch
python train_loso.py --model ssl-wearables --device-type watch --channels 6 --no-bike --experiment-tag exp0h_ssl_watch_6ch
python train_loso.py --model ssl-wearables --device-type all   --channels 6 --no-bike --experiment-tag exp0i_ssl_all_6ch

# --- ssl-wearables 3ch (baseline for pretrained-weight experiments) ---
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --no-bike --experiment-tag exp0j_ssl_phone_3ch
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --no-bike --experiment-tag exp0k_ssl_watch_3ch
python train_loso.py --model ssl-wearables --device-type all   --channels 3 --no-bike --experiment-tag exp0l_ssl_all_3ch

# --- ResNet-Baseline 6ch (basic 1D ResNet, no SSL pretraining) ---
python train_loso.py --model resnet-baseline --device-type phone --channels 6 --no-bike --experiment-tag exp0m_resbase_phone_6ch
python train_loso.py --model resnet-baseline --device-type watch --channels 6 --no-bike --experiment-tag exp0n_resbase_watch_6ch
python train_loso.py --model resnet-baseline --device-type all   --channels 6 --no-bike --experiment-tag exp0o_resbase_all_6ch

# --- ResNet-Baseline 3ch ---
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --no-bike --experiment-tag exp0p_resbase_phone_3ch
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --no-bike --experiment-tag exp0q_resbase_watch_3ch
python train_loso.py --model resnet-baseline --device-type all   --channels 3 --no-bike --experiment-tag exp0r_resbase_all_3ch
```

---

### Experiment 5 -- Cross-Dataset Generalizability (HHAR -> WISDM)

> **Goal:** Can models trained on all HHAR subjects generalise to an unseen dataset (WISDM)?
>
> No LOSO -- train on **all** HHAR subjects, test on **all** WISDM subjects.
> Two modes: **3-class** (sit/stand/walk) and **5-class + stairs analysis** (`--with-stairs`).
>
> **Reporting note:** A single run reports one set of test metrics only, so there is no LOSO-style fold-wise standard deviation by default. If you want uncertainty estimates, rerun with different seeds and report repeated-run mean/std.

| ID | Model | Device | Channels | Tag |
|---|---|---|---|---|
| **5A** | HART | phone | 6ch | `exp5_hart_phone` |
| **5B** | HART | watch | 6ch | `exp5_hart_watch` |
| **5C** | HART | all | 6ch | `exp5_hart_all` |
| **5D** | LIMU-BERT | phone | 6ch | `exp5_limubert_phone` |
| **5E** | LIMU-BERT | watch | 6ch | `exp5_limubert_watch` |
| **5F** | LIMU-BERT | all | 6ch | `exp5_limubert_all` |
| **5G** | LIMU-BERT-X | phone | 6ch | `exp5_limubertx_phone` |
| **5H** | LIMU-BERT-X | watch | 6ch | `exp5_limubertx_watch` |
| **5I** | LIMU-BERT-X | all | 6ch | `exp5_limubertx_all` |
| **5J** | ssl-wearables | phone | 6ch | `exp5_ssl_phone_6ch` |
| **5K** | ssl-wearables | watch | 6ch | `exp5_ssl_watch_6ch` |
| **5L** | ssl-wearables | all | 6ch | `exp5_ssl_all_6ch` |
| **5M** | ssl-wearables | phone | 3ch | `exp5_ssl_phone_3ch` |
| **5N** | ssl-wearables | watch | 3ch | `exp5_ssl_watch_3ch` |
| **5O** | ssl-wearables | all | 3ch | `exp5_ssl_all_3ch` |
| **5P** | ResNet-Baseline | phone | 6ch | `exp5_resbase_phone_6ch` |
| **5Q** | ResNet-Baseline | watch | 6ch | `exp5_resbase_watch_6ch` |
| **5R** | ResNet-Baseline | all | 6ch | `exp5_resbase_all_6ch` |
| **5S** | ResNet-Baseline | phone | 3ch | `exp5_resbase_phone_3ch` |
| **5T** | ResNet-Baseline | watch | 3ch | `exp5_resbase_watch_3ch` |
| **5U** | ResNet-Baseline | all | 3ch | `exp5_resbase_all_3ch` |

**Commands:**

```powershell
# Step 1: Prepare WISDM data (all models, phone + watch + all)
python prepare_wisdm_data.py

# Step 2: 3-class cross-dataset evaluation (sit / stand / walk)
# --- HART 6ch ---
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --experiment-tag exp5_hart_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type watch --channels 6 --experiment-tag exp5_hart_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type all   --channels 6 --experiment-tag exp5_hart_all --seed 42 --repeat-seeds 3

# --- LIMU-BERT-X 6ch ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type phone --experiment-tag exp5_limubertx_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type watch --experiment-tag exp5_limubertx_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all   --experiment-tag exp5_limubertx_all --seed 42 --repeat-seeds 3

# --- ssl-wearables 6ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 6 --experiment-tag exp5_ssl_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 6 --experiment-tag exp5_ssl_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 6 --experiment-tag exp5_ssl_all_6ch --seed 42 --repeat-seeds 3

# --- ssl-wearables 3ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 3 --experiment-tag exp5_ssl_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 3 --experiment-tag exp5_ssl_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 3 --experiment-tag exp5_ssl_all_3ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 6ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 6 --experiment-tag exp5_resbase_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 6 --experiment-tag exp5_resbase_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 6 --experiment-tag exp5_resbase_all_6ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 3ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 3 --experiment-tag exp5_resbase_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 3 --experiment-tag exp5_resbase_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 3 --experiment-tag exp5_resbase_all_3ch --seed 42 --repeat-seeds 3
```

**Repeated-run mean/std** (recommended when you want uncertainty, since this is not LOSO):

```powershell
# Example: 3 repeated seeds for HART all
python eval_cross_dataset.py --model hart --device-type all --channels 6 --experiment-tag exp5_hart_all --seed 42 --repeat-seeds 3
```

**With stairs analysis** (append `--with-stairs`, prefix tag with `exp5s_`):

```powershell
# Example: HART phone with stairs analysis + repeated seeds
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --with-stairs --experiment-tag exp5s_hart_phone --seed 42 --repeat-seeds 3

# Example: LIMU-BERT-X all with stairs analysis + repeated seeds
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all --with-stairs --experiment-tag exp5s_limubertx_all --seed 42 --repeat-seeds 3
```

> **Note:** `--with-stairs` trains HHAR with 5 classes (sit/stand/walk/upstair/downstairs), evaluates WISDM main metrics on 3 classes (sit/stand/walk), and adds a supplementary analysis showing how the model classifies WISDM "stairs" samples across the 5 HHAR classes. Default mode (no flag) trains and tests purely on 3 classes.

---

### Experiment 6 -- Cross-Dataset Generalizability (WISDM -> HHAR)

> **Goal:** Reverse of Exp 5 -- train on all WISDM subjects, test on all HHAR subjects.
>
> Uses the same `eval_cross_dataset.py` script with `--reverse` flag.
> WISDM has ~2.5x more training data than HHAR, testing cross-dataset transfer in the other direction.
>
> **Reporting note:** As in Exp 5, a single run gives one test result only. There is no LOSO-style fold-wise standard deviation unless you add repeated runs manually with different seeds.

| ID | Model | Device | Channels | Tag |
|---|---|---|---|---|
| **6A** | HART | phone | 6ch | `exp6_hart_phone` |
| **6B** | HART | watch | 6ch | `exp6_hart_watch` |
| **6C** | HART | all | 6ch | `exp6_hart_all` |
| **6D** | LIMU-BERT | phone | 6ch | `exp6_limubert_phone` |
| **6E** | LIMU-BERT | watch | 6ch | `exp6_limubert_watch` |
| **6F** | LIMU-BERT | all | 6ch | `exp6_limubert_all` |
| **6G** | LIMU-BERT-X | phone | 6ch | `exp6_limubertx_phone` |
| **6H** | LIMU-BERT-X | watch | 6ch | `exp6_limubertx_watch` |
| **6I** | LIMU-BERT-X | all | 6ch | `exp6_limubertx_all` |
| **6J** | ssl-wearables | phone | 6ch | `exp6_ssl_phone_6ch` |
| **6K** | ssl-wearables | watch | 6ch | `exp6_ssl_watch_6ch` |
| **6L** | ssl-wearables | all | 6ch | `exp6_ssl_all_6ch` |
| **6M** | ssl-wearables | phone | 3ch | `exp6_ssl_phone_3ch` |
| **6N** | ssl-wearables | watch | 3ch | `exp6_ssl_watch_3ch` |
| **6O** | ssl-wearables | all | 3ch | `exp6_ssl_all_3ch` |
| **6P** | ResNet-Baseline | phone | 6ch | `exp6_resbase_phone_6ch` |
| **6Q** | ResNet-Baseline | watch | 6ch | `exp6_resbase_watch_6ch` |
| **6R** | ResNet-Baseline | all | 6ch | `exp6_resbase_all_6ch` |
| **6S** | ResNet-Baseline | phone | 3ch | `exp6_resbase_phone_3ch` |
| **6T** | ResNet-Baseline | watch | 3ch | `exp6_resbase_watch_3ch` |
| **6U** | ResNet-Baseline | all | 3ch | `exp6_resbase_all_3ch` |

**Commands** (same as Exp 5 but with `--reverse`; examples below use `--seed 42 --repeat-seeds 3` so you get repeated-run mean/std):

```powershell
# Prerequisites: same as Exp 5 (prepare_wisdm_data.py already run)

# --- HART 6ch ---
# Use a smaller batch size for 4 GB GPUs.
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --batch-size 64 --reverse --experiment-tag exp6_hart_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type watch --channels 6 --batch-size 64 --reverse --experiment-tag exp6_hart_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type all   --channels 6 --batch-size 64 --reverse --experiment-tag exp6_hart_all --seed 42 --repeat-seeds 3

# --- LIMU-BERT-X 6ch ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type phone --reverse --experiment-tag exp6_limubertx_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type watch --reverse --experiment-tag exp6_limubertx_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all   --reverse --experiment-tag exp6_limubertx_all --seed 42 --repeat-seeds 3

# --- ssl-wearables 6ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 6 --reverse --experiment-tag exp6_ssl_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 6 --reverse --experiment-tag exp6_ssl_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 6 --reverse --experiment-tag exp6_ssl_all_6ch --seed 42 --repeat-seeds 3

# --- ssl-wearables 3ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 3 --reverse --experiment-tag exp6_ssl_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 3 --reverse --experiment-tag exp6_ssl_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 3 --reverse --experiment-tag exp6_ssl_all_3ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 6ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 6 --reverse --experiment-tag exp6_resbase_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 6 --reverse --experiment-tag exp6_resbase_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 6 --reverse --experiment-tag exp6_resbase_all_6ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 3ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 3 --reverse --experiment-tag exp6_resbase_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 3 --reverse --experiment-tag exp6_resbase_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 3 --reverse --experiment-tag exp6_resbase_all_3ch --seed 42 --repeat-seeds 3
```
### Additional Training / Priority Queue (Exp5 & Exp6 first)

> Do **3 trials minimum per configuration** (`--seed 42 --repeat-seeds 3`).
> Cross-dataset outputs include **Accuracy + F1 weighted + F1 macro** (saved in JSON/CSV).
> For repeated runs, report **mean ± std** for accuracy/F1.
> Variables kept with the same names as requested:
> `Architecture Type`, `Device Type`, `Training Dataset`, `Testing Dataset`,
> `3ch vs 6ch data`, `Imported Pretrained Weights`, `Additional Pretraining dataset`,
> `Data Fraction`, `Testing Method`.

| Architecture Type | Device Type | Training Dataset | Testing Dataset | 3ch vs 6ch data | Imported Pretrained Weights | Additional Pretraining dataset | Data Fraction | Testing Method |
|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | HHAR | WISDM | 6ch | Yes | None | 100% | Cross |
| SSL-Wearables | Watch | HHAR | WISDM | 3ch | Yes | None | 100% | Cross |
| LIMU-BERT-X | Phone | WISDM | HHAR | 6ch | Yes | None | 100% | Cross |
| SSL-Wearables | Watch | WISDM | HHAR | 3ch | Yes | None | 100% | Cross |

> Note: `eval_cross_dataset.py` now supports `--pretrained` for `limu-bert` and `ssl-wearables`.
> The commands below explicitly load imported pretrained weights (`Imported Pretrained Weights = Yes`).

**Commands (Exp5/6 priority):**

```powershell
# --- Exp5: HHAR -> WISDM (Cross) ---
# Run 1: LIMU-BERT-X, phone, 6ch
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --data-fraction 1.0 --experiment-tag exp5_add_limuX_phone_6ch --seed 42 --repeat-seeds 3

# Run 2: SSL-Wearables, watch, 3ch
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 1.0 --experiment-tag exp5_add_ssl_watch_3ch --seed 42 --repeat-seeds 3

# --- Exp6: WISDM -> HHAR (Cross, reverse) ---
# Run 3: LIMU-BERT-X, phone, 6ch
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --data-fraction 1.0 --reverse --experiment-tag exp6_add_limuX_phone_6ch --seed 42 --repeat-seeds 3

# Run 4: SSL-Wearables, watch, 3ch
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 1.0 --reverse --experiment-tag exp6_add_ssl_watch_3ch --seed 42 --repeat-seeds 3
```

---

### Experiment 5.1 -- Cross-Dataset Generalizability (HHAR -> PAMAP2+SBHAR)

> **Goal:** Can models trained on all HHAR subjects generalise to a combined PAMAP2+SBHAR dataset?
>
> Same protocol as Exp 5 but the test set is PAMAP2 (wrist/watch) + SBHAR (phone) instead of WISDM.
> **5-class** (sit/stand/walk/upstairs/downstairs). No LOSO — train on **all** HHAR, test on **all** PAMAP2+SBHAR.
> Device-type mapping: `phone` → SBHAR only, `watch` → PAMAP2 only, `all` → both combined.
>
> **Prerequisite:** Data must be prepared for PAMAP2+SBHAR in all model formats.
> Uses `eval_cross_dataset.py` with `--target-dataset pamap2_sbhar`.

| ID | Model | Device | Channels | Pretrained | Tag |
|---|---|---|---|---|---|
| **5.1A** | HART | phone | 6ch | No | `exp51_hart_phone` |
| **5.1B** | HART | watch | 6ch | No | `exp51_hart_watch` |
| **5.1C** | HART | all | 6ch | No | `exp51_hart_all` |
| **5.1D** | LIMU-BERT | phone | 6ch | No | `exp51_limubert_phone` |
| **5.1E** | LIMU-BERT | watch | 6ch | No | `exp51_limubert_watch` |
| **5.1F** | LIMU-BERT | all | 6ch | No | `exp51_limubert_all` |
| **5.1G** | LIMU-BERT-X | phone | 6ch | No | `exp51_limubertx_phone` |
| **5.1H** | LIMU-BERT-X | watch | 6ch | No | `exp51_limubertx_watch` |
| **5.1I** | LIMU-BERT-X | all | 6ch | No | `exp51_limubertx_all` |
| **5.1J** | ssl-wearables | phone | 6ch | No | `exp51_ssl_phone_6ch` |
| **5.1K** | ssl-wearables | watch | 6ch | No | `exp51_ssl_watch_6ch` |
| **5.1L** | ssl-wearables | all | 6ch | No | `exp51_ssl_all_6ch` |
| **5.1M** | ssl-wearables | phone | 3ch | No | `exp51_ssl_phone_3ch` |
| **5.1N** | ssl-wearables | watch | 3ch | No | `exp51_ssl_watch_3ch` |
| **5.1O** | ssl-wearables | all | 3ch | No | `exp51_ssl_all_3ch` |
| **5.1P** | ResNet-Baseline | phone | 6ch | No | `exp51_resbase_phone_6ch` |
| **5.1Q** | ResNet-Baseline | watch | 6ch | No | `exp51_resbase_watch_6ch` |
| **5.1R** | ResNet-Baseline | all | 6ch | No | `exp51_resbase_all_6ch` |
| **5.1S** | ResNet-Baseline | phone | 3ch | No | `exp51_resbase_phone_3ch` |
| **5.1T** | ResNet-Baseline | watch | 3ch | No | `exp51_resbase_watch_3ch` |
| **5.1U** | ResNet-Baseline | all | 3ch | No | `exp51_resbase_all_3ch` |
| **5.1V** | LIMU-BERT-X | phone | 6ch | Yes | `exp51_add_limuX_phone_6ch` |
| **5.1W** | ssl-wearables | watch | 3ch | Yes | `exp51_add_ssl_watch_3ch` |

**Commands:**

```powershell
# Prerequisites: prepare PAMAP2+SBHAR aligned data (8+8 subjects)
# python prepare_pamap2_sbhar_data.py --aligned

# --- HART 6ch ---
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_hart_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type watch --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_hart_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type all   --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_hart_all --seed 42 --repeat-seeds 3

# --- LIMU-BERT-X 6ch ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_limubertx_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_limubertx_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all   --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_limubertx_all --seed 42 --repeat-seeds 3

# --- ssl-wearables 6ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_all_6ch --seed 42 --repeat-seeds 3

# --- ssl-wearables 3ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_ssl_all_3ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 6ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 6 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_all_6ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 3ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 3 --target-dataset pamap2_sbhar --with-stairs --experiment-tag exp51_resbase_all_3ch --seed 42 --repeat-seeds 3

# --- Imported Pretrained Weights = Yes ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --data-fraction 1.0 --experiment-tag exp51_add_limuX_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --data-fraction 1.0 --experiment-tag exp51_add_ssl_watch_3ch --seed 42 --repeat-seeds 3
```

> Notes:
> - `--with-stairs` enables 5-class mode (sit/stand/walk/upstairs/downstairs). Remove it to fall back to 3-class.
> - `--target-dataset pamap2_sbhar` = aligned 8+8 (default). Use `pamap2_sbhar_full` for the full 8+30 variant.
> - PAMAP2 contributes watch/wrist data; SBHAR contributes phone data.

---

### Experiment 6.1 -- Cross-Dataset Generalizability (PAMAP2+SBHAR -> HHAR)

> **Goal:** Reverse of Exp 5.1 — train on all PAMAP2+SBHAR subjects, test on all HHAR subjects.
>
> Same protocol as Exp 6 but training set is PAMAP2 (wrist/watch) + SBHAR (phone) instead of WISDM.
> Uses `eval_cross_dataset.py` with `--target-dataset pamap2_sbhar --reverse`.
>
> **Reporting note:** As in Exp 5.1, a single run gives one test result only. Use `--repeat-seeds 3` for mean ± std.

| ID | Model | Device | Channels | Pretrained | Tag |
|---|---|---|---|---|---|
| **6.1A** | HART | phone | 6ch | No | `exp61_hart_phone` |
| **6.1B** | HART | watch | 6ch | No | `exp61_hart_watch` |
| **6.1C** | HART | all | 6ch | No | `exp61_hart_all` |
| **6.1D** | LIMU-BERT | phone | 6ch | No | `exp61_limubert_phone` |
| **6.1E** | LIMU-BERT | watch | 6ch | No | `exp61_limubert_watch` |
| **6.1F** | LIMU-BERT | all | 6ch | No | `exp61_limubert_all` |
| **6.1G** | LIMU-BERT-X | phone | 6ch | No | `exp61_limubertx_phone` |
| **6.1H** | LIMU-BERT-X | watch | 6ch | No | `exp61_limubertx_watch` |
| **6.1I** | LIMU-BERT-X | all | 6ch | No | `exp61_limubertx_all` |
| **6.1J** | ssl-wearables | phone | 6ch | No | `exp61_ssl_phone_6ch` |
| **6.1K** | ssl-wearables | watch | 6ch | No | `exp61_ssl_watch_6ch` |
| **6.1L** | ssl-wearables | all | 6ch | No | `exp61_ssl_all_6ch` |
| **6.1M** | ssl-wearables | phone | 3ch | No | `exp61_ssl_phone_3ch` |
| **6.1N** | ssl-wearables | watch | 3ch | No | `exp61_ssl_watch_3ch` |
| **6.1O** | ssl-wearables | all | 3ch | No | `exp61_ssl_all_3ch` |
| **6.1P** | ResNet-Baseline | phone | 6ch | No | `exp61_resbase_phone_6ch` |
| **6.1Q** | ResNet-Baseline | watch | 6ch | No | `exp61_resbase_watch_6ch` |
| **6.1R** | ResNet-Baseline | all | 6ch | No | `exp61_resbase_all_6ch` |
| **6.1S** | ResNet-Baseline | phone | 3ch | No | `exp61_resbase_phone_3ch` |
| **6.1T** | ResNet-Baseline | watch | 3ch | No | `exp61_resbase_watch_3ch` |
| **6.1U** | ResNet-Baseline | all | 3ch | No | `exp61_resbase_all_3ch` |
| **6.1V** | LIMU-BERT-X | phone | 6ch | Yes | `exp61_add_limuX_phone_6ch` |
| **6.1W** | ssl-wearables | watch | 3ch | Yes | `exp61_add_ssl_watch_3ch` |

**Commands** (same as Exp 5.1 but with `--reverse`):

```powershell
# Prerequisites: same as Exp 5.1 (python prepare_pamap2_sbhar_data.py --aligned)

# --- HART 6ch ---
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --batch-size 64 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_hart_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type watch --channels 6 --batch-size 64 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_hart_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model hart --device-type all   --channels 6 --batch-size 64 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_hart_all --seed 42 --repeat-seeds 3

# --- LIMU-BERT-X 6ch ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_limubertx_phone --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_limubertx_watch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all   --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_limubertx_all --seed 42 --repeat-seeds 3

# --- ssl-wearables 6ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_all_6ch --seed 42 --repeat-seeds 3

# --- ssl-wearables 3ch ---
python eval_cross_dataset.py --model ssl-wearables --device-type phone --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --device-type all   --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_ssl_all_3ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 6ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_watch_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 6 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_all_6ch --seed 42 --repeat-seeds 3

# --- ResNet-Baseline 3ch ---
python eval_cross_dataset.py --model resnet-baseline --device-type phone --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_phone_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_watch_3ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model resnet-baseline --device-type all   --channels 3 --target-dataset pamap2_sbhar --with-stairs --reverse --experiment-tag exp61_resbase_all_3ch --seed 42 --repeat-seeds 3

# --- Imported Pretrained Weights = Yes ---
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --target-dataset pamap2_sbhar --with-stairs --data-fraction 1.0 --reverse --experiment-tag exp61_add_limuX_phone_6ch --seed 42 --repeat-seeds 3
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --target-dataset pamap2_sbhar --with-stairs --data-fraction 1.0 --reverse --experiment-tag exp61_add_ssl_watch_3ch --seed 42 --repeat-seeds 3
```

> Notes:
> - Same notes as Exp 5.1. `--reverse` swaps direction: PAMAP2+SBHAR trains, HHAR tests.
> - HART uses `--batch-size 64` for reverse direction (matching Exp 6 convention for smaller GPUs).
> - `--target-dataset pamap2_sbhar` = aligned 8+8 (default). Use `pamap2_sbhar_full` for the full 8+30 variant.
> - Compare Exp5.1 vs Exp6.1 to assess bidirectional transferability between HHAR and PAMAP2+SBHAR, analogous to Exp5 vs Exp6 for HHAR ↔ WISDM.

---


### Experiment 5.2 -- Cross-Dataset Fine-Tuning (HHAR -> finetune Target -> test Target)

> **Goal:** Train from scratch on ALL HHAR, then fine-tune on target-dataset data and test on remaining target data.
>
> **Two separate protocols (mutually exclusive flags):**
>
> | Protocol | Flag | What varies | Fine-tune data | Test data |
> |---|---|---|---|---|
> | **A: Leave-N-in** | `--finetune-subjects N` | # of subjects | 100% of N randomly-selected subjects | All remaining subjects |
> | **B: Data-fraction** | `--per-subject-fraction F` | % of data per subject | F% from EACH subject | Remaining (1−F)% from each subject |
>
> **Script:** `eval_cross_finetune.py`
>
> **Models:** LIMU-BERT-X (6ch) and SSL-Wearables (3ch only). No pretrained weights.
>
> **Classification:** WISDM = 3-class (sit/stand/walk). PAMAP2+SBHAR = 5-class (`--with-stairs`).
>
> **Stage 1 caching:** HHAR training is cached per (model, device, num_classes). E.g. `stage1_limubertx_phone_3cls.pt`.
> ALL subsequent runs (any protocol, any target, any seed) with the same class count skip Stage 1 and load the cache.
> Use `--force-stage1` to retrain. **8 unique Stage 1 trainings** needed (2 models × 3 devices × but split by 3cls/5cls).
>
> **Repeats:** `--repeat-seeds 3`. Seeds only affect Stage 2 (subject selection / data split / fine-tuning).

**Experiment Matrix:**

| | Leave-N-in (Protocol A) | Data-fraction (Protocol B) |
|---|---|---|
| Target dataset | WISDM (3cls), PAMAP2+SBHAR (5cls) (2) | same (2) |
| Model × Ch | LIMU-BERT-X 6ch, SSL 3ch (2) | same (2) |
| Device | phone, watch, all (3) | same (3) |
| Varying dim | WISDM: N = 1, 5, 25, 40 (4); P2SB: N = 1, 5 (2) | F = 10%, 25%, 75% (3) |
| **Subtotal** | **WISDM: 2×3×4=24 + P2SB: 2×3×2=12 → 36** | **2 × 2 × 3 × 3 = 36** |
| | | **Grand total: 72 commands** |

Tag patterns:
- Leave-in: `exp52_{model}_{device}_{ch}ch_{target}_leavein{N}`
- Data-fraction: `exp52_{model}_{device}_{ch}ch_{target}_frac{pct}`

#### 5.2-A: Leave-N-in — WISDM, 3-class (24 commands)

| ID | Model | Device | Ch | N | Tag |
|---|---|---|---|---|---|
| A-01 | LIMU-BERT-X | phone | 6ch | 1 | `exp52_limubertx_phone_6ch_wisdm_leavein1` |
| A-02 | LIMU-BERT-X | phone | 6ch | 5 | `exp52_limubertx_phone_6ch_wisdm_leavein5` |
| A-03 | LIMU-BERT-X | phone | 6ch | 25 | `exp52_limubertx_phone_6ch_wisdm_leavein25` |
| A-04 | LIMU-BERT-X | phone | 6ch | 40 | `exp52_limubertx_phone_6ch_wisdm_leavein40` |
| A-05 | LIMU-BERT-X | watch | 6ch | 1 | `exp52_limubertx_watch_6ch_wisdm_leavein1` |
| A-06 | LIMU-BERT-X | watch | 6ch | 5 | `exp52_limubertx_watch_6ch_wisdm_leavein5` |
| A-07 | LIMU-BERT-X | watch | 6ch | 25 | `exp52_limubertx_watch_6ch_wisdm_leavein25` |
| A-08 | LIMU-BERT-X | watch | 6ch | 40 | `exp52_limubertx_watch_6ch_wisdm_leavein40` |
| A-09 | LIMU-BERT-X | all | 6ch | 1 | `exp52_limubertx_all_6ch_wisdm_leavein1` |
| A-10 | LIMU-BERT-X | all | 6ch | 5 | `exp52_limubertx_all_6ch_wisdm_leavein5` |
| A-11 | LIMU-BERT-X | all | 6ch | 25 | `exp52_limubertx_all_6ch_wisdm_leavein25` |
| A-12 | LIMU-BERT-X | all | 6ch | 40 | `exp52_limubertx_all_6ch_wisdm_leavein40` |
| A-13 | SSL-Wearables | phone | 3ch | 1 | `exp52_ssl_phone_3ch_wisdm_leavein1` |
| A-14 | SSL-Wearables | phone | 3ch | 5 | `exp52_ssl_phone_3ch_wisdm_leavein5` |
| A-15 | SSL-Wearables | phone | 3ch | 25 | `exp52_ssl_phone_3ch_wisdm_leavein25` |
| A-16 | SSL-Wearables | phone | 3ch | 40 | `exp52_ssl_phone_3ch_wisdm_leavein40` |
| A-17 | SSL-Wearables | watch | 3ch | 1 | `exp52_ssl_watch_3ch_wisdm_leavein1` |
| A-18 | SSL-Wearables | watch | 3ch | 5 | `exp52_ssl_watch_3ch_wisdm_leavein5` |
| A-19 | SSL-Wearables | watch | 3ch | 25 | `exp52_ssl_watch_3ch_wisdm_leavein25` |
| A-20 | SSL-Wearables | watch | 3ch | 40 | `exp52_ssl_watch_3ch_wisdm_leavein40` |
| A-21 | SSL-Wearables | all | 3ch | 1 | `exp52_ssl_all_3ch_wisdm_leavein1` |
| A-22 | SSL-Wearables | all | 3ch | 5 | `exp52_ssl_all_3ch_wisdm_leavein5` |
| A-23 | SSL-Wearables | all | 3ch | 25 | `exp52_ssl_all_3ch_wisdm_leavein25` |
| A-24 | SSL-Wearables | all | 3ch | 40 | `exp52_ssl_all_3ch_wisdm_leavein40` |

```powershell
# ===== LIMU-BERT-X 6ch =====
# The results from 0 finetuning can be found from the original exp 5 and exp 6
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_limubertx_phone_6ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_limubertx_phone_6ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_limubertx_phone_6ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_limubertx_phone_6ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_limubertx_watch_6ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_limubertx_watch_6ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_limubertx_watch_6ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_limubertx_watch_6ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_limubertx_all_6ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_limubertx_all_6ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_limubertx_all_6ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_limubertx_all_6ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
# ===== SSL-Wearables 3ch =====
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_ssl_phone_3ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_ssl_phone_3ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_ssl_phone_3ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_ssl_phone_3ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_ssl_watch_3ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_ssl_watch_3ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_ssl_watch_3ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_ssl_watch_3ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --finetune-subjects 1 --experiment-tag exp52_ssl_all_3ch_wisdm_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --finetune-subjects 5 --experiment-tag exp52_ssl_all_3ch_wisdm_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --finetune-subjects 25 --experiment-tag exp52_ssl_all_3ch_wisdm_leavein25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --finetune-subjects 40 --experiment-tag exp52_ssl_all_3ch_wisdm_leavein40 --seed 42 --repeat-seeds 3
```

#### 5.2-B: Leave-N-in — PAMAP2+SBHAR, 5-class `--with-stairs` (12 commands)

| ID | Model | Device | Ch | N | Tag |
|---|---|---|---|---|---|
| B-01 | LIMU-BERT-X | phone | 6ch | 1 | `exp52_limubertx_phone_6ch_p2sb_leavein1` |
| B-02 | LIMU-BERT-X | phone | 6ch | 5 | `exp52_limubertx_phone_6ch_p2sb_leavein5` |
| B-03 | LIMU-BERT-X | watch | 6ch | 1 | `exp52_limubertx_watch_6ch_p2sb_leavein1` |
| B-04 | LIMU-BERT-X | watch | 6ch | 5 | `exp52_limubertx_watch_6ch_p2sb_leavein5` |
| B-05 | LIMU-BERT-X | all | 6ch | 1 | `exp52_limubertx_all_6ch_p2sb_leavein1` |
| B-06 | LIMU-BERT-X | all | 6ch | 5 | `exp52_limubertx_all_6ch_p2sb_leavein5` |
| B-07 | SSL-Wearables | phone | 3ch | 1 | `exp52_ssl_phone_3ch_p2sb_leavein1` |
| B-08 | SSL-Wearables | phone | 3ch | 5 | `exp52_ssl_phone_3ch_p2sb_leavein5` |
| B-09 | SSL-Wearables | watch | 3ch | 1 | `exp52_ssl_watch_3ch_p2sb_leavein1` |
| B-10 | SSL-Wearables | watch | 3ch | 5 | `exp52_ssl_watch_3ch_p2sb_leavein5` |
| B-11 | SSL-Wearables | all | 3ch | 1 | `exp52_ssl_all_3ch_p2sb_leavein1` |
| B-12 | SSL-Wearables | all | 3ch | 5 | `exp52_ssl_all_3ch_p2sb_leavein5` |

```powershell
# Prerequisites: python prepare_pamap2_sbhar_data.py --aligned
# ===== LIMU-BERT-X 6ch =====
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_limubertx_phone_6ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_limubertx_phone_6ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_limubertx_watch_6ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_limubertx_watch_6ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_limubertx_all_6ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_limubertx_all_6ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
# ===== SSL-Wearables 3ch =====
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_ssl_phone_3ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_ssl_phone_3ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_ssl_watch_3ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_ssl_watch_3ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 1 --experiment-tag exp52_ssl_all_3ch_p2sb_leavein1 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset pamap2_sbhar --with-stairs --finetune-subjects 5 --experiment-tag exp52_ssl_all_3ch_p2sb_leavein5 --seed 42 --repeat-seeds 3
```

#### 5.2-C: Data-fraction — WISDM, 3-class (18 commands)

| ID | Model | Device | Ch | Frac | Tag |
|---|---|---|---|---|---|
| C-01 | LIMU-BERT-X | phone | 6ch | 10% | `exp52_limubertx_phone_6ch_wisdm_frac10` |
| C-02 | LIMU-BERT-X | phone | 6ch | 25% | `exp52_limubertx_phone_6ch_wisdm_frac25` |
| C-03 | LIMU-BERT-X | phone | 6ch | 75% | `exp52_limubertx_phone_6ch_wisdm_frac75` |
| C-04 | LIMU-BERT-X | watch | 6ch | 10% | `exp52_limubertx_watch_6ch_wisdm_frac10` |
| C-05 | LIMU-BERT-X | watch | 6ch | 25% | `exp52_limubertx_watch_6ch_wisdm_frac25` |
| C-06 | LIMU-BERT-X | watch | 6ch | 75% | `exp52_limubertx_watch_6ch_wisdm_frac75` |
| C-07 | LIMU-BERT-X | all | 6ch | 10% | `exp52_limubertx_all_6ch_wisdm_frac10` |
| C-08 | LIMU-BERT-X | all | 6ch | 25% | `exp52_limubertx_all_6ch_wisdm_frac25` |
| C-09 | LIMU-BERT-X | all | 6ch | 75% | `exp52_limubertx_all_6ch_wisdm_frac75` |
| C-10 | SSL-Wearables | phone | 3ch | 10% | `exp52_ssl_phone_3ch_wisdm_frac10` |
| C-11 | SSL-Wearables | phone | 3ch | 25% | `exp52_ssl_phone_3ch_wisdm_frac25` |
| C-12 | SSL-Wearables | phone | 3ch | 75% | `exp52_ssl_phone_3ch_wisdm_frac75` |
| C-13 | SSL-Wearables | watch | 3ch | 10% | `exp52_ssl_watch_3ch_wisdm_frac10` |
| C-14 | SSL-Wearables | watch | 3ch | 25% | `exp52_ssl_watch_3ch_wisdm_frac25` |
| C-15 | SSL-Wearables | watch | 3ch | 75% | `exp52_ssl_watch_3ch_wisdm_frac75` |
| C-16 | SSL-Wearables | all | 3ch | 10% | `exp52_ssl_all_3ch_wisdm_frac10` |
| C-17 | SSL-Wearables | all | 3ch | 25% | `exp52_ssl_all_3ch_wisdm_frac25` |
| C-18 | SSL-Wearables | all | 3ch | 75% | `exp52_ssl_all_3ch_wisdm_frac75` |

```powershell
# ===== LIMU-BERT-X 6ch =====
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_phone_6ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_phone_6ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_phone_6ch_wisdm_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_watch_6ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_watch_6ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_watch_6ch_wisdm_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_all_6ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_all_6ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_all_6ch_wisdm_frac75 --seed 42 --repeat-seeds 3
# ===== SSL-Wearables 3ch =====
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_ssl_phone_3ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_ssl_phone_3ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_ssl_phone_3ch_wisdm_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_ssl_watch_3ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_ssl_watch_3ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_ssl_watch_3ch_wisdm_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --per-subject-fraction 0.10 --experiment-tag exp52_ssl_all_3ch_wisdm_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --per-subject-fraction 0.25 --experiment-tag exp52_ssl_all_3ch_wisdm_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset wisdm --per-subject-fraction 0.75 --experiment-tag exp52_ssl_all_3ch_wisdm_frac75 --seed 42 --repeat-seeds 3
```

#### 5.2-D: Data-fraction — PAMAP2+SBHAR, 5-class `--with-stairs` (18 commands)

| ID | Model | Device | Ch | Frac | Tag |
|---|---|---|---|---|---|
| D-01 | LIMU-BERT-X | phone | 6ch | 10% | `exp52_limubertx_phone_6ch_p2sb_frac10` |
| D-02 | LIMU-BERT-X | phone | 6ch | 25% | `exp52_limubertx_phone_6ch_p2sb_frac25` |
| D-03 | LIMU-BERT-X | phone | 6ch | 75% | `exp52_limubertx_phone_6ch_p2sb_frac75` |
| D-04 | LIMU-BERT-X | watch | 6ch | 10% | `exp52_limubertx_watch_6ch_p2sb_frac10` |
| D-05 | LIMU-BERT-X | watch | 6ch | 25% | `exp52_limubertx_watch_6ch_p2sb_frac25` |
| D-06 | LIMU-BERT-X | watch | 6ch | 75% | `exp52_limubertx_watch_6ch_p2sb_frac75` |
| D-07 | LIMU-BERT-X | all | 6ch | 10% | `exp52_limubertx_all_6ch_p2sb_frac10` |
| D-08 | LIMU-BERT-X | all | 6ch | 25% | `exp52_limubertx_all_6ch_p2sb_frac25` |
| D-09 | LIMU-BERT-X | all | 6ch | 75% | `exp52_limubertx_all_6ch_p2sb_frac75` |
| D-10 | SSL-Wearables | phone | 3ch | 10% | `exp52_ssl_phone_3ch_p2sb_frac10` |
| D-11 | SSL-Wearables | phone | 3ch | 25% | `exp52_ssl_phone_3ch_p2sb_frac25` |
| D-12 | SSL-Wearables | phone | 3ch | 75% | `exp52_ssl_phone_3ch_p2sb_frac75` |
| D-13 | SSL-Wearables | watch | 3ch | 10% | `exp52_ssl_watch_3ch_p2sb_frac10` |
| D-14 | SSL-Wearables | watch | 3ch | 25% | `exp52_ssl_watch_3ch_p2sb_frac25` |
| D-15 | SSL-Wearables | watch | 3ch | 75% | `exp52_ssl_watch_3ch_p2sb_frac75` |
| D-16 | SSL-Wearables | all | 3ch | 10% | `exp52_ssl_all_3ch_p2sb_frac10` |
| D-17 | SSL-Wearables | all | 3ch | 25% | `exp52_ssl_all_3ch_p2sb_frac25` |
| D-18 | SSL-Wearables | all | 3ch | 75% | `exp52_ssl_all_3ch_p2sb_frac75` |

```powershell
# Prerequisites: python prepare_pamap2_sbhar_data.py --aligned
# ===== LIMU-BERT-X 6ch =====
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_phone_6ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_phone_6ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_phone_6ch_p2sb_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_watch_6ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_watch_6ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_watch_6ch_p2sb_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_limubertx_all_6ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_limubertx_all_6ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model limu-bert --limu-seq-len 20 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_limubertx_all_6ch_p2sb_frac75 --seed 42 --repeat-seeds 3
# ===== SSL-Wearables 3ch =====
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_ssl_phone_3ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_ssl_phone_3ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type phone --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_ssl_phone_3ch_p2sb_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_ssl_watch_3ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_ssl_watch_3ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type watch --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_ssl_watch_3ch_p2sb_frac75 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.10 --experiment-tag exp52_ssl_all_3ch_p2sb_frac10 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.25 --experiment-tag exp52_ssl_all_3ch_p2sb_frac25 --seed 42 --repeat-seeds 3
python eval_cross_finetune.py --model ssl-wearables --channels 3 --device-type all --target-dataset pamap2_sbhar --with-stairs --per-subject-fraction 0.75 --experiment-tag exp52_ssl_all_3ch_p2sb_frac75 --seed 42 --repeat-seeds 3
```

> Notes:
> - **Two protocols are mutually exclusive:** `--finetune-subjects` (leave-in) vs `--per-subject-fraction` (data-fraction).
> - **`--with-stairs`** used for all PAMAP2+SBHAR commands (5-class). WISDM stays 3-class (no stairs in WISDM).
> - **Stage 1 cached per (model, device, num_classes):** e.g. `stage1_limubertx_phone_3cls.pt` and `stage1_limubertx_phone_5cls.pt` are separate caches. Within the same class count, leave-in and data-fraction share Stage 1.
> - `--repeat-seeds 3` runs seeds 42, 43, 44 — each seed gives different Stage 2 splits. Stage 1 always uses seed 42.
> - Stage 1 (HHAR) default: 200 epochs, patience 15. Stage 2 (fine-tune) default: 100 epochs, patience 10.
> - Override with `--epochs` (stage 1) and `--ft-epochs` (stage 2). Use `--force-stage1` to retrain Stage 1.
> - LIMU-BERT-X always uses 6ch. SSL-Wearables uses 3ch only.
> - Target `pamap2_sbhar` = aligned 8+8. Use `pamap2_sbhar_full` for 8+30.
> - Compare with Exp 5.1 (zero-shot cross) to quantify fine-tuning gain.

---

**Repeated-run mean/std**:

```powershell
# Example: 3 repeated seeds for LIMU-BERT-X all
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all --reverse --experiment-tag exp6_limubertx_all --seed 42 --repeat-seeds 3
```
### Experiment 7 -- Leave-One-In Generalization (K-Subject Fine-Tuning)

> **Goal:** Run a complete leave-one-subject-in (LOSI) matrix on HHAR, covering the requested architecture/device/pretraining settings with train-subject counts `K in {1, 5, 8}`.
> 
> This is a variation of Experiment 1, but instead of using a percentage of the data (`--data-fraction`), we use a **leave-one-in** protocol:
> train on selected subjects (`--train-subjects K --leave-one-in`) and report test accuracy on each remaining subject individually.
>
> **Subject set note:** HHAR uses 9 subjects here, consistent with Experiment 1 settings.
>
> **LOSI note:** All commands below are LOSI (`--leave-one-in`).

**Commands:**

```powershell
# --- Part A: LIMU-BERT-X (Transformer, 6ch) ---
# A1) pretrained=Yes, phone, K=1/5/8
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_8subj --seed 42 --repeat-seeds 3

# A2) pretrained=No, phone/watch/all, K=1/5/8
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_8subj --seed 42 --repeat-seeds 3

# --- Part B: ssl-wearables (CNN, 3ch) ---
# B1) pretrained=Yes, watch, K=1/5/8
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_8subj --seed 42 --repeat-seeds 3

# B2) pretrained=No, phone/watch/all, K=1/5/8
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_8subj --seed 42 --repeat-seeds 3

# --- Part C: HART (Transformer, 6ch), pretrained=No, K=1/5/8 ---
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_8subj --seed 42 --repeat-seeds 3

# --- Part D: ResNet-Baseline (CNN, 3ch), pretrained=No, K=1/5/8 ---
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_8subj --seed 42 --repeat-seeds 3
```

> `--leave-one-in` output format: for each train-subject fold, the script prints `train=<id> -> test=<id>: Acc=...` for every remaining subject instead of only reporting a single fold mean.
>
> **Combination count by K (from 9 subjects):**
> - `K=1`: `C(9,1)=9` train combinations, `8` held-out subjects each fold
> - `K=5`: `C(9,5)=126` train combinations, `4` held-out subjects each fold
> - `K=8`: `C(9,8)=9` train combinations, `1` held-out subject each fold


### Experiment 8 -- Custom-Collected Data Evaluation

> **Goal:** Evaluate all model architectures on our own custom-collected Xsens IMU data.
>
> **Two modes:**
> - **Mode A (cross):** Use HHAR-trained Exp0 scratch models directly on custom data (no retraining). Tests whether HHAR generalization holds on completely new hardware and subjects.
> - **Mode B (LOSO):** Leave-one-subject-out within the custom dataset only. Trains from scratch on 4 subjects, tests on 1.
>
> **Custom data characteristics:**
> - **Source:** `data_combined/` (TXT files only; ignore MTB files)
> - **5 subjects:** B, C, D, O, Y
> - **4 Xsens sensors per recording:** right wrist, left wrist (watch), right pocket, left pocket (phone)
> - **5 activities:** sit (0), stand (1), walk (2), stairUp (3), stairDown (4)
> - **2 conditions:** control (controlled posture), uncontrolled (natural posture)
> - **6 channels available:** `Acc_X/Y/Z + Roll/Pitch/Yaw`, ~100 Hz, ~2 min per recording.
> - GT is parsed from filename: subject / activity / condition / duration / device id.

**Cross-dataset (Mode A):**

| ID | Model | Mode | Device | Condition | Tag |
|---|---|---|---|---|---|
| **8A** | HART | cross | all | all | `exp8a_hart_cross_all` |
| **8C** | LIMU-BERT-X | cross | all | all | `exp8c_limuX_cross_all` |
| **8E** | ssl-wearables | cross | all | all | `exp8e_ssl_cross_all` |
| **8G** | ResNet-Baseline | cross | all | all | `exp8g_resbase_cross_all` |
| **8I** | HART | cross | watch | control | `exp8i_hart_cross_watch_ctrl` |
| **8J** | HART | cross | phone | control | `exp8j_hart_cross_phone_ctrl` |
| **8K** | HART | cross | watch | uncontrolled | `exp8k_hart_cross_watch_unc` |
| **8L** | HART | cross | phone | uncontrolled | `exp8l_hart_cross_phone_unc` |

**LOSO (Mode B) -- all condition × device combinations (9 combos × 4 models = 36 runs):**

| Model | Device | Condition | Tag |
|---|---|---|---|
| HART | all | all | `exp8b_hart_loso_all_all` |
| HART | phone | all | `exp8b_hart_loso_all_phone` |
| HART | watch | all | `exp8b_hart_loso_all_watch` |
| HART | all | control | `exp8b_hart_loso_ctrl_all` |
| HART | phone | control | `exp8b_hart_loso_ctrl_phone` |
| HART | watch | control | `exp8b_hart_loso_ctrl_watch` |
| HART | all | uncontrolled | `exp8b_hart_loso_unc_all` |
| HART | phone | uncontrolled | `exp8b_hart_loso_unc_phone` |
| HART | watch | uncontrolled | `exp8b_hart_loso_unc_watch` |
| LIMU-BERT-X | all | all | `exp8d_limuX_loso_all_all` |
| LIMU-BERT-X | phone | all | `exp8d_limuX_loso_all_phone` |
| LIMU-BERT-X | watch | all | `exp8d_limuX_loso_all_watch` |
| LIMU-BERT-X | all | control | `exp8d_limuX_loso_ctrl_all` |
| LIMU-BERT-X | phone | control | `exp8d_limuX_loso_ctrl_phone` |
| LIMU-BERT-X | watch | control | `exp8d_limuX_loso_ctrl_watch` |
| LIMU-BERT-X | all | uncontrolled | `exp8d_limuX_loso_unc_all` |
| LIMU-BERT-X | phone | uncontrolled | `exp8d_limuX_loso_unc_phone` |
| LIMU-BERT-X | watch | uncontrolled | `exp8d_limuX_loso_unc_watch` |
| ssl-wearables | all | all | `exp8f_ssl_loso_all_all` |
| ssl-wearables | phone | all | `exp8f_ssl_loso_all_phone` |
| ssl-wearables | watch | all | `exp8f_ssl_loso_all_watch` |
| ssl-wearables | all | control | `exp8f_ssl_loso_ctrl_all` |
| ssl-wearables | phone | control | `exp8f_ssl_loso_ctrl_phone` |
| ssl-wearables | watch | control | `exp8f_ssl_loso_ctrl_watch` |
| ssl-wearables | all | uncontrolled | `exp8f_ssl_loso_unc_all` |
| ssl-wearables | phone | uncontrolled | `exp8f_ssl_loso_unc_phone` |
| ssl-wearables | watch | uncontrolled | `exp8f_ssl_loso_unc_watch` |
| ResNet-Baseline | all | all | `exp8h_resbase_loso_all_all` |
| ResNet-Baseline | phone | all | `exp8h_resbase_loso_all_phone` |
| ResNet-Baseline | watch | all | `exp8h_resbase_loso_all_watch` |
| ResNet-Baseline | all | control | `exp8h_resbase_loso_ctrl_all` |
| ResNet-Baseline | phone | control | `exp8h_resbase_loso_ctrl_phone` |
| ResNet-Baseline | watch | control | `exp8h_resbase_loso_ctrl_watch` |
| ResNet-Baseline | all | uncontrolled | `exp8h_resbase_loso_unc_all` |
| ResNet-Baseline | phone | uncontrolled | `exp8h_resbase_loso_unc_phone` |
| ResNet-Baseline | watch | uncontrolled | `exp8h_resbase_loso_unc_watch` |

**Commands:**

```powershell
# Step 1: Prepare custom data (all conditions, all devices)
python prepare_custom_data.py --force

# Step 2A: Cross-dataset evaluation (HHAR-trained models -> custom data)
# Upstream run IDs mapped to Experiment 8 matrix (Run 47-88).
# Use --custom-test-repeats 3 for three custom-data tests per run.

# --- Our Controlled (condition=control) ---
# Run 47: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run47_limuX_phone_ctrl_preY
# Run 48: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run48_limuX_phone_ctrl_preN
# Run 49
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run49_limuX_watch_ctrl
# Run 50
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run50_limuX_all_ctrl
# Run 51
python eval_custom_data.py --model hart --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run51_hart_phone_ctrl
# Run 52
python eval_custom_data.py --model hart --mode cross --device-type watch --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run52_hart_watch_ctrl
# Run 53
python eval_custom_data.py --model hart --mode cross --device-type all --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run53_hart_all_ctrl
# Run 54: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run54_ssl_watch_ctrl_preY
# Run 55
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run55_ssl_phone_ctrl
# Run 56: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run56_ssl_watch_ctrl_preN
# Run 57
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run57_ssl_all_ctrl
# Run 58
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run58_resbase_phone_ctrl_3ch
# Run 59
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run59_resbase_watch_ctrl_3ch
# Run 60
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run60_resbase_all_ctrl_3ch

# --- Our Uncontrolled (condition=uncontrolled) ---
# Run 61: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run61_limuX_phone_unc_preY
# Run 62: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run62_limuX_phone_unc_preN
# Run 63
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run63_limuX_watch_unc
# Run 64
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run64_limuX_all_unc
# Run 65
python eval_custom_data.py --model hart --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run65_hart_phone_unc
# Run 66
python eval_custom_data.py --model hart --mode cross --device-type watch --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run66_hart_watch_unc
# Run 67
python eval_custom_data.py --model hart --mode cross --device-type all --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run67_hart_all_unc
# Run 68: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run68_ssl_watch_unc_preY
# Run 69
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run69_ssl_phone_unc
# Run 70: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run70_ssl_watch_unc_preN
# Run 71
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run71_ssl_all_unc
# Run 72
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run72_resbase_phone_unc_3ch
# Run 73
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run73_resbase_watch_unc_3ch
# Run 74
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run74_resbase_all_unc_3ch

# --- Our All (condition=all) ---
# Run 75: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run75_limuX_phone_all_preY
# Run 76: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run76_limuX_phone_all_preN
# Run 77
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run77_limuX_watch_all
# Run 78
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run78_limuX_all_all
# Run 79
python eval_custom_data.py --model hart --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run79_hart_phone_all
# Run 80
python eval_custom_data.py --model hart --mode cross --device-type watch --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run80_hart_watch_all
# Run 81
python eval_custom_data.py --model hart --mode cross --device-type all --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run81_hart_all_all
# Run 82: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run82_ssl_watch_all_preY
# Run 83
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run83_ssl_phone_all
# Run 84: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run84_ssl_watch_all_preN
# Run 85
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run85_ssl_all_all
# Run 86
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run86_resbase_phone_all_3ch
# Run 87
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run87_resbase_watch_all_3ch
# Run 88
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run88_resbase_all_all_3ch

# Note:
# - eval_custom_data.py does not expose --repeat-seeds; use --custom-test-repeats 3 here.
# - "Imported Pretrained Weights = Yes" rows explicitly pass --pretrained in commands.

# Step 2B: LOSO within custom data (all condition × device combinations)
# --- HART (all 9 combos) ---
# Run 89
python eval_custom_data.py --model hart --mode loso --device-type all --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_all_run89 --seed 42
# Run 90
python eval_custom_data.py --model hart --mode loso --device-type phone --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_phone_run90 --seed 42
# Run 91
python eval_custom_data.py --model hart --mode loso --device-type watch --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_watch_run91 --seed 42
# Run 92
python eval_custom_data.py --model hart --mode loso --device-type all --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_all_run92 --seed 42
# Run 93
python eval_custom_data.py --model hart --mode loso --device-type phone --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_phone_run93 --seed 42
# Run 94
python eval_custom_data.py --model hart --mode loso --device-type watch --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_watch_run94 --seed 42
# Run 95
python eval_custom_data.py --model hart --mode loso --device-type all --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_all_run95 --seed 42
# Run 96
python eval_custom_data.py --model hart --mode loso --device-type phone --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_phone_run96 --seed 42
# Run 97
python eval_custom_data.py --model hart --mode loso --device-type watch --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_watch_run97 --seed 42

# --- LIMU-BERT-X (all 9 combos) ---
# Run 98
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_all_run98 --seed 42
# Run 99
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_phone_run99 --seed 42
# Run 100
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_watch_run100 --seed 42
# Run 101
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_all_run101 --seed 42
# Run 102
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_phone_run102 --seed 42
# Run 103
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_watch_run103 --seed 42
# Run 104
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_all_run104 --seed 42
# Run 105
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_phone_run105 --seed 42
# Run 106
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_watch_run106 --seed 42

# --- ssl-wearables 6ch (all 9 combos) ---
# Run 107
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_all_run107 --seed 42
# Run 108
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_phone_run108 --seed 42
# Run 109
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_watch_run109 --seed 42
# Run 110
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_all_run110 --seed 42
# Run 111
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_phone_run111 --seed 42
# Run 112
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_watch_run112 --seed 42
# Run 113
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_all_run113 --seed 42
# Run 114
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_phone_run114 --seed 42
# Run 115
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_watch_run115 --seed 42

# --- ResNet-Baseline 6ch (all 9 combos) ---
# Run 116
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_all_run116 --seed 42
# Run 117
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_phone_run117 --seed 42
# Run 118
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_watch_run118 --seed 42
# Run 119
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_all_run119 --seed 42
# Run 120
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_phone_run120 --seed 42
# Run 121
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_watch_run121 --seed 42
# Run 122
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_all_run122 --seed 42
# Run 123
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_phone_run123 --seed 42
# Run 124
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_watch_run124 --seed 42
```

**Reported metrics:**

- Mode A (cross): accuracy, F1 (weighted/macro), per-subject, per-device, per-class metrics, confusion matrix
- Mode A std: available when `--custom-test-repeats > 1` (stored in repeat summary).
- Mode B (LOSO): per-fold accuracy/F1 + mean/std (accuracy + F1), per-device breakdown

**Notes:**

- Custom data is parsed from `data_combined` **TXT** files only (MTB files are ignored).
- Custom data is **6ch (`Acc_X/Y/Z + Roll/Pitch/Yaw`)**.
- If a cross-dataset checkpoint was trained in 3ch only, run with `--channels 3` for that model to avoid input-shape mismatch.
- The `--condition` flag enables separate evaluation of `control` vs `uncontrolled` recordings.
- The `--device-type` flag supports `watch` (wrist sensors), `phone` (pocket sensors), or `all`.
- Mode A loads the best fold-0 checkpoint from Exp0 LOSO training. Make sure Exp0 has been run first.
- LOSO in Mode B uses only 5 subjects, so variance may still be high.

---

### Experiment 8.5 -- Custom Data LOSO Matrix (Table-Aligned)

> **Goal:** Run a second LOSO matrix on custom data with tags aligned to your table schema.
>
> This section is designed to be summarized with:
> `python summarize_experiment_results.py --exp exp85`
>
> All runs below are LOSO on custom data (`mode=loso`) and write `experiment_table_metadata`
> so output CSV columns match your requested template.

| Architecture Type | Device Type | Testing Dataset | Imported Pretrained Weights | 3ch vs 6ch data | Data Fraction | Additional Pretraining dataset | Training Dataset | Testing Method | # Run |
|---|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | Our Controlled | No | 6ch | 100% | None | None | LOSO | 124 |
| LIMU-BERT-X | Watch | Our Controlled | No | 6ch | 100% | None | None | LOSO | 125 |
| LIMU-BERT-X | All | Our Controlled | No | 6ch | 100% | None | None | LOSO | 126 |
| LIMU-BERT-X | Phone | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 127 |
| LIMU-BERT-X | Watch | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 128 |
| LIMU-BERT-X | All | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 129 |
| LIMU-BERT-X | Phone | Our All | No | 6ch | 100% | None | None | LOSO | 130 |
| LIMU-BERT-X | Watch | Our All | No | 6ch | 100% | None | None | LOSO | 131 |
| LIMU-BERT-X | All | Our All | No | 6ch | 100% | None | None | LOSO | 132 |
| LIMU-BERT-X | Phone | Our Controlled | Yes | 6ch | 100% | None | None | LOSO | 133 |
| LIMU-BERT-X | Phone | Our Uncontrolled | Yes | 6ch | 100% | None | None | LOSO | 134 |
| LIMU-BERT-X | Phone | Our All | Yes | 6ch | 100% | None | None | LOSO | 135 |
| HART | Phone | Our Controlled | No | 6ch | 100% | None | None | LOSO | 136 |
| HART | Watch | Our Controlled | No | 6ch | 100% | None | None | LOSO | 137 |
| HART | All | Our Controlled | No | 6ch | 100% | None | None | LOSO | 138 |
| HART | Phone | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 139 |
| HART | Watch | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 140 |
| HART | All | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 141 |
| HART | Phone | Our All | No | 6ch | 100% | None | None | LOSO | 142 |
| HART | Watch | Our All | No | 6ch | 100% | None | None | LOSO | 143 |
| HART | All | Our All | No | 6ch | 100% | None | None | LOSO | 144 |
| SSL-Wearables | Phone | Our Controlled | No | 3ch | 100% | None | None | LOSO | 145 |
| SSL-Wearables | Watch | Our Controlled | No | 3ch | 100% | None | None | LOSO | 146 |
| SSL-Wearables | All | Our Controlled | No | 3ch | 100% | None | None | LOSO | 147 |
| SSL-Wearables | Phone | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 148 |
| SSL-Wearables | Watch | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 149 |
| SSL-Wearables | All | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 150 |
| SSL-Wearables | Phone | Our All | No | 3ch | 100% | None | None | LOSO | 151 |
| SSL-Wearables | Watch | Our All | No | 3ch | 100% | None | None | LOSO | 152 |
| SSL-Wearables | All | Our All | No | 3ch | 100% | None | None | LOSO | 153 |
| SSL-Wearables | Watch | Our Controlled | Yes | 3ch | 100% | None | None | LOSO | 154 |
| SSL-Wearables | Watch | Our Uncontrolled | Yes | 3ch | 100% | None | None | LOSO | 155 |
| SSL-Wearables | Watch | Our All | Yes | 3ch | 100% | None | None | LOSO | 156 |
| ResNet-Baseline | Phone | Our Controlled | No | 3ch | 100% | None | None | LOSO | 157 |
| ResNet-Baseline | Watch | Our Controlled | No | 3ch | 100% | None | None | LOSO | 158 |
| ResNet-Baseline | All | Our Controlled | No | 3ch | 100% | None | None | LOSO | 159 |
| ResNet-Baseline | Phone | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 160 |
| ResNet-Baseline | Watch | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 161 |
| ResNet-Baseline | All | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 162 |
| ResNet-Baseline | Phone | Our All | No | 3ch | 100% | None | None | LOSO | 163 |
| ResNet-Baseline | Watch | Our All | No | 3ch | 100% | None | None | LOSO | 164 |
| ResNet-Baseline | All | Our All | No | 3ch | 100% | None | None | LOSO | 165 |

```powershell
# --- LIMU-BERT-X (No pretrained): Run 124-132 ---
# Run 124
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp85_run124_limuX_phone_ctrl_preN --seed 42
# Run 125
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp85_run125_limuX_watch_ctrl_preN --seed 42
# Run 126
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition control --channels 6 --experiment-tag exp85_run126_limuX_all_ctrl_preN --seed 42
# Run 127
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp85_run127_limuX_phone_unc_preN --seed 42
# Run 128
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp85_run128_limuX_watch_unc_preN --seed 42
# Run 129
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp85_run129_limuX_all_unc_preN --seed 42
# Run 130
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp85_run130_limuX_phone_all_preN --seed 42
# Run 131
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp85_run131_limuX_watch_all_preN --seed 42
# Run 132
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition all --channels 6 --experiment-tag exp85_run132_limuX_all_all_preN --seed 42

# --- LIMU-BERT-X (Imported pretrained=Yes): Run 133-135 ---
# Run 133
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp85_run133_limuX_phone_ctrl_preY --seed 42
# Run 134
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp85_run134_limuX_phone_unc_preY --seed 42
# Run 135
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp85_run135_limuX_phone_all_preY --seed 42

# --- HART: Run 136-144 ---
# Run 136
python eval_custom_data.py --model hart --mode loso --device-type phone --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run136_hart_phone_ctrl --seed 42
# Run 137
python eval_custom_data.py --model hart --mode loso --device-type watch --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run137_hart_watch_ctrl --seed 42
# Run 138
python eval_custom_data.py --model hart --mode loso --device-type all --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run138_hart_all_ctrl --seed 42
# Run 139
python eval_custom_data.py --model hart --mode loso --device-type phone --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run139_hart_phone_unc --seed 42
# Run 140
python eval_custom_data.py --model hart --mode loso --device-type watch --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run140_hart_watch_unc --seed 42
# Run 141
python eval_custom_data.py --model hart --mode loso --device-type all --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run141_hart_all_unc --seed 42
# Run 142
python eval_custom_data.py --model hart --mode loso --device-type phone --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run142_hart_phone_all --seed 42
# Run 143
python eval_custom_data.py --model hart --mode loso --device-type watch --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run143_hart_watch_all --seed 42
# Run 144
python eval_custom_data.py --model hart --mode loso --device-type all --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run144_hart_all_all --seed 42

# --- SSL-Wearables (No pretrained): Run 145-153 ---
# Run 145
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition control --channels 3 --experiment-tag exp85_run145_ssl_phone_ctrl_preN --seed 42
# Run 146
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run146_ssl_watch_ctrl_preN --seed 42
# Run 147
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition control --channels 3 --experiment-tag exp85_run147_ssl_all_ctrl_preN --seed 42
# Run 148
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp85_run148_ssl_phone_unc_preN --seed 42
# Run 149
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run149_ssl_watch_unc_preN --seed 42
# Run 150
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition uncontrolled --channels 3 --experiment-tag exp85_run150_ssl_all_unc_preN --seed 42
# Run 151
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition all --channels 3 --experiment-tag exp85_run151_ssl_phone_all_preN --seed 42
# Run 152
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run152_ssl_watch_all_preN --seed 42
# Run 153
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition all --channels 3 --experiment-tag exp85_run153_ssl_all_all_preN --seed 42

# --- SSL-Wearables (Imported pretrained=Yes): Run 154-156 ---
# Run 154
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run154_ssl_watch_ctrl_preY --seed 42
# Run 155
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run155_ssl_watch_unc_preY --seed 42
# Run 156
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run156_ssl_watch_all_preY --seed 42

# --- ResNet-Baseline: Run 157-165 ---
# Run 157
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition control --channels 3 --experiment-tag exp85_run157_resbase_phone_ctrl_3ch --seed 42
# Run 158
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run158_resbase_watch_ctrl_3ch --seed 42
# Run 159
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition control --channels 3 --experiment-tag exp85_run159_resbase_all_ctrl_3ch --seed 42
# Run 160
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp85_run160_resbase_phone_unc_3ch --seed 42
# Run 161
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run161_resbase_watch_unc_3ch --seed 42
# Run 162
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition uncontrolled --channels 3 --experiment-tag exp85_run162_resbase_all_unc_3ch --seed 42
# Run 163
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition all --channels 3 --experiment-tag exp85_run163_resbase_phone_all_3ch --seed 42
# Run 164
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run164_resbase_watch_all_3ch --seed 42
# Run 165
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition all --channels 3 --experiment-tag exp85_run165_resbase_all_all_3ch --seed 42
```

> Notes:
> - Exp8.5 tags use `exp85_runXXX_*`, so summary filtering is direct with `--exp exp85`.
> - For `Imported Pretrained Weights = Yes`, commands explicitly pass `--pretrained`.
> - JSON output includes `experiment_table_metadata`, so the summarizer keeps your table columns stable.

---
### Additional Training / Priority Queue (Exp5 & Exp6 first)

> Do **3 trials minimum per configuration** (`--seed 42 --repeat-seeds 3`).
> Cross-dataset outputs include **Accuracy + F1 weighted + F1 macro** (saved in JSON/CSV).
> For repeated runs, report **mean ± std** for accuracy/F1.
> Variables kept with the same names as requested:
> `Architecture Type`, `Device Type`, `Training Dataset`, `Testing Dataset`,
> `3ch vs 6ch data`, `Imported Pretrained Weights`, `Additional Pretraining dataset`,
> `Data Fraction`, `Testing Method`.

| Architecture Type | Device Type | Training Dataset | Testing Dataset | 3ch vs 6ch data | Imported Pretrained Weights | Additional Pretraining dataset | Data Fraction | Testing Method |
|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | HHAR | WISDM | 6ch | Yes | None | 100% | Cross |
| SSL-Wearables | Watch | HHAR | WISDM | 3ch | Yes | None | 100% | Cross |
| LIMU-BERT-X | Phone | WISDM | HHAR | 6ch | Yes | None | 100% | Cross |
| SSL-Wearables | Watch | WISDM | HHAR | 3ch | Yes | None | 100% | Cross |

> Note: `eval_cross_dataset.py` now supports `--pretrained` for `limu-bert` and `ssl-wearables`.
> The commands below explicitly load imported pretrained weights (`Imported Pretrained Weights = Yes`).

**Commands (Exp5/6 priority):**

```powershell
# --- Exp5: HHAR -> WISDM (Cross) ---
# Run 1: LIMU-BERT-X, phone, 6ch
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --data-fraction 1.0 --experiment-tag exp5_add_limuX_phone_6ch --seed 42 --repeat-seeds 3

# Run 2: SSL-Wearables, watch, 3ch
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 1.0 --experiment-tag exp5_add_ssl_watch_3ch --seed 42 --repeat-seeds 3

# --- Exp6: WISDM -> HHAR (Cross, reverse) ---
# Run 3: LIMU-BERT-X, phone, 6ch
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --channels 6 --data-fraction 1.0 --reverse --experiment-tag exp6_add_limuX_phone_6ch --seed 42 --repeat-seeds 3

# Run 4: SSL-Wearables, watch, 3ch
python eval_cross_dataset.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 1.0 --reverse --experiment-tag exp6_add_ssl_watch_3ch --seed 42 --repeat-seeds 3
```

### Experiment 1 Additional Training (after Exp5/6)

> Do all configurations with **3 trials** (`--seed 42 --repeat-seeds 3`), as requested.

| Architecture Type | Device Type | Data Fraction | 3ch vs 6ch data | Imported Pretrained Weights | Additional Pretraining dataset | Training Dataset | Testing Dataset | Testing Method |
|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | 75% | 6ch | No | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 75% | 3ch | No | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 50% | 6ch | No | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 50% | 3ch | No | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 25% | 6ch | No | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 25% | 3ch | No | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 12.5% | 6ch | Yes | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 12.5% | 3ch | Yes | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 12.5% | 6ch | No | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 12.5% | 3ch | No | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 5% | 6ch | Yes | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 5% | 3ch | Yes | None | HHAR | HHAR | LOSO |
| LIMU-BERT-X | Phone | 5% | 6ch | No | None | HHAR | HHAR | LOSO |
| SSL-Wearables | Watch | 5% | 3ch | No | None | HHAR | HHAR | LOSO |

**Commands--Exp1 added :**

```powershell
# --- 75% ---
# Run 5: LIMU-BERT-X, phone, 75%, 6ch, pretrained=No
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --channels 6 --data-fraction 0.75 --no-bike --experiment-tag exp1_add_limuX_phone_75_scratch --seed 42 --repeat-seeds 3
# Run 6: SSL-Wearables, watch, 75%, 3ch, pretrained=No
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --data-fraction 0.75 --no-bike --experiment-tag exp1_add_ssl_watch_75_scratch --seed 42 --repeat-seeds 3

# --- 50% ---
# Run 7: LIMU-BERT-X, phone, 50%, 6ch, pretrained=No
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --channels 6 --data-fraction 0.5 --no-bike --experiment-tag exp1_add_limuX_phone_50_scratch --seed 42 --repeat-seeds 3
# Run 8: SSL-Wearables, watch, 50%, 3ch, pretrained=No
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --data-fraction 0.5 --no-bike --experiment-tag exp1_add_ssl_watch_50_scratch --seed 42 --repeat-seeds 3

# --- 25% ---
# Run 9: LIMU-BERT-X, phone, 25%, 6ch, pretrained=No
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --channels 6 --data-fraction 0.25 --no-bike --experiment-tag exp1_add_limuX_phone_25_scratch --seed 42 --repeat-seeds 3
# Run 10: SSL-Wearables, watch, 25%, 3ch, pretrained=No
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --data-fraction 0.25 --no-bike --experiment-tag exp1_add_ssl_watch_25_scratch --seed 42 --repeat-seeds 3

# --- 12.5% (pretrained = Yes) ---
# Run 11: LIMU-BERT-X, phone, 12.5%, 6ch, pretrained=Yes
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --channels 6 --data-fraction 0.125 --no-bike --experiment-tag exp1_add_limuX_phone_125_pretrained --seed 42 --repeat-seeds 3
# Run 12: SSL-Wearables, watch, 12.5%, 3ch, pretrained=Yes
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.125 --no-bike --experiment-tag exp1_add_ssl_watch_125_pretrained --seed 42 --repeat-seeds 3

# --- 12.5% (pretrained = No) ---
# Run 13: LIMU-BERT-X, phone, 12.5%, 6ch, pretrained=No
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --channels 6 --data-fraction 0.125 --no-bike --experiment-tag exp1_add_limuX_phone_125_scratch --seed 42 --repeat-seeds 3
# Run 14: SSL-Wearables, watch, 12.5%, 3ch, pretrained=No
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --data-fraction 0.125 --no-bike --experiment-tag exp1_add_ssl_watch_125_scratch --seed 42 --repeat-seeds 3

# --- 5% (pretrained = Yes) ---
# Run 15: LIMU-BERT-X, phone, 5%, 6ch, pretrained=Yes
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type phone --channels 6 --data-fraction 0.05 --no-bike --experiment-tag exp1_add_limuX_phone_05_pretrained --seed 42 --repeat-seeds 3
# Run 16: SSL-Wearables, watch, 5%, 3ch, pretrained=Yes
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --data-fraction 0.05 --no-bike --experiment-tag exp1_add_ssl_watch_05_pretrained --seed 42 --repeat-seeds 3

# --- 5% (pretrained = No) ---
# Run 17: LIMU-BERT-X, phone, 5%, 6ch, pretrained=No
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --channels 6 --data-fraction 0.05 --no-bike --experiment-tag exp1_add_limuX_phone_05_scratch --seed 42 --repeat-seeds 3
# Run 18: SSL-Wearables, watch, 5%, 3ch, pretrained=No
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --data-fraction 0.05 --no-bike --experiment-tag exp1_add_ssl_watch_05_scratch --seed 42 --repeat-seeds 3
```

### Experiment 2 Additional Trainings (after Exp1 added)

> Run all with **3 trials** (`--seed 42 --repeat-seeds 3`), LOSO on HHAR no-bike.

| Architecture Type | Device Type | 3ch vs 6ch data | Additional Pretraining dataset | Imported Pretrained Weights | Data Fraction | Training Dataset | Testing Dataset | Testing Method |
|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Watch | 6ch | None | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 3ch | PAMAP2 + WISDM (Watch only) | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 3ch | WISDM (Watch and Phone) | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 6ch | PAMAP2 + WISDM (Watch only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 6ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 3ch | PAMAP2 + WISDM (Watch only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | Watch | 3ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | PAMAP2 + WISDM (Watch only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | None | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | PAMAP2 + WISDM (Watch only) | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | WISDM (Watch and Phone) | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | PAMAP2 + WISDM (Watch only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 6ch | PAMAP2 + WISDM (Watch only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 6ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |

**Commands--Exp2 added:**

```powershell
# Note on "--pretrained" usage:
# - Imported Pretrained Weights = Yes + Additional Pretraining dataset = None
#   -> load base checkpoint (e.g., weights/limu_bert_x or model_check_point/mtl_best.mdl).
# - Imported Pretrained Weights = No + Additional Pretraining dataset != None
#   -> still load pretrain_data/* checkpoint (these are additional-pretraining outputs, not base imported weights).

# Run 19: LIMU-BERT-X (watch, 6ch, imported pretrained = Yes, additional pretraining = None)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type watch --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_limuX_watch_6ch_pretrained --seed 42 --repeat-seeds 3

# Run 26: LIMU-BERT-X (all, 6ch, watch-only scratch pretraining)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_watch_scratch.pt --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_limuX_all_6ch_watchscratch --seed 42 --repeat-seeds 3

# SSL-Wearables (watch, 3ch)
# Run 20: SSL-Wearables (watch, 3ch, PAMAP2 + WISDM watch-only, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_finetune_3ch.mdl --device-type watch --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_3ch_watchft --seed 42 --repeat-seeds 3
# Run 21: SSL-Wearables (watch, 3ch, WISDM all, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_finetune_3ch.mdl --device-type watch --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_3ch_wisdmft --seed 42 --repeat-seeds 3
# Run 22: SSL-Wearables (watch, 6ch, PAMAP2 + WISDM watch-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_scratch_6ch.mdl --device-type watch --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_6ch_watchscratch --seed 42 --repeat-seeds 3
# Run 23: SSL-Wearables (watch, 6ch, WISDM all, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_6ch.mdl --device-type watch --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_6ch_wisdmscratch --seed 42 --repeat-seeds 3
# Run 24: SSL-Wearables (watch, 3ch, PAMAP2 + WISDM watch-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_scratch_3ch.mdl --device-type watch --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_3ch_watchscratch --seed 42 --repeat-seeds 3
# Run 25: SSL-Wearables (watch, 3ch, WISDM all, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_3ch.mdl --device-type watch --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_watch_3ch_wisdmscratch --seed 42 --repeat-seeds 3

# SSL-Wearables (all, 3ch/6ch)
# Run 27: SSL-Wearables (all, 3ch, base pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_3ch_basepretrained --seed 42 --repeat-seeds 3
# Run 28: SSL-Wearables (all, 3ch, PAMAP2 + WISDM watch-only, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_finetune_3ch.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_3ch_watchft --seed 42 --repeat-seeds 3
# Run 29: SSL-Wearables (all, 3ch, WISDM all, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_finetune_3ch.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_3ch_wisdmft --seed 42 --repeat-seeds 3
# Run 30: SSL-Wearables (all, 3ch, PAMAP2 + WISDM watch-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_scratch_3ch.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_3ch_watchscratch --seed 42 --repeat-seeds 3
# Run 31: SSL-Wearables (all, 3ch, WISDM all, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_3ch.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_3ch_wisdmscratch --seed 42 --repeat-seeds 3
# Run 32: SSL-Wearables (all, 6ch, PAMAP2 + WISDM watch-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_watch_scratch_6ch.mdl --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_6ch_watchscratch --seed 42 --repeat-seeds 3
# Run 33: SSL-Wearables (all, 6ch, WISDM all, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_wisdm_all_scratch_6ch.mdl --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp2_add_ssl_all_6ch_wisdmscratch --seed 42 --repeat-seeds 3
```

> Note: commands that use `pretrain_data/ssl_watch_*` weights assume those watch-only SSL checkpoints have already been created.

### Experiment 3 Additional Trainings (same format)

> Run all with **3 trials** (`--seed 42 --repeat-seeds 3`), LOSO on HHAR no-bike.

| Architecture Type | Device Type | 3ch vs 6ch data | Additional Pretraining dataset | Imported Pretrained Weights | Data Fraction | Training Dataset | Testing Dataset | Testing Method |
|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | 6ch | SBHAR + WISDM (Phone only) | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | Phone | 6ch | WISDM (Watch and Phone) | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | Phone | 6ch | SBHAR + WISDM (Phone only) | No | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | Phone | 6ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | Phone | 3ch | None | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | None | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | SBHAR + WISDM (Phone only) | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | WISDM (Watch and Phone) | Yes | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | SBHAR + WISDM (Phone only) | No | 100% | None | HHAR | LOSO |
| LIMU-BERT-X | All | 6ch | WISDM (Watch and Phone) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | None | Yes | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 3ch | SBHAR + WISDM (Phone only) | No | 100% | None | HHAR | LOSO |
| SSL-Wearables | All | 6ch | SBHAR + WISDM (Phone only) | No | 100% | None | HHAR | LOSO |

**Commands--Exp3 added:**

```powershell
# LIMU-BERT-X (phone, 6ch)
# Run 34: LIMU-BERT-X (phone, 6ch, SBHAR + WISDM phone-only, pretrained=Yes)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_phone_finetune.pt --device-type phone --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_phone_6ch_phoneft --seed 42 --repeat-seeds 3
# Run 35: LIMU-BERT-X (phone, 6ch, WISDM all, pretrained=Yes)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_finetune.pt --device-type phone --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_phone_6ch_wisdmft --seed 42 --repeat-seeds 3
# Run 36: LIMU-BERT-X (phone, 6ch, SBHAR + WISDM phone-only, pretrained=No)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_phone_scratch.pt --device-type phone --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_phone_6ch_phonescratch --seed 42 --repeat-seeds 3
# Run 37: LIMU-BERT-X (phone, 6ch, WISDM all, pretrained=No)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_scratch.pt --device-type phone --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_phone_6ch_wisdmscratch --seed 42 --repeat-seeds 3

# SSL-Wearables (phone/all, 3ch/6ch)
# Run 38: SSL-Wearables (phone, 3ch, additional pretraining=None, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type phone --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_ssl_phone_3ch_basepretrained --seed 42 --repeat-seeds 3
# Run 44: SSL-Wearables (all, 3ch, additional pretraining=None, pretrained=Yes)
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_ssl_all_3ch_basepretrained --seed 42 --repeat-seeds 3
# Run 45: SSL-Wearables (all, 3ch, SBHAR + WISDM phone-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_scratch_3ch.mdl --device-type all --channels 3 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_ssl_all_3ch_phonescratch --seed 42 --repeat-seeds 3
# Run 46: SSL-Wearables (all, 6ch, SBHAR + WISDM phone-only, pretrained=No)
python train_loso.py --model ssl-wearables --pretrained pretrain_data/ssl_phone_scratch_6ch.mdl --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_ssl_all_6ch_phonescratch --seed 42 --repeat-seeds 3

# LIMU-BERT-X (all, 6ch)
# Run 39: LIMU-BERT-X (all, 6ch, additional pretraining=None, pretrained=Yes)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_all_6ch_basepretrained --seed 42 --repeat-seeds 3
# Run 40: LIMU-BERT-X (all, 6ch, SBHAR + WISDM phone-only, pretrained=Yes)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_phone_finetune.pt --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_all_6ch_phoneft --seed 42 --repeat-seeds 3
# Run 41: LIMU-BERT-X (all, 6ch, WISDM all, pretrained=Yes)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_finetune.pt --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_all_6ch_wisdmft --seed 42 --repeat-seeds 3
# Run 42: LIMU-BERT-X (all, 6ch, SBHAR + WISDM phone-only, pretrained=No)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_phone_scratch.pt --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_all_6ch_phonescratch --seed 42 --repeat-seeds 3
# Run 43: LIMU-BERT-X (all, 6ch, WISDM all, pretrained=No)
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained pretrain_data/limu_wisdm_all_scratch.pt --device-type all --channels 6 --data-fraction 1.0 --no-bike --experiment-tag exp3_add_limuX_all_6ch_wisdmscratch --seed 42 --repeat-seeds 3
```
**Experiment 7 is the old experiment 9**:

```powershell
# Example: 3 repeated seeds for LIMU-BERT-X all
python eval_cross_dataset.py --model limu-bert --limu-seq-len 20 --device-type all --reverse --experiment-tag exp6_limubertx_all --seed 42 --repeat-seeds 3
```
### Experiment 7 -- Leave-One-In Generalization (K-Subject Fine-Tuning)

> **Goal:** Run a complete leave-one-subject-in (LOSI) matrix on HHAR, covering the requested architecture/device/pretraining settings with train-subject counts `K in {1, 5, 8}`.
> 
> This is a variation of Experiment 1, but instead of using a percentage of the data (`--data-fraction`), we use a **leave-one-in** protocol:
> train on selected subjects (`--train-subjects K --leave-one-in`) and report test accuracy on each remaining subject individually.
>
> **Subject set note:** HHAR uses 9 subjects here, consistent with Experiment 1 settings.
>
> **LOSI note:** All commands below are LOSI (`--leave-one-in`).

**Commands:**

```powershell
# --- Part A: LIMU-BERT-X (Transformer, 6ch) ---
# A1) pretrained=Yes, phone, K=1/5/8
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x.pt --device-type phone --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_pretrained_phone_8subj --seed 42 --repeat-seeds 3

# A2) pretrained=No, phone/watch/all, K=1/5/8
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type phone --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type watch --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model limu-bert --limu-seq-len 20 --device-type all --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9a_limu_scratch_all_8subj --seed 42 --repeat-seeds 3

# --- Part B: ssl-wearables (CNN, 3ch) ---
# B1) pretrained=Yes, watch, K=1/5/8
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_pretrained_watch_8subj --seed 42 --repeat-seeds 3

# B2) pretrained=No, phone/watch/all, K=1/5/8
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type phone --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model ssl-wearables --device-type all --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9b_ssl_scratch_all_8subj --seed 42 --repeat-seeds 3

# --- Part C: HART (Transformer, 6ch), pretrained=No, K=1/5/8 ---
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type phone --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type watch --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model hart --device-type all --channels 6 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9c_hart_all_8subj --seed 42 --repeat-seeds 3

# --- Part D: ResNet-Baseline (CNN, 3ch), pretrained=No, K=1/5/8 ---
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 1 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_1subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 5 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_5subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type phone --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_phone_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type watch --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_watch_8subj --seed 42 --repeat-seeds 3
python train_loso.py --model resnet-baseline --device-type all --channels 3 --train-subjects 8 --leave-one-in --no-bike --experiment-tag exp9d_resbase_all_8subj --seed 42 --repeat-seeds 3
```

> `--leave-one-in` output format: for each train-subject fold, the script prints `train=<id> -> test=<id>: Acc=...` for every remaining subject instead of only reporting a single fold mean.
>
> **Combination count by K (from 9 subjects):**
> - `K=1`: `C(9,1)=9` train combinations, `8` held-out subjects each fold
> - `K=5`: `C(9,5)=126` train combinations, `4` held-out subjects each fold
> - `K=8`: `C(9,8)=9` train combinations, `1` held-out subject each fold


### Experiment 8 -- Custom-Collected Data Evaluation

> **Goal:** Evaluate all model architectures on our own custom-collected Xsens IMU data.
>
> **Two modes:**
> - **Mode A (cross):** Use HHAR-trained Exp0 scratch models directly on custom data (no retraining). Tests whether HHAR generalization holds on completely new hardware and subjects.
> - **Mode B (LOSO):** Leave-one-subject-out within the custom dataset only. Trains from scratch on 4 subjects, tests on 1.
>
> **Custom data characteristics:**
> - **Source:** `data_combined/` (TXT files only; ignore MTB files)
> - **5 subjects:** B, C, D, O, Y
> - **4 Xsens sensors per recording:** right wrist, left wrist (watch), right pocket, left pocket (phone)
> - **5 activities:** sit (0), stand (1), walk (2), stairUp (3), stairDown (4)
> - **2 conditions:** control (controlled posture), uncontrolled (natural posture)
> - **6 channels available:** `Acc_X/Y/Z + Roll/Pitch/Yaw`, ~100 Hz, ~2 min per recording.
> - GT is parsed from filename: subject / activity / condition / duration / device id.

**Cross-dataset (Mode A):**

| ID | Model | Mode | Device | Condition | Tag |
|---|---|---|---|---|---|
| **8A** | HART | cross | all | all | `exp8a_hart_cross_all` |
| **8C** | LIMU-BERT-X | cross | all | all | `exp8c_limuX_cross_all` |
| **8E** | ssl-wearables | cross | all | all | `exp8e_ssl_cross_all` |
| **8G** | ResNet-Baseline | cross | all | all | `exp8g_resbase_cross_all` |
| **8I** | HART | cross | watch | control | `exp8i_hart_cross_watch_ctrl` |
| **8J** | HART | cross | phone | control | `exp8j_hart_cross_phone_ctrl` |
| **8K** | HART | cross | watch | uncontrolled | `exp8k_hart_cross_watch_unc` |
| **8L** | HART | cross | phone | uncontrolled | `exp8l_hart_cross_phone_unc` |

**LOSO (Mode B) -- all condition × device combinations (9 combos × 4 models = 36 runs):**

| Model | Device | Condition | Tag |
|---|---|---|---|
| HART | all | all | `exp8b_hart_loso_all_all` |
| HART | phone | all | `exp8b_hart_loso_all_phone` |
| HART | watch | all | `exp8b_hart_loso_all_watch` |
| HART | all | control | `exp8b_hart_loso_ctrl_all` |
| HART | phone | control | `exp8b_hart_loso_ctrl_phone` |
| HART | watch | control | `exp8b_hart_loso_ctrl_watch` |
| HART | all | uncontrolled | `exp8b_hart_loso_unc_all` |
| HART | phone | uncontrolled | `exp8b_hart_loso_unc_phone` |
| HART | watch | uncontrolled | `exp8b_hart_loso_unc_watch` |
| LIMU-BERT-X | all | all | `exp8d_limuX_loso_all_all` |
| LIMU-BERT-X | phone | all | `exp8d_limuX_loso_all_phone` |
| LIMU-BERT-X | watch | all | `exp8d_limuX_loso_all_watch` |
| LIMU-BERT-X | all | control | `exp8d_limuX_loso_ctrl_all` |
| LIMU-BERT-X | phone | control | `exp8d_limuX_loso_ctrl_phone` |
| LIMU-BERT-X | watch | control | `exp8d_limuX_loso_ctrl_watch` |
| LIMU-BERT-X | all | uncontrolled | `exp8d_limuX_loso_unc_all` |
| LIMU-BERT-X | phone | uncontrolled | `exp8d_limuX_loso_unc_phone` |
| LIMU-BERT-X | watch | uncontrolled | `exp8d_limuX_loso_unc_watch` |
| ssl-wearables | all | all | `exp8f_ssl_loso_all_all` |
| ssl-wearables | phone | all | `exp8f_ssl_loso_all_phone` |
| ssl-wearables | watch | all | `exp8f_ssl_loso_all_watch` |
| ssl-wearables | all | control | `exp8f_ssl_loso_ctrl_all` |
| ssl-wearables | phone | control | `exp8f_ssl_loso_ctrl_phone` |
| ssl-wearables | watch | control | `exp8f_ssl_loso_ctrl_watch` |
| ssl-wearables | all | uncontrolled | `exp8f_ssl_loso_unc_all` |
| ssl-wearables | phone | uncontrolled | `exp8f_ssl_loso_unc_phone` |
| ssl-wearables | watch | uncontrolled | `exp8f_ssl_loso_unc_watch` |
| ResNet-Baseline | all | all | `exp8h_resbase_loso_all_all` |
| ResNet-Baseline | phone | all | `exp8h_resbase_loso_all_phone` |
| ResNet-Baseline | watch | all | `exp8h_resbase_loso_all_watch` |
| ResNet-Baseline | all | control | `exp8h_resbase_loso_ctrl_all` |
| ResNet-Baseline | phone | control | `exp8h_resbase_loso_ctrl_phone` |
| ResNet-Baseline | watch | control | `exp8h_resbase_loso_ctrl_watch` |
| ResNet-Baseline | all | uncontrolled | `exp8h_resbase_loso_unc_all` |
| ResNet-Baseline | phone | uncontrolled | `exp8h_resbase_loso_unc_phone` |
| ResNet-Baseline | watch | uncontrolled | `exp8h_resbase_loso_unc_watch` |

**Commands:**

```powershell
# Step 1: Prepare custom data (all conditions, all devices)
python prepare_custom_data.py --force

# Step 2A: Cross-dataset evaluation (HHAR-trained models -> custom data)
# Upstream run IDs mapped to Experiment 8 matrix (Run 47-88).
# Use --custom-test-repeats 3 for three custom-data tests per run.

# --- Our Controlled (condition=control) ---
# Run 47: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run47_limuX_phone_ctrl_preY
# Run 48: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run48_limuX_phone_ctrl_preN
# Run 49
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run49_limuX_watch_ctrl
# Run 50
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run50_limuX_all_ctrl
# Run 51
python eval_custom_data.py --model hart --mode cross --device-type phone --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run51_hart_phone_ctrl
# Run 52
python eval_custom_data.py --model hart --mode cross --device-type watch --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run52_hart_watch_ctrl
# Run 53
python eval_custom_data.py --model hart --mode cross --device-type all --condition control --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run53_hart_all_ctrl
# Run 54: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run54_ssl_watch_ctrl_preY
# Run 55
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run55_ssl_phone_ctrl
# Run 56: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run56_ssl_watch_ctrl_preN
# Run 57
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run57_ssl_all_ctrl
# Run 58
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run58_resbase_phone_ctrl_3ch
# Run 59
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run59_resbase_watch_ctrl_3ch
# Run 60
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition control --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run60_resbase_all_ctrl_3ch

# --- Our Uncontrolled (condition=uncontrolled) ---
# Run 61: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run61_limuX_phone_unc_preY
# Run 62: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run62_limuX_phone_unc_preN
# Run 63
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run63_limuX_watch_unc
# Run 64
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run64_limuX_all_unc
# Run 65
python eval_custom_data.py --model hart --mode cross --device-type phone --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run65_hart_phone_unc
# Run 66
python eval_custom_data.py --model hart --mode cross --device-type watch --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run66_hart_watch_unc
# Run 67
python eval_custom_data.py --model hart --mode cross --device-type all --condition uncontrolled --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run67_hart_all_unc
# Run 68: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run68_ssl_watch_unc_preY
# Run 69
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run69_ssl_phone_unc
# Run 70: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run70_ssl_watch_unc_preN
# Run 71
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run71_ssl_all_unc
# Run 72
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run72_resbase_phone_unc_3ch
# Run 73
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run73_resbase_watch_unc_3ch
# Run 74
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition uncontrolled --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run74_resbase_all_unc_3ch

# --- Our All (condition=all) ---
# Run 75: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run75_limuX_phone_all_preY
# Run 76: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run76_limuX_phone_all_preN
# Run 77
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type watch --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run77_limuX_watch_all
# Run 78
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode cross --device-type all --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run78_limuX_all_all
# Run 79
python eval_custom_data.py --model hart --mode cross --device-type phone --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run79_hart_phone_all
# Run 80
python eval_custom_data.py --model hart --mode cross --device-type watch --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run80_hart_watch_all
# Run 81
python eval_custom_data.py --model hart --mode cross --device-type all --condition all --channels 6 --custom-test-repeats 3 --experiment-tag exp8_run81_hart_all_all
# Run 82: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run82_ssl_watch_all_preY
# Run 83
python eval_custom_data.py --model ssl-wearables --mode cross --device-type phone --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run83_ssl_phone_all
# Run 84: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run84_ssl_watch_all_preN
# Run 85
python eval_custom_data.py --model ssl-wearables --mode cross --device-type all --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run85_ssl_all_all
# Run 86
python eval_custom_data.py --model resnet-baseline --mode cross --device-type phone --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run86_resbase_phone_all_3ch
# Run 87
python eval_custom_data.py --model resnet-baseline --mode cross --device-type watch --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run87_resbase_watch_all_3ch
# Run 88
python eval_custom_data.py --model resnet-baseline --mode cross --device-type all --condition all --channels 3 --custom-test-repeats 3 --experiment-tag exp8_run88_resbase_all_all_3ch

# Note:
# - eval_custom_data.py does not expose --repeat-seeds; use --custom-test-repeats 3 here.
# - "Imported Pretrained Weights = Yes" rows explicitly pass --pretrained in commands.

# Step 2B: LOSO within custom data (all condition × device combinations)
# --- HART (all 9 combos) ---
# Run 89
python eval_custom_data.py --model hart --mode loso --device-type all --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_all_run89 --seed 42
# Run 90
python eval_custom_data.py --model hart --mode loso --device-type phone --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_phone_run90 --seed 42
# Run 91
python eval_custom_data.py --model hart --mode loso --device-type watch --condition all --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_all_watch_run91 --seed 42
# Run 92
python eval_custom_data.py --model hart --mode loso --device-type all --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_all_run92 --seed 42
# Run 93
python eval_custom_data.py --model hart --mode loso --device-type phone --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_phone_run93 --seed 42
# Run 94
python eval_custom_data.py --model hart --mode loso --device-type watch --condition control --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_ctrl_watch_run94 --seed 42
# Run 95
python eval_custom_data.py --model hart --mode loso --device-type all --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_all_run95 --seed 42
# Run 96
python eval_custom_data.py --model hart --mode loso --device-type phone --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_phone_run96 --seed 42
# Run 97
python eval_custom_data.py --model hart --mode loso --device-type watch --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp8b_hart_loso_unc_watch_run97 --seed 42

# --- LIMU-BERT-X (all 9 combos) ---
# Run 98
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_all_run98 --seed 42
# Run 99
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_phone_run99 --seed 42
# Run 100
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8d_limuX_loso_all_watch_run100 --seed 42
# Run 101
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_all_run101 --seed 42
# Run 102
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_phone_run102 --seed 42
# Run 103
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8d_limuX_loso_ctrl_watch_run103 --seed 42
# Run 104
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_all_run104 --seed 42
# Run 105
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_phone_run105 --seed 42
# Run 106
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8d_limuX_loso_unc_watch_run106 --seed 42

# --- ssl-wearables 6ch (all 9 combos) ---
# Run 107
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_all_run107 --seed 42
# Run 108
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_phone_run108 --seed 42
# Run 109
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8f_ssl_loso_all_watch_run109 --seed 42
# Run 110
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_all_run110 --seed 42
# Run 111
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_phone_run111 --seed 42
# Run 112
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8f_ssl_loso_ctrl_watch_run112 --seed 42
# Run 113
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_all_run113 --seed 42
# Run 114
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_phone_run114 --seed 42
# Run 115
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8f_ssl_loso_unc_watch_run115 --seed 42

# --- ResNet-Baseline 6ch (all 9 combos) ---
# Run 116
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_all_run116 --seed 42
# Run 117
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_phone_run117 --seed 42
# Run 118
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp8h_resbase_loso_all_watch_run118 --seed 42
# Run 119
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_all_run119 --seed 42
# Run 120
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_phone_run120 --seed 42
# Run 121
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp8h_resbase_loso_ctrl_watch_run121 --seed 42
# Run 122
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_all_run122 --seed 42
# Run 123
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_phone_run123 --seed 42
# Run 124
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp8h_resbase_loso_unc_watch_run124 --seed 42
```

**Reported metrics:**

- Mode A (cross): accuracy, F1 (weighted/macro), per-subject, per-device, per-class metrics, confusion matrix
- Mode A std: available when `--custom-test-repeats > 1` (stored in repeat summary).
- Mode B (LOSO): per-fold accuracy/F1 + mean/std (accuracy + F1), per-device breakdown

**Notes:**

- Custom data is parsed from `data_combined` **TXT** files only (MTB files are ignored).
- Custom data is **6ch (`Acc_X/Y/Z + Roll/Pitch/Yaw`)**.
- If a cross-dataset checkpoint was trained in 3ch only, run with `--channels 3` for that model to avoid input-shape mismatch.
- The `--condition` flag enables separate evaluation of `control` vs `uncontrolled` recordings.
- The `--device-type` flag supports `watch` (wrist sensors), `phone` (pocket sensors), or `all`.
- Mode A loads the best fold-0 checkpoint from Exp0 LOSO training. Make sure Exp0 has been run first.
- LOSO in Mode B uses only 5 subjects, so variance may still be high.

---

### Experiment 8.5 -- Custom Data LOSO Matrix (Table-Aligned)

> **Goal:** Run a second LOSO matrix on custom data with tags aligned to your table schema.
>
> This section is designed to be summarized with:
> `python summarize_experiment_results.py --exp exp85`
>
> All runs below are LOSO on custom data (`mode=loso`) and write `experiment_table_metadata`
> so output CSV columns match your requested template.

| Architecture Type | Device Type | Testing Dataset | Imported Pretrained Weights | 3ch vs 6ch data | Data Fraction | Additional Pretraining dataset | Training Dataset | Testing Method | # Run |
|---|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | Our Controlled | No | 6ch | 100% | None | None | LOSO | 124 |
| LIMU-BERT-X | Watch | Our Controlled | No | 6ch | 100% | None | None | LOSO | 125 |
| LIMU-BERT-X | All | Our Controlled | No | 6ch | 100% | None | None | LOSO | 126 |
| LIMU-BERT-X | Phone | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 127 |
| LIMU-BERT-X | Watch | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 128 |
| LIMU-BERT-X | All | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 129 |
| LIMU-BERT-X | Phone | Our All | No | 6ch | 100% | None | None | LOSO | 130 |
| LIMU-BERT-X | Watch | Our All | No | 6ch | 100% | None | None | LOSO | 131 |
| LIMU-BERT-X | All | Our All | No | 6ch | 100% | None | None | LOSO | 132 |
| LIMU-BERT-X | Phone | Our Controlled | Yes | 6ch | 100% | None | None | LOSO | 133 |
| LIMU-BERT-X | Phone | Our Uncontrolled | Yes | 6ch | 100% | None | None | LOSO | 134 |
| LIMU-BERT-X | Phone | Our All | Yes | 6ch | 100% | None | None | LOSO | 135 |
| HART | Phone | Our Controlled | No | 6ch | 100% | None | None | LOSO | 136 |
| HART | Watch | Our Controlled | No | 6ch | 100% | None | None | LOSO | 137 |
| HART | All | Our Controlled | No | 6ch | 100% | None | None | LOSO | 138 |
| HART | Phone | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 139 |
| HART | Watch | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 140 |
| HART | All | Our Uncontrolled | No | 6ch | 100% | None | None | LOSO | 141 |
| HART | Phone | Our All | No | 6ch | 100% | None | None | LOSO | 142 |
| HART | Watch | Our All | No | 6ch | 100% | None | None | LOSO | 143 |
| HART | All | Our All | No | 6ch | 100% | None | None | LOSO | 144 |
| SSL-Wearables | Phone | Our Controlled | No | 3ch | 100% | None | None | LOSO | 145 |
| SSL-Wearables | Watch | Our Controlled | No | 3ch | 100% | None | None | LOSO | 146 |
| SSL-Wearables | All | Our Controlled | No | 3ch | 100% | None | None | LOSO | 147 |
| SSL-Wearables | Phone | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 148 |
| SSL-Wearables | Watch | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 149 |
| SSL-Wearables | All | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 150 |
| SSL-Wearables | Phone | Our All | No | 3ch | 100% | None | None | LOSO | 151 |
| SSL-Wearables | Watch | Our All | No | 3ch | 100% | None | None | LOSO | 152 |
| SSL-Wearables | All | Our All | No | 3ch | 100% | None | None | LOSO | 153 |
| SSL-Wearables | Watch | Our Controlled | Yes | 3ch | 100% | None | None | LOSO | 154 |
| SSL-Wearables | Watch | Our Uncontrolled | Yes | 3ch | 100% | None | None | LOSO | 155 |
| SSL-Wearables | Watch | Our All | Yes | 3ch | 100% | None | None | LOSO | 156 |
| ResNet-Baseline | Phone | Our Controlled | No | 3ch | 100% | None | None | LOSO | 157 |
| ResNet-Baseline | Watch | Our Controlled | No | 3ch | 100% | None | None | LOSO | 158 |
| ResNet-Baseline | All | Our Controlled | No | 3ch | 100% | None | None | LOSO | 159 |
| ResNet-Baseline | Phone | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 160 |
| ResNet-Baseline | Watch | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 161 |
| ResNet-Baseline | All | Our Uncontrolled | No | 3ch | 100% | None | None | LOSO | 162 |
| ResNet-Baseline | Phone | Our All | No | 3ch | 100% | None | None | LOSO | 163 |
| ResNet-Baseline | Watch | Our All | No | 3ch | 100% | None | None | LOSO | 164 |
| ResNet-Baseline | All | Our All | No | 3ch | 100% | None | None | LOSO | 165 |

```powershell
# --- LIMU-BERT-X (No pretrained): Run 124-132 ---
# Run 124
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp85_run124_limuX_phone_ctrl_preN --seed 42
# Run 125
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition control --channels 6 --experiment-tag exp85_run125_limuX_watch_ctrl_preN --seed 42
# Run 126
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition control --channels 6 --experiment-tag exp85_run126_limuX_all_ctrl_preN --seed 42
# Run 127
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp85_run127_limuX_phone_unc_preN --seed 42
# Run 128
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp85_run128_limuX_watch_unc_preN --seed 42
# Run 129
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition uncontrolled --channels 6 --experiment-tag exp85_run129_limuX_all_unc_preN --seed 42
# Run 130
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp85_run130_limuX_phone_all_preN --seed 42
# Run 131
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type watch --condition all --channels 6 --experiment-tag exp85_run131_limuX_watch_all_preN --seed 42
# Run 132
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode loso --device-type all --condition all --channels 6 --experiment-tag exp85_run132_limuX_all_all_preN --seed 42

# --- LIMU-BERT-X (Imported pretrained=Yes): Run 133-135 ---
# Run 133
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition control --channels 6 --experiment-tag exp85_run133_limuX_phone_ctrl_preY --seed 42
# Run 134
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp85_run134_limuX_phone_unc_preY --seed 42
# Run 135
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode loso --device-type phone --condition all --channels 6 --experiment-tag exp85_run135_limuX_phone_all_preY --seed 42

# --- HART: Run 136-144 ---
# Run 136
python eval_custom_data.py --model hart --mode loso --device-type phone --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run136_hart_phone_ctrl --seed 42
# Run 137
python eval_custom_data.py --model hart --mode loso --device-type watch --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run137_hart_watch_ctrl --seed 42
# Run 138
python eval_custom_data.py --model hart --mode loso --device-type all --condition control --channels 6 --batch-size 64 --experiment-tag exp85_run138_hart_all_ctrl --seed 42
# Run 139
python eval_custom_data.py --model hart --mode loso --device-type phone --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run139_hart_phone_unc --seed 42
# Run 140
python eval_custom_data.py --model hart --mode loso --device-type watch --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run140_hart_watch_unc --seed 42
# Run 141
python eval_custom_data.py --model hart --mode loso --device-type all --condition uncontrolled --channels 6 --batch-size 64 --experiment-tag exp85_run141_hart_all_unc --seed 42
# Run 142
python eval_custom_data.py --model hart --mode loso --device-type phone --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run142_hart_phone_all --seed 42
# Run 143
python eval_custom_data.py --model hart --mode loso --device-type watch --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run143_hart_watch_all --seed 42
# Run 144
python eval_custom_data.py --model hart --mode loso --device-type all --condition all --channels 6 --batch-size 64 --experiment-tag exp85_run144_hart_all_all --seed 42

# --- SSL-Wearables (No pretrained): Run 145-153 ---
# Run 145
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition control --channels 3 --experiment-tag exp85_run145_ssl_phone_ctrl_preN --seed 42
# Run 146
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run146_ssl_watch_ctrl_preN --seed 42
# Run 147
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition control --channels 3 --experiment-tag exp85_run147_ssl_all_ctrl_preN --seed 42
# Run 148
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp85_run148_ssl_phone_unc_preN --seed 42
# Run 149
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run149_ssl_watch_unc_preN --seed 42
# Run 150
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition uncontrolled --channels 3 --experiment-tag exp85_run150_ssl_all_unc_preN --seed 42
# Run 151
python eval_custom_data.py --model ssl-wearables --mode loso --device-type phone --condition all --channels 3 --experiment-tag exp85_run151_ssl_phone_all_preN --seed 42
# Run 152
python eval_custom_data.py --model ssl-wearables --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run152_ssl_watch_all_preN --seed 42
# Run 153
python eval_custom_data.py --model ssl-wearables --mode loso --device-type all --condition all --channels 3 --experiment-tag exp85_run153_ssl_all_all_preN --seed 42

# --- SSL-Wearables (Imported pretrained=Yes): Run 154-156 ---
# Run 154
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run154_ssl_watch_ctrl_preY --seed 42
# Run 155
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run155_ssl_watch_unc_preY --seed 42
# Run 156
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run156_ssl_watch_all_preY --seed 42

# --- ResNet-Baseline: Run 157-165 ---
# Run 157
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition control --channels 3 --experiment-tag exp85_run157_resbase_phone_ctrl_3ch --seed 42
# Run 158
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition control --channels 3 --experiment-tag exp85_run158_resbase_watch_ctrl_3ch --seed 42
# Run 159
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition control --channels 3 --experiment-tag exp85_run159_resbase_all_ctrl_3ch --seed 42
# Run 160
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp85_run160_resbase_phone_unc_3ch --seed 42
# Run 161
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp85_run161_resbase_watch_unc_3ch --seed 42
# Run 162
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition uncontrolled --channels 3 --experiment-tag exp85_run162_resbase_all_unc_3ch --seed 42
# Run 163
python eval_custom_data.py --model resnet-baseline --mode loso --device-type phone --condition all --channels 3 --experiment-tag exp85_run163_resbase_phone_all_3ch --seed 42
# Run 164
python eval_custom_data.py --model resnet-baseline --mode loso --device-type watch --condition all --channels 3 --experiment-tag exp85_run164_resbase_watch_all_3ch --seed 42
# Run 165
python eval_custom_data.py --model resnet-baseline --mode loso --device-type all --condition all --channels 3 --experiment-tag exp85_run165_resbase_all_all_3ch --seed 42
```

> Notes:
> - Exp8.5 tags use `exp85_runXXX_*`, so summary filtering is direct with `--exp exp85`.
> - For `Imported Pretrained Weights = Yes`, commands explicitly pass `--pretrained`.
> - JSON output includes `experiment_table_metadata`, so the summarizer keeps your table columns stable.

---

### Experiment 8.6 -- Reverse Cross-Dataset: Train Custom Data, Test HHAR

> **Goal:** Reverse of Exp8 Mode A. Train all four model architectures on our custom-collected
> Xsens data, then evaluate on all HHAR subjects (no-bike, 5-class). This tests whether a model
> trained on our small custom dataset can generalize to the larger, more heterogeneous HHAR
> population — the symmetric counterpart to Exp8 Mode A which trained on HHAR and tested on custom.
>
> Uses `--mode reverse` added to `eval_custom_data.py`.
> Training uses all custom subjects (no LOSO) with 10% random validation split.
> Testing uses all HHAR subjects (no-bike, 5 activities) in a single evaluation.
>
> Summarize with: `python summarize_experiment_results.py --exp exp86`

| Architecture Type | Device Type | Testing Dataset | Imported Pretrained Weights | 3ch vs 6ch data | Data Fraction | Additional Pretraining dataset | Training Dataset | Testing Method | # Run |
|---|---|---|---|---|---|---|---|---|---|
| LIMU-BERT-X | Phone | HHAR | Yes | 6ch | 100% | None | Our Controlled | Cross | 166 |
| LIMU-BERT-X | Phone | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 167 |
| LIMU-BERT-X | Watch | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 168 |
| LIMU-BERT-X | All | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 169 |
| HART | Phone | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 170 |
| HART | Watch | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 171 |
| HART | All | HHAR | No | 6ch | 100% | None | Our Controlled | Cross | 172 |
| SSL-Wearables | Watch | HHAR | Yes | 3ch | 100% | None | Our Controlled | Cross | 173 |
| SSL-Wearables | Phone | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 174 |
| SSL-Wearables | Watch | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 175 |
| SSL-Wearables | All | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 176 |
| ResNet-Baseline | Phone | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 177 |
| ResNet-Baseline | Watch | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 178 |
| ResNet-Baseline | All | HHAR | No | 3ch | 100% | None | Our Controlled | Cross | 179 |
| LIMU-BERT-X | Phone | HHAR | Yes | 6ch | 100% | None | Our Uncontrolled | Cross | 180 |
| LIMU-BERT-X | Phone | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 181 |
| LIMU-BERT-X | Watch | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 182 |
| LIMU-BERT-X | All | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 183 |
| HART | Phone | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 184 |
| HART | Watch | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 185 |
| HART | All | HHAR | No | 6ch | 100% | None | Our Uncontrolled | Cross | 186 |
| SSL-Wearables | Watch | HHAR | Yes | 3ch | 100% | None | Our Uncontrolled | Cross | 187 |
| SSL-Wearables | Phone | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 188 |
| SSL-Wearables | Watch | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 189 |
| SSL-Wearables | All | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 190 |
| ResNet-Baseline | Phone | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 191 |
| ResNet-Baseline | Watch | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 192 |
| ResNet-Baseline | All | HHAR | No | 3ch | 100% | None | Our Uncontrolled | Cross | 193 |
| LIMU-BERT-X | Phone | HHAR | Yes | 6ch | 100% | None | Our All | Cross | 194 |
| LIMU-BERT-X | Phone | HHAR | No | 6ch | 100% | None | Our All | Cross | 195 |
| LIMU-BERT-X | Watch | HHAR | No | 6ch | 100% | None | Our All | Cross | 196 |
| LIMU-BERT-X | All | HHAR | No | 6ch | 100% | None | Our All | Cross | 197 |
| HART | Phone | HHAR | No | 6ch | 100% | None | Our All | Cross | 198 |
| HART | Watch | HHAR | No | 6ch | 100% | None | Our All | Cross | 199 |
| HART | All | HHAR | No | 6ch | 100% | None | Our All | Cross | 200 |
| SSL-Wearables | Watch | HHAR | Yes | 3ch | 100% | None | Our All | Cross | 201 |
| SSL-Wearables | Phone | HHAR | No | 3ch | 100% | None | Our All | Cross | 202 |
| SSL-Wearables | Watch | HHAR | No | 3ch | 100% | None | Our All | Cross | 203 |
| SSL-Wearables | All | HHAR | No | 3ch | 100% | None | Our All | Cross | 204 |
| ResNet-Baseline | Phone | HHAR | No | 3ch | 100% | None | Our All | Cross | 205 |
| ResNet-Baseline | Watch | HHAR | No | 3ch | 100% | None | Our All | Cross | 206 |
| ResNet-Baseline | All | HHAR | No | 3ch | 100% | None | Our All | Cross | 207 |

```powershell
# ============================================================
# Exp8.6: Reverse Cross-Dataset (Custom -> HHAR), Runs 166-207
# ============================================================

# --- Our Controlled (condition=control), Runs 166-179 ---

# --- LIMU-BERT-X ---
# Run 166: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode reverse --device-type phone --condition control --channels 6 --experiment-tag exp86_run166_limuX_phone_ctrl_preY --seed 42
# Run 167: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type phone --condition control --channels 6 --experiment-tag exp86_run167_limuX_phone_ctrl_preN --seed 42
# Run 168
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type watch --condition control --channels 6 --experiment-tag exp86_run168_limuX_watch_ctrl --seed 42
# Run 169
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type all --condition control --channels 6 --experiment-tag exp86_run169_limuX_all_ctrl --seed 42

# --- HART ---
# Run 170
python eval_custom_data.py --model hart --mode reverse --device-type phone --condition control --channels 6 --experiment-tag exp86_run170_hart_phone_ctrl --seed 42
# Run 171
python eval_custom_data.py --model hart --mode reverse --device-type watch --condition control --channels 6 --experiment-tag exp86_run171_hart_watch_ctrl --seed 42
# Run 172
python eval_custom_data.py --model hart --mode reverse --device-type all --condition control --channels 6 --experiment-tag exp86_run172_hart_all_ctrl --seed 42

# --- SSL-Wearables ---
# Run 173: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode reverse --device-type watch --condition control --channels 3 --experiment-tag exp86_run173_ssl_watch_ctrl_preY --seed 42
# Run 174
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type phone --condition control --channels 3 --experiment-tag exp86_run174_ssl_phone_ctrl --seed 42
# Run 175: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type watch --condition control --channels 3 --experiment-tag exp86_run175_ssl_watch_ctrl_preN --seed 42
# Run 176
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type all --condition control --channels 3 --experiment-tag exp86_run176_ssl_all_ctrl --seed 42

# --- ResNet-Baseline ---
# Run 177
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type phone --condition control --channels 3 --experiment-tag exp86_run177_resbase_phone_ctrl_3ch --seed 42
# Run 178
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type watch --condition control --channels 3 --experiment-tag exp86_run178_resbase_watch_ctrl_3ch --seed 42
# Run 179
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type all --condition control --channels 3 --experiment-tag exp86_run179_resbase_all_ctrl_3ch --seed 42

# --- Our Uncontrolled (condition=uncontrolled), Runs 180-193 ---

# --- LIMU-BERT-X ---
# Run 180: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode reverse --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp86_run180_limuX_phone_unc_preY --seed 42
# Run 181: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp86_run181_limuX_phone_unc_preN --seed 42
# Run 182
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp86_run182_limuX_watch_unc --seed 42
# Run 183
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type all --condition uncontrolled --channels 6 --experiment-tag exp86_run183_limuX_all_unc --seed 42

# --- HART ---
# Run 184
python eval_custom_data.py --model hart --mode reverse --device-type phone --condition uncontrolled --channels 6 --experiment-tag exp86_run184_hart_phone_unc --seed 42
# Run 185
python eval_custom_data.py --model hart --mode reverse --device-type watch --condition uncontrolled --channels 6 --experiment-tag exp86_run185_hart_watch_unc --seed 42
# Run 186
python eval_custom_data.py --model hart --mode reverse --device-type all --condition uncontrolled --channels 6 --experiment-tag exp86_run186_hart_all_unc --seed 42

# --- SSL-Wearables ---
# Run 187: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode reverse --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp86_run187_ssl_watch_unc_preY --seed 42
# Run 188
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp86_run188_ssl_phone_unc --seed 42
# Run 189: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp86_run189_ssl_watch_unc_preN --seed 42
# Run 190
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type all --condition uncontrolled --channels 3 --experiment-tag exp86_run190_ssl_all_unc --seed 42

# --- ResNet-Baseline ---
# Run 191
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type phone --condition uncontrolled --channels 3 --experiment-tag exp86_run191_resbase_phone_unc_3ch --seed 42
# Run 192
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type watch --condition uncontrolled --channels 3 --experiment-tag exp86_run192_resbase_watch_unc_3ch --seed 42
# Run 193
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type all --condition uncontrolled --channels 3 --experiment-tag exp86_run193_resbase_all_unc_3ch --seed 42

# --- Our All (condition=all), Runs 194-207 ---

# --- LIMU-BERT-X ---
# Run 194: LIMU-BERT-X phone, Imported Pretrained=Yes
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --pretrained weights/limu_bert_x --mode reverse --device-type phone --condition all --channels 6 --experiment-tag exp86_run194_limuX_phone_all_preY --seed 42
# Run 195: LIMU-BERT-X phone, Imported Pretrained=No
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type phone --condition all --channels 6 --experiment-tag exp86_run195_limuX_phone_all_preN --seed 42
# Run 196
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type watch --condition all --channels 6 --experiment-tag exp86_run196_limuX_watch_all --seed 42
# Run 197
python eval_custom_data.py --model limu-bert --limu-seq-len 20 --mode reverse --device-type all --condition all --channels 6 --experiment-tag exp86_run197_limuX_all_all --seed 42

# --- HART ---
# Run 198
python eval_custom_data.py --model hart --mode reverse --device-type phone --condition all --channels 6 --experiment-tag exp86_run198_hart_phone_all --seed 42
# Run 199
python eval_custom_data.py --model hart --mode reverse --device-type watch --condition all --channels 6 --experiment-tag exp86_run199_hart_watch_all --seed 42
# Run 200
python eval_custom_data.py --model hart --mode reverse --device-type all --condition all --channels 6 --experiment-tag exp86_run200_hart_all_all --seed 42

# --- SSL-Wearables ---
# Run 201: SSL watch, Imported Pretrained=Yes
python eval_custom_data.py --model ssl-wearables --pretrained model_check_point/mtl_best.mdl --mode reverse --device-type watch --condition all --channels 3 --experiment-tag exp86_run201_ssl_watch_all_preY --seed 42
# Run 202
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type phone --condition all --channels 3 --experiment-tag exp86_run202_ssl_phone_all --seed 42
# Run 203: SSL watch, Imported Pretrained=No
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type watch --condition all --channels 3 --experiment-tag exp86_run203_ssl_watch_all_preN --seed 42
# Run 204
python eval_custom_data.py --model ssl-wearables --mode reverse --device-type all --condition all --channels 3 --experiment-tag exp86_run204_ssl_all_all --seed 42

# --- ResNet-Baseline ---
# Run 205
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type phone --condition all --channels 3 --experiment-tag exp86_run205_resbase_phone_all_3ch --seed 42
# Run 206
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type watch --condition all --channels 3 --experiment-tag exp86_run206_resbase_watch_all_3ch --seed 42
# Run 207
python eval_custom_data.py --model resnet-baseline --mode reverse --device-type all --condition all --channels 3 --experiment-tag exp86_run207_resbase_all_all_3ch --seed 42
```

> Notes:
> - Exp8.6 tags use `exp86_runXXX_*`, so summary filtering is direct with `--exp exp86`.
> - All runs use `--mode reverse`: trains on custom data (all subjects, 10% val split), tests on all HHAR subjects (no-bike, 5-class).
> - LIMU-BERT-X and HART use `--channels 6`; SSL-Wearables and ResNet-Baseline use `--channels 3`.
> - `Imported Pretrained Weights = Yes` runs explicitly pass `--pretrained` (LIMU: `weights/limu_bert_x`, SSL: `model_check_point/mtl_best.mdl`).
> - No `--custom-test-repeats` needed: training is deterministic given `--seed 42`, and HHAR test set is fixed.
> - This is the symmetric counterpart to Exp8 Mode A (Runs 47-88). Compare Exp8.6 vs Exp8 Mode A to assess bidirectional cross-dataset transferability.
> - JSON output includes `experiment_table_metadata` with Training Dataset = Our Controlled/Uncontrolled/All and Testing Dataset = HHAR.

> Note: `pretrain_data/limu_phone_finetune.pt` and `pretrain_data/limu_phone_scratch.pt` must exist before running the phone-only LIMU additional-training commands.
## Quick Reference: Common Workflows

### "I just want to train HART on phone data"

```powershell
python prepare_hhar_data.py --hart --phone-only --no-bike
python train_loso.py --model hart --device-type phone --channels 6 --no-bike
```

### "I want results for all models on smartwatch"

```powershell
python prepare_hhar_data.py --watch-only --no-bike
python train_loso.py --model hart --device-type watch --channels 6 --no-bike
python train_loso.py --model limu-bert --device-type watch --no-bike
python train_loso.py --model ssl-wearables --device-type watch --channels 3 --no-bike
python train_loso.py --model resnet-baseline --device-type watch --channels 6 --no-bike
```

### "I want a basic CNN baseline to compare against ssl-wearables"

```powershell
python prepare_hhar_data.py --ssl --no-bike
python train_loso.py --model resnet-baseline --device-type all --channels 6 --no-bike --experiment-tag exp0o_resbase_all_6ch
```

### "I want to run EXP5 cross-dataset for one model"

```powershell
python prepare_wisdm_data.py
python eval_cross_dataset.py --model hart --device-type phone --channels 6 --experiment-tag exp5_hart_phone
```

### "I want to run all 5 experiments end-to-end on GPU 0"

```powershell
.\run_experiments.ps1 -NoBike -GPU "0"
```

### "I want to quickly test with fewer epochs"

```powershell
.\run_experiments.ps1 -Step exp1 -Epochs 10 -NoBike
```

---

## Output

All results are saved to `loso_results/` as JSON files containing:

- Per-fold accuracy, F1 (weighted), F1 (macro)
- Mean and standard deviation across folds
- Metadata: model, device type, channels, data fraction, no_bike, num_classes
- Experiment tag for easy identification

File naming pattern: `{experiment_tag}_{model}_{scratch|pretrained}_loso_{timestamp}.json`

EXP5 cross-dataset results: `{experiment_tag}_{model}_cross_wisdm_{timestamp}.json`

---

## Prerequisites

### Required Raw Data

Place the HHAR dataset at:

```
heterogeneity+activity+recognition/
  Activity recognition exp/
    Activity recognition exp/
      Phones_accelerometer.csv
      Phones_gyroscope.csv
      Watch_accelerometer.csv
      Watch_gyroscope.csv
```

For **Experiments 2/3** (SSL pretraining) and **Experiment 5** (cross-dataset), place these external datasets:

```
pamap2+physical+activity+monitoring/           # Exp 2 (watch SSL pretrain)
  PAMAP2_Dataset/PAMAP2_Dataset/
    Protocol/   (subject101.dat ... subject109.dat)
    Optional/   (subject101.dat, subject105.dat, ...)

SBHAR/                                          # Exp 3 (phone SSL pretrain)
  RawData/
    acc_expXX_userYY.txt
    gyro_expXX_userYY.txt
    labels.txt

wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/   # Exp 2/3/5
  wisdm-dataset/wisdm-dataset/
    raw/
      phone/accel/   (data_{id}_accel_phone.txt)
      phone/gyro/    (data_{id}_gyro_phone.txt)
      watch/accel/   (data_{id}_accel_watch.txt)
      watch/gyro/    (data_{id}_gyro_watch.txt)
```

### Required Model Code

```
code/
  Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices-main/   (HART)
  LIMU-BERT_Experience-main/                                        (LIMU-BERT)
  ssl-wearables-main/                                               (ssl-wearables)
  resnet-baseline/                                                   (ResNet-Baseline, auto-created)
```

### Optional Pretrained Weights

| Model | Weight Location | Source |
|---|---|---|
| LIMU-BERT | `code/LIMU-BERT_Experience-main/.../weights/limu_bert_x.pt` | Original LIMU-BERT release |
| HARNet | `code/ssl-wearables-main/.../model_check_point/mtl_best.mdl` | UK Biobank self-supervised pretraining |

### Python Dependencies

- numpy, pandas, scipy, scikit-learn
- TensorFlow (for HART)
- PyTorch (for LIMU-BERT, ssl-wearables, and ResNet-Baseline)
- hickle (for HART data format)
