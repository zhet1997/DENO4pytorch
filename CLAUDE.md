# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository-specific instructions

- Reply to the user in Chinese unless they clearly ask for another language. This comes from `.cursor/rules/mode-change.mdc`.
- The repo has a strict user-driven mode workflow in `.cursor/rules/mode-change.mdc`: research / ideation / planning / execution / review. Do not switch modes unless the user explicitly asks. In research mode, only inspect and ask clarifying questions; do not propose implementation.
- Follow the research-code rules in `.cursor/rules/research-code.mdc`: prefer simple, readable, debuggable code; avoid broad compatibility shims; keep branching shallow; let errors surface unless there is a concrete batch/external-call reason to catch them.
- For one-off paper plotting scripts, follow `.cursor/rules/paper-plot.mdc`: keep all tunable plotting parameters in a single top-of-file config section, avoid argparse, and optimize for fast manual iteration rather than reuse.
- For `.ipynb` work, follow `.cursor/rules/ipynb.mdc`: keep notebook and Python versions in sync and use the repository’s notebook-template workflow instead of editing notebook JSON by hand.

## Current project focus

- This repository is being used for an AI4S research project.
- The highest-priority active area is `Demo/satellite_sup_2d/`.
- Before changing satellite research code, read `.specstory/history/2026-03-23_11-23-09Z-模型改进研究分析.md` for prior analysis context.
- When a user asks about “current research”, “ongoing experiments”, “recent satellite work”, or model-improvement direction, start from `Demo/satellite_sup_2d/` rather than older baseline directories.

## Environment and dependencies

- Install dependencies with:
  - `pip install -r requirements.txt`
- Core runtime stack from `requirements.txt`: PyTorch, NumPy/SciPy, scikit-learn, h5py, pandas, matplotlib/seaborn, PyYAML, OmegaConf, Pillow, tqdm.
- There is no discovered top-level build system, package entrypoint, Makefile, or repo-wide lint setup. This repository is script-driven.

## Common commands

Run commands from the repository root unless a script explicitly `cd`s into its own directory.

### Active satellite_sup_2d research workflows

- K-bucket superposition ablation entrypoint:
  - `python Demo/satellite_sup_2d/train_entry.py --work_name test_kmax10_S01 --train_component_nums 1,2,3,4,5,6,7,8,9,10 --super_train_mode S01 --epochs 200`
- Cyclic supervised/consistency/distillation training:
  - `python Demo/satellite_sup_2d/train_distillation_loop.py --max_rounds 10 --supervised_epochs 50 --distill_epochs 20`
- DualHead Fourier Transformer training with YAML config:
  - `python Demo/satellite_sup_2d/train_DualHead_with_config.py --config DualHead_GUT_2d --h5 /data/wqn/datasets/SDNO_test/15c_data_test.h5 --ntrain 4000 --nvalid 1000 --batch_size 16 --epochs 200 --lr 1e-3 --scheduler_step 100 --scheduler_gamma 0.5`
- Quick checks for the active satellite loader stack:
  - `python Demo/satellite_sup_2d/test_GUT_loader.py`
  - `python Demo/satellite_sup_2d/test_multi_gut_loader.py`

### Satellite baseline training

Representative single-run commands are in `Demo/satellite_2d_base/`:

- FNO:
  - `python Demo/satellite_2d_base/run_FNO_satellite.py --data_path /data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5 --ntrain 8000 --nvalid 100 --batch_size 32 --epochs 1000 --lr 1e-4 --cuda_index 6`
- Transformer:
  - `python Demo/satellite_2d_base/run_Trans_satellite.py --data_path /data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5 --ntrain 8000 --nvalid 100 --batch_size 32 --epochs 1000 --lr 5e-4 --down 1 --use_config --cuda_index 6`
- MLP:
  - `python Demo/satellite_2d_base/run_MLP_satellite.py --data_path /data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5 --ntrain 8000 --nvalid 100 --batch_size 32 --epochs 1000 --lr 1e-3 --down 4 --hidden 1024 --layers 4 --cuda_index 6`
- UNet:
  - `python Demo/satellite_2d_base/run_UNet_satellite.py --data_path /data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5 --ntrain 8000 --nvalid 100 --batch_size 32 --epochs 1000 --lr 1e-3 --cuda_index 6`

### Satellite experiment batches

- Serial MLP/FNO sweep over multiple training-set sizes:
  - `bash Demo/satellite_2d_base/run_experiments.sh`
- Full benchmark sweep over multiple datasets and four models:
  - `bash Demo/satellite_2d_base/run_satellite_benchmark.sh`
- Additional batch scripts exist in the same directory (`run_all_satellite_baselines.sh`, `run_satellite_benchmark2.sh`). Read them before reuse because they hard-code dataset paths and GPU indices.

### Running tests

This repo does not expose a single centralized test harness. Tests are mostly standalone scripts.

- Run one script-style test directly, for example:
  - `python Demo/satellite_sup_2d/test_GUT_loader.py`
  - `python Demo/satellite_sup_2d/test_multi_gut_loader.py`
  - `python Demo/GVRB_2d/test/self_supervise_test.py`
  - `python Tools/post_process/test/cfdPost_test.py`
- If you want to run a “single test”, first inspect the target file: many of these are executable scripts rather than pytest-style test cases.

## Big-picture architecture

The repository is organized as a reusable model library plus many experiment-specific entry scripts.

### 1. `Models/` is the reusable model layer

This directory holds the neural operator / surrogate model implementations used across experiments:

- `Models/fno/`: Fourier Neural Operator variants.
- `Models/transformer/`: transformer-based operator models, including the DualHead transformer family.
- `Models/cnn/`: CNN and UNet-style baselines.
- `Models/don/`, `Models/gnn/`, `Models/pinn/`, `Models/basic/`: other model families and shared low-level layers.

When the user asks about “the model implementation”, start here.

### 2. `Demo/` is the application and experiment layer

Most real work happens in dataset/task-specific scripts under `Demo/`. These are the actual training/evaluation entrypoints, not thin wrappers.

Important families:

- `Demo/satellite_sup_2d/`: the active AI4S research area. It contains the current supervised satellite workflows: K-bucket superposition ablations, G/U/T data loading, dynamic U-channel compression, consistency training, pseudo-label generation, cyclic distillation, and DualHead transformer experiments.
- `Demo/satellite_2d_base/`: baseline benchmark area for the satellite thermal dataset. Use this mainly for comparisons or older single-model baselines.
- Other subdirectories such as `Rotor37_2d/`, `PakB_2d/`, `GVRB_2d/`, `TwoLPT_2d/`, `HPT_2d/` are other domain-specific experiment families with their own training conventions.

When the user asks “how do I train/evaluate X?” for current satellite work, check `Demo/satellite_sup_2d/` first.

### 3. `satellite_sup_2d/` internal research structure

The active research code in `Demo/satellite_sup_2d/` is split by responsibility:

- `train_entry.py`: main entry for the K-bucket superposition ablation experiments. It trains across `super_num` settings such as `S0`, `S01`, and `S012` and evaluates across multiple K buckets.
- `trainer_satellite.py`: shared training/evaluation utilities for the ablation framework. Key logic includes dynamic `U` channel compression, multi-`super_num` training, per-K validation, sample inference, and metrics logging.
- `data_loader_satellite.py`: current dataset-loading path for bucketed satellite data.
- `train_distillation_loop.py`: orchestrates the current cyclic research pipeline: supervised phase → consistency phase → teacher eligibility evaluation → pseudo-label shard generation → distillation phase → stopping logic.
- `consistency_modules.py`, `evaluation_modules.py`, `pseudo_label_modules.py`, `distill_modules.py`: modular pieces for the distillation framework.
- `ablation_satellite.py` and `trains_satellite.py`: model settings plus older/shared training wrappers used by several entry scripts.

### 4. `Utilizes/` and `Tools/` are the shared workflow layer

- `Utilizes/` contains shared utilities used widely across the repo: normalization, metrics/losses, processing, visualization, and optimization helpers.
- `Utilizes/process_data.py` is especially important because `DataNormer` is a common normalization primitive used by many training scripts.
- `Tools/` contains higher-level training/post-processing/optimization utilities. Expect older infrastructure here that coexists with newer task-specific code in `Demo/`.

### 5. `configs/` stores model YAML configuration

Transformer-family experiments often rely on YAML from `configs/`, especially satellite transformer configs such as:

- `configs/transformer_config_sate.yml`
- `configs/dualhead_transformer_config_sate.yml`

Check these before changing model dimensions or architectural hyperparameters.

## Architectural patterns that matter

### Script-first execution, not package-first execution

Most workflows are direct Python scripts. There is no single app entrypoint. The normal way to work is to run a task-specific file under `Demo/...` with CLI arguments.

### Many scripts inject project paths into `sys.path`

Several runners manually add the project root and `Models/` to `sys.path` before importing modules. Example patterns: `Demo/satellite_2d_base/run_FNO_satellite.py`, `Demo/satellite_sup_2d/train_entry.py`, and `Demo/satellite_sup_2d/train_distillation_loop.py`.

Implications:
- execution context matters;
- scripts are intended to be run directly;
- if imports fail, check how the script constructs or hard-codes `PROJECT_ROOT` before refactoring imports.

### The current satellite research uses a predictor + superposition decomposition

A recurring pattern in `Demo/satellite_sup_2d/` is:

- predictor network: `DualHeadFourierTransformer`
- superposition network: `FNO2d`
- wrapper/composition model: `supredictor_list_windows(...)`

This stack is used to study how performance changes when training and evaluating with different superposition depths (`super_num`) and different component-count buckets (`K`).

### Dynamic U-channel scaling is a core experiment axis

`trainer_satellite.py` compresses `U` channels dynamically based on `super_num`, using grouped min-reduction when the current channel count exceeds the target count. When debugging current experiments, treat channel-shape semantics as part of the research logic, not as incidental preprocessing.

### The satellite supervised stack uses G/U/T semantics

In `Demo/satellite_sup_2d/`, inputs and targets are consistently organized as:

- `G`: condition field / geometry / positional channels
- `U`: source field channels
- `T`: target temperature field

The DualHead transformer consumes `model(G, U)` rather than concatenated single-tensor input. If a user wants to adapt older FourierTransformer code, first verify whether the code path expects separated `G_dim` / `U_dim` or a single `node_feats` tensor.

### The distillation framework is round-based, not a one-shot fine-tune

`train_distillation_loop.py` implements an iterative loop with explicit stages:

1. supervised training;
2. consistency training;
3. anchor/evaluator-based teacher qualification;
4. pseudo-label shard generation;
5. distillation training;
6. shard cleanup and stop control.

If the user asks to modify “the current distillation research”, inspect the loop controller and the module boundaries before editing details inside one phase.

### Work directories are part of the contract

Training scripts typically create output directories like `runs/...`, `runs_bc/...`, or `work_satellite/...` and write:

- config snapshots
- train logs / metrics JSONL
- checkpoints such as `ckpt_best.pth`, `ckpt_last.pth`, `model_best.pth`
- loss curves and sample prediction figures

When debugging an experiment, inspect the work directory alongside the code.

## Files worth reading first for current satellite research

- `.specstory/history/2026-03-23_11-23-09Z-模型改进研究分析.md`
- `Demo/satellite_sup_2d/train_entry.py`
- `Demo/satellite_sup_2d/trainer_satellite.py`
- `Demo/satellite_sup_2d/data_loader_satellite.py`
- `Demo/satellite_sup_2d/train_distillation_loop.py`
- `Demo/satellite_sup_2d/consistency_modules.py`
- `Demo/satellite_sup_2d/evaluation_modules.py`
- `Demo/satellite_sup_2d/pseudo_label_modules.py`
- `Demo/satellite_sup_2d/distill_modules.py`
- `Demo/satellite_sup_2d/ablation_satellite.py`
- `Demo/satellite_sup_2d/trains_satellite.py`
- `Models/transformer/DualHeadTransformer_README.md`
- `Utilizes/process_data.py`

## Things not currently present

Do not assume the repository has these until you verify them in the target subproject:

- a repo-wide `pytest` suite
- a lint/format command such as `ruff`, `flake8`, or `black`
- a top-level build/package command
- one unified training abstraction shared by every experiment family
