# ch_learn: Discovering Free Energy Functionals via Differentiable Phase-Field Models

A physics-informed machine learning framework that discovers the free energy
functional of a Cahn-Hilliard phase-field model directly from spatiotemporal
field data, by embedding a neural network inside a **differentiable** Firedrake
finite element solver.

The neural network replaces the analytical free energy derivative *df/dc*. It is
trained end-to-end by running the PDE solver forward, comparing the simulated
concentration field against target ("ground-truth") snapshots, and propagating
exact gradients **backward through every PDE solve** using `firedrake-adjoint`.

---

## Table of Contents

1. [Method Overview](#method-overview)
2. [Branch Overview](#branch-overview)
3. [Repository Structure](#repository-structure)
4. [How the Pieces Fit Together](#how-the-pieces-fit-together)
5. [Dependencies and Installation](#dependencies-and-installation)
6. [Quickstart](#quickstart)
7. [Detailed Usage](#detailed-usage)
8. [Command-line Arguments](#command-line-arguments)
9. [Physics and Governing Equations](#physics-and-governing-equations)
10. [Neural Network Architectures](#neural-network-architectures)
11. [Loss Functions](#loss-functions)
12. [File Formats and Outputs](#file-formats-and-outputs)
13. [Experiment Tracking with WandB](#experiment-tracking-with-wandb)

---

## Method Overview

The framework couples three components:

1. **A Firedrake FEM solver** for the Cahn-Hilliard (CH) equation — a nonlinear
   PDE governing the evolution of a conserved concentration field *c*.

2. **A neural network (MLP)** that replaces the analytical free energy
   derivative *df/dc*. The network takes the concentration *c* as input and
   predicts *df/dc(c)*.

3. **`firedrake-adjoint`** for exact gradient propagation backward through the
   entire PDE time-integration loop. This lets the NN weights be trained
   end-to-end by comparing simulated fields against target data.

**Training loop (per epoch):**

1. The FE solver advances the CH system through a full simulation (all
   timesteps) using the current NN prediction for *df/dc*.
2. A loss compares the simulated concentration field against the target
   snapshot at each timestep.
3. `firedrake-adjoint` computes adjoint sensitivities *dJ/d(df/dc)* through all
   PDE solves.
4. The network is re-evaluated at the recorded states to rebuild a PyTorch
   computation graph; the adjoint sensitivities are injected as upstream
   gradients and backpropagated to the NN weights.
5. The Adam optimizer updates the weights; the LR scheduler adjusts the step.
6. A checkpoint and post-processing archive are saved periodically.

The free energy *f(c)* itself is reconstructed *post hoc* by numerically
integrating the learned derivative.

---

## Branch Overview

All branches share the same idea — learn a free energy functional through a
differentiable CH solver — but differ in spatial dimension, loss formulation,
and code organization. **`master` is the legacy baseline; the three feature
branches are the refactored, current codebase.**

| Branch | Mesh / Dimension | Loss | Notable | Status |
|---|---|---|---|---|
| `master` | 1D `IntervalMesh` | FFT | Pre-refactor monolithic scripts | Legacy baseline |
| `hpc-run` | 1D `IntervalMesh(200, 2)` | FFT (with mode truncation) | Includes `ch_ac.py` Allen-Cahn solver; extra datasets | HPC reference |
| `hpc/mse-loss` | 1D `IntervalMesh(200, 2)` | **MSE** (real-space) | Loss variant of `hpc-run` | Loss-comparison branch |
| `3d-adaptation` | 3D (`BoxMesh` / checkpoint meshes) | FFT | BDF2 time integration; checkpoint-based meshes | **Active / experimental** |

### `master` — legacy baseline

The original, pre-refactor implementation. Training logic lives in single
monolithic scripts — `ch_learn.py`, `ch_learn_energy.py`,
`network_fit_test.py`, `energy_fit_test.py` — rather than the modular layout
used by the feature branches. There is no `models/` package, no
`training_utils.py`, and no `learn_dfdc.py`. Kept for reference only; new work
happens on the feature branches.

### `hpc-run` — 1D HPC reference

The stable refactored 1D implementation, organized for cluster runs.

- **Mesh:** `IntervalMesh(200, 2)` — 201 CG1 points on the interval `[0, 2]`,
  kept aligned with the generated `ch_fh_<index>` reference datasets.
- **Loss:** FFT-based, with optional truncation of high-frequency modes via
  `--truncation-modes`.
- **Extra solver:** ships `ch_ac.py`, a coupled Cahn-Hilliard / Allen-Cahn
  reference solver (concentration *c* + crystalline order parameter *η*), in
  addition to the `ch_fh.py` Flory-Huggins CH solver.
- Contains additional bundled ground-truth simulation datasets.

### `hpc/mse-loss` — real-space loss variant

Identical in structure to `hpc-run` but swaps the FFT data-fidelity loss for a
plain **mean-squared-error** loss computed directly on the concentration DOFs.
Use this branch to compare spectral vs. pointwise objectives on the same 1D
problem.

### `3d-adaptation` — 3D extension (current branch, experimental)

Extends the solver and learning loop beyond 1D.

- **Reference solver (`ch_fh.py`):** rewritten for 3D. Loads its mesh and
  initial condition from a Firedrake `CheckpointFile` (`check_128.h5`), uses a
  **BDF2** time-integration scheme with a Backward-Euler startup step, and an
  Additive-Schwarz / ILU monolithic solver. Writes results to
  `simulation_128.h5`. This script is a work in progress.
- **Learning script (`learn_dfdc.py`):** builds its mesh from the target data —
  `UnitSquareMesh`/`UnitCubeMesh` sized from VTI grid dimensions, with a
  `UnitCubeMesh(100, 100, 100)` fallback for VTU input.
- **Loss:** FFT-based (`torch.fft.fft`).
- **Data loading:** VTU/VTI files are mapped onto Firedrake DOFs via a
  `scipy.spatial.KDTree` nearest-neighbour search, after normalizing point
  coordinates to the unit domain — so MD outputs with physical box coordinates
  can be compared against nondimensional CH fields.

### Choosing a branch

```
1D validation, spectral loss, HPC runs        -> hpc-run
1D, comparing real-space vs spectral loss      -> hpc/mse-loss
2D / 3D geometry, preparing for MD coupling    -> 3d-adaptation
Looking at the original implementation         -> master
```

---

## Repository Structure

```
ch_learn/
│
├── ch_fh.py              # Standalone CH / Flory-Huggins reference solver (generates target data)
├── ch_ac.py              # Coupled Cahn-Hilliard / Allen-Cahn solver        (hpc-run branch only)
│
├── simulation.py         # CHSolver class (reusable solver) + load_target_data()
├── learn_dfdc.py         # Main training script — learns df/dc
├── learn_f.py            # Alternate training script — learns f(c) directly
├── training_utils.py     # Argument parsing, device setup, optimizer/scheduler/wandb init
├── checkpoint.py         # Save / load model + optimizer + scheduler checkpoints
├── plotting.py           # Matplotlib / Plotly visualization helpers
├── reproduce_plots.py    # Post-processing: regenerate plots from the .npz archive
├── md_params.py          # Parse LAMMPS/XYZ MD metadata; auto-estimate CH timestep
├── read_npz_data.py      # Small utility to inspect an .npz archive's keys/shapes
│
├── models/
│   ├── __init__.py
│   ├── dfdc.py           # FEDerivative — MLP that predicts df/dc
│   └── energy.py         # FEnergy     — MLP that predicts f(c)
│
├── ch_fh/                # Reference simulation data (VTU snapshots)   [gitignored]
├── ch_learn_adjoint/     # Training-run VTU output                     [gitignored]
├── wandb/                # Weights & Biases run logs                   [gitignored]
│
├── ch_learn_model.pth         # Latest model checkpoint                 [gitignored]
├── post_processing_data.npz   # Archived training history + NN surfaces [gitignored]
└── reproduced_plots/          # Interactive HTML plots                  [gitignored]
```

> **Note:** simulation data, checkpoints, plots, `.npz` archives, and `wandb/`
> are all excluded by `.gitignore` — only source code is tracked.

---

## How the Pieces Fit Together

```
ch_fh.py  ──>  ch_fh/*.vtu          (reference / ground-truth data)
                    │
                    ▼
            simulation.load_target_data()   ─── KDTree maps VTU points → Firedrake DOFs
                    │
                    ▼
   learn_dfdc.py  ──┐
                    ├─ models/dfdc.py        FEDerivative MLP  (predicts df/dc)
                    ├─ simulation.py         CHSolver          (one reusable CH solve)
                    ├─ training_utils.py     argparse, device, optimizer, scheduler, wandb
                    ├─ checkpoint.py         save/load training state
                    └─ plotting.py           loss curves
                    │
                    ▼
   ch_learn_model.pth          (checkpoint — resume training)
   post_processing_data.npz    (full training history + NN-output surfaces)
                    │
                    ▼
   reproduce_plots.py  ──>  reproduced_plots/*.html   (interactive Plotly figures)
```

- **`learn_dfdc.py`** is the main entry point. `setup_problem()` builds the mesh
  and loads target data; `train_epoch()` runs one full forward simulation +
  adjoint gradient + PyTorch backprop; `main()` drives the epoch loop,
  checkpointing, and `.npz` saving.
- **`learn_f.py`** is a sibling script that learns the free energy *f(c)*
  directly (via the `FEnergy` model) instead of its derivative.
- **`simulation.py`**'s `CHSolver` builds the weak form and solver **once** and
  reuses them every timestep — the learned `df/dc` enters as a coefficient
  `Function` whose data is reassigned each step, avoiding UFL recompilation.
- **`training_utils.py`** centralizes all run configuration: CLI parsing, device
  selection, optimizer/scheduler construction, checkpoint resume, and wandb init.

---

## Dependencies and Installation

This project runs inside the **Firedrake** virtual environment. Firedrake must
be installed separately following the
[official instructions](https://www.firedrakeproject.org/download.html). All
other packages are installed into the same environment.

| Library | Role |
|---|---|
| [Firedrake](https://www.firedrakeproject.org/) | FEM solver, mesh I/O, MPI |
| `firedrake.adjoint` | Adjoint AD through the PDE time loop |
| [PyTorch](https://pytorch.org/) | Neural network, autodiff, optimizer |
| [NumPy](https://numpy.org/) | Arrays, FFT |
| [SciPy](https://scipy.org/) | `KDTree` for coordinate mapping |
| [PyVista](https://pyvista.org/) | Reading VTU/VTI files into NumPy arrays |
| [Matplotlib](https://matplotlib.org/) | Static plots (loss curves) |
| [Plotly](https://plotly.com/python/) | Interactive HTML plots |
| [WandB](https://wandb.ai/) | Experiment tracking (optional) |
| [gmsh](https://gmsh.info/) | Mesh generation (imported by `ch_fh.py`) |

```bash
source /path/to/firedrake/bin/activate
pip install torch numpy scipy matplotlib plotly pyvista wandb gmsh
```

---

## Quickstart

```bash
# Activate the Firedrake environment
source /path/to/firedrake/bin/activate

# 1. Generate reference (ground-truth) data
python ch_fh.py

# 2. Train the model (resumes from checkpoint if one exists)
python learn_dfdc.py --data-dir ch_fh

# 3. Regenerate post-processing plots from the saved .npz archive
python reproduce_plots.py
```

---

## Detailed Usage

### Step 1 — Generate reference data

`ch_fh.py` runs a standalone Cahn-Hilliard / Flory-Huggins solve and writes
field snapshots that act as the synthetic "ground truth" the NN learns from.

```bash
python ch_fh.py
```

On the 1D branches it writes VTU snapshots to a `ch_fh*/` directory plus a
`.pvd` ParaView collection. On `3d-adaptation` it instead reads its mesh and
initial condition from a `CheckpointFile` and writes a Firedrake checkpoint
(`simulation_128.h5`).

### Step 2 — Train the model

`learn_dfdc.py` is the main training entry point.

```bash
# Default run (resumes from checkpoint if ch_learn_model.pth exists)
python learn_dfdc.py

# Fresh run, ignoring any checkpoint
python learn_dfdc.py --no-resume

# Custom hyperparameters
python learn_dfdc.py \
    --data-dir ch_fh \
    --epochs 5000 \
    --learning-rate 1e-3 \
    --scheduler cosine \
    --warmup-epochs 100 \
    --seed 12

# Disable WandB
python learn_dfdc.py --no-wandb

# Quick debug run (2 epochs only)
python learn_dfdc.py --profile
```

To learn the free energy *f(c)* itself rather than its derivative, run
`learn_f.py` instead (same CLI surface, uses the `FEnergy` model).

### Step 3 — Reproduce plots

After training, `reproduce_plots.py` reads `post_processing_data.npz` and
regenerates interactive Plotly HTML files into `reproduced_plots/`.

```bash
python reproduce_plots.py
```

To inspect the contents of any `.npz` archive:

```bash
python read_npz_data.py post_processing_data.npz
```

---

## Command-line Arguments

All arguments below are accepted by `learn_dfdc.py` (and `learn_f.py`). Run
`python learn_dfdc.py --help` for the authoritative list. Defaults shown are
for the `3d-adaptation` branch.

### Training

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `ch_fh` | Directory with target data files (`.vtu` or `.vti`) |
| `--epochs` | 5000 | Number of training epochs |
| `--learning-rate` | 1e-3 | Initial learning rate |
| `--resume-lr` | — | Override LR when resuming (keeps Adam momentum state) |
| `--seed` | 12 | Random seed for reproducibility |
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `plateau`, or `none` |
| `--warmup-epochs` | 100 | Linear LR warm-up epochs (cosine scheduler) |
| `--patience` | 100 | Patience for `ReduceLROnPlateau` |
| `--factor` | 0.8 | LR reduction factor for `ReduceLROnPlateau` |
| `--no-resume` | — | Ignore existing checkpoint, start fresh |
| `--no-wandb` | — | Disable Weights & Biases logging |
| `--profile` | — | Debug mode — runs only 2 epochs |
| `--output-dir` | — | Output directory (else `$OUTPUT_DIR`, else `.`) |
| `--cpu` | — | Force CPU even if CUDA is available |

### Physics / simulation parameters

| Argument | Default | Description |
|---|---|---|
| `--chi` | 1.0 | Flory-Huggins interaction parameter χ |
| `--N1` | 5.0 | Degree of polymerization, species 1 |
| `--N2` | 5.0 | Degree of polymerization, species 2 |
| `--M` | 1.0 | Cahn-Hilliard mobility coefficient |
| `--dt` | auto | Time step; if omitted, auto-estimated from MD metadata |
| `--zeta` | 1.0 | Monomer friction coefficient (for dt auto-estimation) |
| `--dump-interval` | — | MD dump interval, in MD steps (enables dt auto-estimation) |

> **Timestep selection:** if `--dt` is given it is used directly. Otherwise, if
> `--dump-interval` is given, `md_params.auto_estimate_dt()` parses LAMMPS
> `*.in` and `*.xyz` files in the data directory and computes a nondimensional
> CH timestep `dt = (dump_interval · dt_md) / (N_avg · ζ · L²)`. If neither is
> available, it falls back to `dt = 1e-3`.

> The `hpc-run` branch additionally exposes `--truncation-modes` (number of
> high-frequency FFT modes to discard from the loss).

---

## Physics and Governing Equations

### Cahn-Hilliard equation (conserved dynamics)

The concentration field *c* evolves as a conserved phase-field:

```
∂c/∂t = ∇·(M ∇μ)
μ      = df/dc − λ² ∇²c
```

where *μ* is the chemical potential, *M* the mobility, and *λ* the
gradient-energy (interface width) coefficient.

### Free energy functional (Flory-Huggins)

The bulk free energy used to generate reference data:

```
f(c) = c·ln(c)/N1 + (1−c)·ln(1−c)/N2 + χ·c·(1−c)
```

The neural network learns **`df/dc`** (or, in `learn_f.py`, `f` itself); the
energy surface is reconstructed afterward by numerical integration.

### Finite element discretization

- **Function spaces:** a mixed `CG1 × CG1` space holds *(c, μ)*, solved together
  as one nonlinear system each step.
- **Time integration:** semi-implicit on the 1D branches; BDF2 with a
  Backward-Euler startup on `3d-adaptation`.
- **Linear solver:** direct LU via MUMPS (1D / `CHSolver`); Additive-Schwarz +
  ILU for the 3D monolithic solve.

---

## Neural Network Architectures

Two small MLPs live in [models/](models/):

### `FEDerivative` — [models/dfdc.py](models/dfdc.py)

Predicts the free energy derivative *df/dc*. Used by `learn_dfdc.py`.

```
Input:  c                       (1 feature)
        Linear(1 → 50) + Tanh
        Linear(50 → 20) + Tanh
        Linear(20 → 1)
Output: df/dc                    (mean-subtracted across the batch)
```

**Zero-mean constraint:** the CH equation only depends on *∇μ*, so adding a
constant to *df/dc* is physically unobservable. The output has its batch mean
removed to fix this gauge freedom and stabilize training.

### `FEnergy` — [models/energy.py](models/energy.py)

Predicts the free energy *f(c)* directly. Used by `learn_f.py`.

```
Input:  c                            (1 feature)
        Linear(1 → 50) + LeakyReLU
        Linear(50 → 50) + LeakyReLU
        Linear(50 → 1)
Output: f(c)
```

Both models are trained in `float64` to match Firedrake's double-precision
vectors and avoid repeated dtype promotion when exchanging arrays.

---

## Loss Functions

The data-fidelity loss compares the simulated concentration field against the
target snapshot at every timestep; the per-timestep losses are summed.

### FFT loss (`learn_dfdc.py` on `hpc-run` and `3d-adaptation`)

```
L = 0.5 · mean( |FFT(c_sim) − FFT(c_target)|² )
```

Comparing spectra rather than raw DOF values makes the objective sensitive to
phase-separated morphology and characteristic length scales across all modes.
On `hpc-run`, high-frequency modes can optionally be truncated via
`--truncation-modes`.

### MSE loss (`hpc/mse-loss` branch)

```
L = mean( (c_sim − c_target)² )
```

A pointwise, real-space objective — the alternative being compared against the
spectral loss.

### Gradient flow

PyTorch computes `dL/dc` on the simulated field. That gradient is injected into
Firedrake as a coefficient `Function`; `firedrake-adjoint`'s
`ReducedFunctional.derivative()` propagates it through all PDE solves to obtain
`dL/d(df/dc)` at each timestep. The network is then re-evaluated at the recorded
states and these adjoint sensitivities are used as upstream gradients for
PyTorch's `.backward()`, updating the NN weights via Adam.

---

## File Formats and Outputs

### Reference simulation data (`ch_fh/`, etc.)

- **Format:** VTK Unstructured Grid (`.vtu`) or Image Data (`.vti`).
- **Field:** `Volume Fraction` — the concentration *c*.
- **Loaded by:** `simulation.load_target_data()`, which maps VTK points onto
  Firedrake DOFs with a `scipy.spatial.KDTree` nearest-neighbour search after
  normalizing coordinates to the unit domain.

### Model checkpoint (`ch_learn_model.pth`)

A PyTorch checkpoint containing `model_state_dict`, `optimizer_state_dict`,
`scheduler_state_dict`, the loss history (`epoch_losses`, `epoch_numbers`), and
the last completed `epoch` — enough to resume optimization, not just inference.

### Post-processing archive (`post_processing_data.npz`)

A compressed NumPy archive written periodically during training. Key contents:

| Key | Description |
|---|---|
| `preds_collection` | Final-timestep simulated field, per saved epoch |
| `epochs_collection` | Corresponding epoch indices |
| `target_final_global` | Reference *c* at the final timestep |
| `all_epochs_comparison_data` | Per-epoch, timestep-by-timestep comparisons |
| `all_nn_outputs` | NN `df/dc` evaluated on a 1D `c` grid, per epoch |
| `c_values_nn` | The `c` grid coordinates |
| `epoch_losses`, `epoch_numbers` | Loss history |

### Generated outputs

| File / Directory | Generated by | Description |
|---|---|---|
| `ch_fh/`, `ch_fh*.pvd` | `ch_fh.py` | Reference simulation snapshots |
| `ch_learn_adjoint/`, `*.pvd` | `learn_dfdc.py` | VTU output from the final training epoch |
| `ch_learn_model.pth` | `learn_dfdc.py` | Latest model checkpoint |
| `post_processing_data.npz` | `learn_dfdc.py` | Training history + NN-output surfaces |
| `lve_dfdc.png` | `learn_dfdc.py` | Loss-vs-epoch plot (log scale) |
| `reproduced_plots/*.html` | `reproduce_plots.py` | Interactive Plotly visualizations |
| `wandb/` | WandB SDK | Experiment logs (gitignored) |

---

## Experiment Tracking with WandB

Training metrics and hyperparameters are logged to
[Weights & Biases](https://wandb.ai) (project `ch_learn`) by default.

```bash
pip install wandb
wandb login
python learn_dfdc.py            # WandB active unless --no-wandb is passed
```

Logged per epoch: total `loss` and `learning_rate`. The run config records the
learning rate, epoch count, seed, device, scheduler, and resume state. Model
checkpoints are also uploaded as WandB artifacts when WandB is active.

---

**Keywords:** Phase-Field Modeling · Cahn-Hilliard · Flory-Huggins · Adjoint
Methods · Differentiable Physics · Thermodynamics Discovery · Finite Element
Method · Firedrake · Physics-Informed Machine Learning
