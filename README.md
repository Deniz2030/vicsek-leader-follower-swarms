# GPU Vicsek simulations of leader–follower swarms

Code and data for the manuscript *"GPU-Accelerated Simulation and Automated
Critical-Point Detection for Leader–Follower Swarms: Informed-Agent Thresholds
and Communication Topology Calibrated to Starling Flocks"* (manuscript in preparation).

A two-dimensional Vicsek model with heterogeneous noise: informed agents
(noise η1 = 0.10) are mixed with uninformed agents (η2 = 0.50). Density,
perception range and speed are calibrated to field data from starling flocks.
The code runs on NVIDIA GPUs through CuPy and falls back to NumPy on the CPU.

## Contents

| File | Purpose |
|---|---|
| `vicsek_core.py` | Simulation core: GPU/CPU backend, periodic cell list, exact k-nearest-neighbour search, observables, start-up self-test against brute force |
| `vicsek_report.py` | Shared command-line options, run profiles, figure and CSV helpers |
| `1sweep_n1_ratio.py` | Informed-agent fraction sweep (Fig. 2) |
| `2_sweep_rho.py` | Density sweep (Fig. 3) |
| `3_sweep_r0.py` | Perception-range sweep (Fig. 4) |
| `4_sweep_factor.py` | Speed sweep (Fig. 5) |
| `5_sweep_eta.py` | Noise sweep of a homogeneous flock (Fig. 1); `--fine` for 0.44–0.56 |
| `6_sweep_topological.py` | Topological interaction: k sweep (part A, Fig. 6) and density robustness (part B, Fig. 7; `--scale_speed` for the v0 ∝ ρ^(−1/2) control) |
| `7_finite_size.py` | Finite-size scaling over flock sizes N; per-seed checkpoints, resumable |
| `merge_finite_size.py` | Merges runs of `7_finite_size.py` made in separate sittings; Binder crossings of fitted U4 curves, χ_max(L), f_c estimates (Fig. 8) |
| `8_correlation_starling.py` | Velocity-fluctuation correlation C(r) and correlation length ξ(L) (Fig. 9) |
| `find_critical.py` | Algorithm 1: Binder-cumulant bisection with adaptive seeds for any control parameter |
| `make_figures.py` | Publication figures 1–9 from the saved CSV files (no simulation) |
| `data/` | Metric tables (CSV) used in the paper |
| `figures/` | Figures 1–9 (600-dpi PNG and vector PDF) |

## Installation

```
pip install -r requirements.txt
pip install cupy-cuda12x        # optional, for NVIDIA GPUs (match your CUDA version)
```

## Usage

Every script prints `[self-test] ... OK` at start-up; the run stops if the
neighbour search does not match a brute-force check. Run GPU scripts one at a
time, since they share GPU memory.

```
python 5_sweep_eta.py                       # test profile: 3000 steps, 3 seeds
python 1sweep_n1_ratio.py --profile paper   # 10000 steps, 5 seeds
python 2_sweep_rho.py --steps 5000 --tail 2500 --seeds 4
python find_critical.py --param eta --lo 0.30 --hi 0.70 --tol 0.01
python find_critical.py --param k --lo 1 --hi 25 --tol 1
python 7_finite_size.py --N_list 1000 2000 4000 --steps 6000 --seeds 5
python 7_finite_size.py --N_list 8000 --steps 6000 --tail 3000 --seeds 5
python merge_finite_size.py
python make_figures.py                      # writes figures/Fig1–Fig9
```

Set `VICSEK_BACKEND=numpy` to force the CPU.

| Profile | Steps | Steady-state window | Seeds |
|---|---|---|---|
| `smoke` | 60 | 30 | 2 |
| `test` | 3000 | 1500 | 3 |
| `paper` | 10000 | 5000 | 5 |

`--steps`, `--tail` and `--seeds` override any profile. Results in the paper
use the `test` profile unless stated otherwise; fixed seeds make every run
exactly reproducible.

## Model

- Update: θ_i(t+1) = arg(Σ_j e^{iθ_j}) + η_i ξ, with ξ ~ U(−π, π)
  (η_std = 2πη in the standard Vicsek convention).
- Neighbourhood: metric (|r_ij| < r0, self included) or topological
  (self plus the k nearest agents).
- Periodic box of side L = √(N/ρ).

## Calibration to starling flocks

| Parameter | Value | Basis |
|---|---|---|
| N | 3000 | observed flocks: 122–4268 birds |
| ρ | 2.2 | πr0²ρ ≈ 6.9 neighbours; starlings track 6–7 |
| r0 | 1 (≈ 3 m) | mean nearest-neighbour distance 0.34 r0 ≈ 1 m |
| v0 | 0.4 | 11.8 m/s × Δt ≈ 0.1 s (Δt assumed) |
| η1 (informed) | 0.10 | homogeneous Ω ≈ 0.97, the observed polarization |
| η2 (uninformed) | 0.50 | at the homogeneous critical noise η_c ≈ 0.48 |

## Observables

- `Omega_Avg`: steady-state order parameter; error bar = standard error across seeds
- `Chi`: susceptibility N·Var(Ω)
- `Binder_U4`: 1 − ⟨Ω⁴⟩/(3⟨Ω²⟩²); 1/3 when disordered, 2/3 when ordered

## Hardware

Results in the paper were produced on an NVIDIA GeForce RTX 4060 Laptop GPU
(8 GB), driver 581.86, CUDA 13.0.

## Citation

Please cite the article once published (details to be added).

## License

MIT, see `LICENSE`.
