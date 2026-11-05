# =============================================================================
# 2_sweep_rho.py - Density-driven phase transition (starling parameters)
#
# Fixed flock composition from sweep 1: 10 % experienced (wise) birds, which is
# where the susceptibility peaked. The density rho (and thus the mean number of
# neighbours k = pi r0^2 rho) is swept across the order-disorder transition.
# The starling reference value rho = 2.2 (k ~ 7 neighbours) is marked.
#
# Usage:
#   python 2_sweep_rho.py                         # test profile (3000 steps, 3 seeds)
#   python 2_sweep_rho.py --profile paper
#   python 2_sweep_rho.py --steps 5000 --seeds 4 --wise 0.10
#   python 2_sweep_rho.py --profile paper --N 20000
# =============================================================================
import time

import numpy as np
import pandas as pd

import vicsek_core as vc
import vicsek_report as vr

args, prof = vr.parse_args(
    default_N=3000,
    extra=lambda p: p.add_argument("--wise", type=float, default=0.10, help="wise fraction N1/N"),
)

# -----------------------------------------------------------------------------
# PARAMETERS (same starling calibration as 1sweep_n1_ratio.py)
# -----------------------------------------------------------------------------
TOTAL_N = args.N
r0 = 1.0
deltat = 1.0
factor = 0.4                  # v*dt / r0 for starlings (~11.8 m/s, dt ~ 0.1 s)
v0 = (r0 / deltat) * factor
eta1 = 0.10                   # wise: Omega ~ 0.97 alone
eta2 = 0.50                   # naive: at eta_c (critical on its own)
WISE_FRACTION = args.wise
RHO_STARLING = 2.2            # pi*r0^2*rho ~ 6.9 neighbours (6-7 topological neighbours)

RHO_VALUES = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.20, 2.50,
                       2.75, 3.00, 3.50, 4.00, 5.00, 6.00])

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"2_sweep_rho_{args.profile}_N{TOTAL_N}"

N1 = int(round(WISE_FRACTION * TOTAL_N))
noise_profile = np.full(TOTAL_N, eta2, dtype=np.float32)
noise_profile[:N1] = eta1
eta_e = (N1 * eta1 + (TOTAL_N - N1) * eta2) / TOTAL_N

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N={TOTAL_N} wise={WISE_FRACTION*100:.0f}% (N1={N1}) r0={r0} "
      f"v0={v0:.2f} eta1={eta1} eta2={eta2} | steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------
rows, time_series = [], {}
t_global = time.time()
for rho in RHO_VALUES:
    t0 = time.time()
    rho = float(rho)
    L = float(np.sqrt(TOTAL_N / rho))
    k_mean = np.pi * r0 ** 2 * rho
    series = []
    for s in range(n_seeds):
        print(f"\r >> rho={rho:<5} (k={k_mean:4.1f}) seed {s+1}/{n_seeds} ...", end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0, steps, seed=s))
    time_series[rho] = series[0]
    st = vc.steady_state_stats(series, tail, TOTAL_N)
    rows.append(dict(Rho=rho, Mean_Neighbours=k_mean, Box_L=L, N1_Wise=N1, Wise_Ratio=WISE_FRACTION,
                     Effective_Noise=eta_e, Sim_Steps=steps, Seeds=n_seeds, **st))
    print(f"\r >> rho={rho:<5} (k={k_mean:4.1f}) | <Omega>={st['Omega_Avg']:.4f} ± {st['Omega_StdErr']:.4f}"
          f" | Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f} | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# FIGURE + EXPORT
# -----------------------------------------------------------------------------
param_text = (f"N = {TOTAL_N} | wise = {WISE_FRACTION*100:.0f}% | r0 = {r0} | v0 = {v0:.2f} | "
              f"eta1 = {eta1} | eta2 = {eta2} | steps = {steps} (tail {tail}) | seeds = {n_seeds} | "
              f"backend = {backend}")
png = f"{out_dir}/{tag}.png"
rho_peak = vr.make_sweep_figure(
    df, x_col="Rho", x_label=r"Density $\rho$  (mean neighbours $k=\pi r_0^2\rho$)", x_fmt="{:.2f}",
    time_series=time_series, legend_fmt=lambda r: f"rho={r:g}",
    title=rf"Density-Driven Phase Transition ($\rho$) - starling parameters, {WISE_FRACTION*100:.0f}% wise",
    param_text=param_text,
    table_cols=[("rho", "Rho", "{:.2f}"), ("k", "Mean_Neighbours", "{:.1f}"), ("L", "Box_L", "{:.1f}"),
                ("<Omega>", "Omega_Avg", "{:.3f}"), ("StdErr", "Omega_StdErr", "{:.3f}"),
                ("Chi", "Chi", "{:.2f}"), ("U4", "Binder_U4", "{:.3f}")],
    png_path=png, steps=steps, tail=tail,
    ref_vline=RHO_STARLING, ref_vline_label=r"Starling $\rho$ (k$\approx$7)",
)
m, t = vr.export_csv(df, time_series, out_dir, tag, ts_prefix="rho_")
print(f"Susceptibility peak at rho ~ {rho_peak:g} (k ~ {np.pi*rho_peak:.1f} neighbours)")
print(f"Figure  -> {png}\nMetrics -> {m}\nSeries  -> {t}\nALL DONE.")
