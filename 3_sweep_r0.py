# =============================================================================
# 3_sweep_r0.py - Interaction-radius (perception range) driven phase transition
#
# Starling parameters, fixed flock composition (10 % wise, chi peak of sweep 1),
# fixed density rho = 2.2 and fixed speed v0 = 0.4 (physical flight speed).
# Sweeping r0 changes two dimensionless groups at once:
#   k = pi r0^2 rho    (mean number of neighbours)
#   v0 / r0            (displacement per update relative to perception range)
# Compare with 2_sweep_rho.py, where k changes but v0/r0 stays fixed: equal k
# with different v0/r0 isolates the effect of the update displacement.
#
# Usage:
#   python 3_sweep_r0.py                          # test profile (3000 steps, 3 seeds)
#   python 3_sweep_r0.py --profile paper
#   python 3_sweep_r0.py --steps 5000 --seeds 4 --wise 0.10
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
# PARAMETERS (starling calibration; r0 = 1 is the reference perception range)
# -----------------------------------------------------------------------------
TOTAL_N = args.N
rho = 2.2
v0 = 0.4                      # fixed physical speed (v*dt ~ 0.4 x reference r0)
eta1 = 0.10
eta2 = 0.50                   # naive: at eta_c (critical on its own)
WISE_FRACTION = args.wise
R0_STARLING = 1.0             # k ~ 6.9 neighbours at rho = 2.2

R0_VALUES = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00,
                      1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 2.50])

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"3_sweep_r0_{args.profile}_N{TOTAL_N}"
L = float(np.sqrt(TOTAL_N / rho))

N1 = int(round(WISE_FRACTION * TOTAL_N))
noise_profile = np.full(TOTAL_N, eta2, dtype=np.float32)
noise_profile[:N1] = eta1
eta_e = (N1 * eta1 + (TOTAL_N - N1) * eta2) / TOTAL_N

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N={TOTAL_N} wise={WISE_FRACTION*100:.0f}% (N1={N1}) rho={rho} L={L:.2f} "
      f"v0={v0} eta1={eta1} eta2={eta2} | steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------
rows, time_series = [], {}
t_global = time.time()
for r0 in R0_VALUES:
    t0 = time.time()
    r0 = float(r0)
    k_mean = np.pi * r0 ** 2 * rho
    series = []
    for s in range(n_seeds):
        print(f"\r >> r0={r0:<5} (k={k_mean:4.1f}, v0/r0={v0/r0:.2f}) seed {s+1}/{n_seeds} ...",
              end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0, steps, seed=s))
    time_series[r0] = series[0]
    st = vc.steady_state_stats(series, tail, TOTAL_N)
    rows.append(dict(R0=r0, Mean_Neighbours=k_mean, V0_over_R0=v0 / r0, Rho=rho, N1_Wise=N1,
                     Wise_Ratio=WISE_FRACTION, Effective_Noise=eta_e, Sim_Steps=steps,
                     Seeds=n_seeds, **st))
    print(f"\r >> r0={r0:<5} (k={k_mean:4.1f}, v0/r0={v0/r0:.2f}) | <Omega>={st['Omega_Avg']:.4f}"
          f" ± {st['Omega_StdErr']:.4f} | Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f}"
          f" | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# FIGURE + EXPORT
# -----------------------------------------------------------------------------
param_text = (f"N = {TOTAL_N} | wise = {WISE_FRACTION*100:.0f}% | rho = {rho} | L = {L:.2f} | v0 = {v0} | "
              f"eta1 = {eta1} | eta2 = {eta2} | steps = {steps} (tail {tail}) | seeds = {n_seeds} | "
              f"backend = {backend}")
png = f"{out_dir}/{tag}.png"
r0_peak = vr.make_sweep_figure(
    df, x_col="R0", x_label=r"Interaction radius $r_0$  ($k=\pi r_0^2\rho$)", x_fmt="{:.2f}",
    time_series=time_series, legend_fmt=lambda r: f"r0={r:g}",
    title=rf"Perception-Range Driven Phase Transition ($r_0$) - starling parameters, "
          rf"{WISE_FRACTION*100:.0f}% wise",
    param_text=param_text,
    table_cols=[("r0", "R0", "{:.2f}"), ("k", "Mean_Neighbours", "{:.1f}"),
                ("v0/r0", "V0_over_R0", "{:.2f}"), ("<Omega>", "Omega_Avg", "{:.3f}"),
                ("StdErr", "Omega_StdErr", "{:.3f}"), ("Chi", "Chi", "{:.2f}"),
                ("U4", "Binder_U4", "{:.3f}")],
    png_path=png, steps=steps, tail=tail,
    ref_vline=R0_STARLING, ref_vline_label=r"Starling $r_0$ (k$\approx$7)",
)
m, t = vr.export_csv(df, time_series, out_dir, tag, ts_prefix="r0_")
print(f"Susceptibility peak at r0 ~ {r0_peak:g} (k ~ {np.pi*r0_peak**2*rho:.1f} neighbours)")
print(f"Figure  -> {png}\nMetrics -> {m}\nSeries  -> {t}\nALL DONE.")
