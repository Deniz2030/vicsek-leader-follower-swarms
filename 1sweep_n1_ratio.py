# =============================================================================
# 1sweep_n1_ratio.py - Experienced (Wise) Leader Ratio Scan
#   Heterogeneous-noise Vicsek model with starling-based parameters
#
# Usage:
#   python 1sweep_n1_ratio.py                           # test profile (3000 steps, 3 seeds)
#   python 1sweep_n1_ratio.py --profile paper           # 10000 steps, 5 seeds
#   python 1sweep_n1_ratio.py --steps 5000 --tail 2500 --seeds 4
#   python 1sweep_n1_ratio.py --profile paper --N 20000
#   VICSEK_BACKEND=numpy python 1sweep_n1_ratio.py      # force CPU
# =============================================================================
import time

import numpy as np
import pandas as pd

import vicsek_core as vc
import vicsek_report as vr

args, prof = vr.parse_args(default_N=3000)

# -----------------------------------------------------------------------------
# STARLING-BASED DIMENSIONLESS PARAMETERS
# -----------------------------------------------------------------------------
# Field data (STARFLAG): flocks of 122-4268 birds, nearest-neighbour distance
# 0.68-1.51 m, speeds ~8-19 m/s (mean ~11.8), polarization 0.84-0.99 (mean ~0.97),
# each bird tracks ~6-7 nearest neighbours.
TOTAL_N = args.N        # 3000 lies inside the observed flock-size range
r0 = 1.0                # interaction radius = length unit
rho = 2.2               # pi*r0^2*rho ~ 6.9 neighbours  -> matches 6-7 topological neighbours
# Mean NN distance in 2D at rho=2.2: 1/(2*sqrt(rho)) ~ 0.34 r0  => r0 ~ 3 m for d_NN ~ 1 m.
# With v ~ 11.8 m/s and an update (reaction) time dt ~ 0.1 s (assumption): v*dt ~ 0.4 r0
deltat = 1.0
factor = 0.4
v0 = (r0 / deltat) * factor

# Noise levels (eta * U(-pi, pi)); calibrated for rho=2.2, v0=0.4, N=3000:
#   homogeneous eta=0.10 -> Omega ~ 0.97 (= observed starling polarization)
#   homogeneous eta_c ~ 0.48-0.50 (chi peak and U4 drop, GPU run of 5_sweep_eta.py)
eta1 = 0.10   # experienced (wise) birds
eta2 = 0.50   # naive (panic) birds: AT eta_c -> a naive-only flock is critical (U4 ~ 0.42)
L = float(np.sqrt(TOTAL_N / rho))

# Wise fraction: dense in the transition region (0-30 %), coarse beyond
WISE_FRACTIONS = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16,
                           0.18, 0.20, 0.25, 0.30, 0.40, 0.50])
N1_values = np.unique(np.round(WISE_FRACTIONS * TOTAL_N).astype(int))

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"1sweep_n1_ratio_{args.profile}_N{TOTAL_N}"

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N={TOTAL_N} rho={rho} L={L:.2f} r0={r0} v0={v0:.2f} "
      f"eta1={eta1} eta2={eta2} | steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------
rows, time_series = [], {}
t_global = time.time()
for N1 in N1_values:
    t0 = time.time()
    N1 = int(N1)
    wise_ratio = N1 / TOTAL_N
    eta_e = (N1 * eta1 + (TOTAL_N - N1) * eta2) / TOTAL_N
    noise_profile = np.full(TOTAL_N, eta2, dtype=np.float32)
    noise_profile[:N1] = eta1

    series = []
    for s in range(n_seeds):
        print(f"\r >> N1={N1:<6} ({wise_ratio*100:5.1f}%) seed {s+1}/{n_seeds} ...", end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0, steps, seed=s))
    time_series[N1] = series[0]

    st = vc.steady_state_stats(series, tail, TOTAL_N)
    rows.append(dict(N1_Wise=N1, Wise_Ratio=wise_ratio, Sim_Steps=steps, Seeds=n_seeds,
                     Effective_Noise=eta_e, **st, Sigma_eta=st["Chi"] / eta_e))
    print(f"\r >> N1={N1:<6} ({wise_ratio*100:5.1f}%) | <Omega>={st['Omega_Avg']:.4f}"
          f" ± {st['Omega_StdErr']:.4f} | Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f}"
          f" | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# FIGURE + EXPORT
# -----------------------------------------------------------------------------
param_text = (f"N = {TOTAL_N} | rho = {rho} | L = {L:.2f} | r0 = {r0} | v0 = {v0:.2f} | "
              f"eta1 = {eta1} | eta2 = {eta2} | steps = {steps} (tail {tail}) | "
              f"seeds = {n_seeds} | backend = {backend}")
png = f"{out_dir}/{tag}.png"
f_peak = vr.make_sweep_figure(
    df, x_col="Wise_Ratio", x_label=r"Wise ratio $N_1/N$", x_fmt="{:.2f}",
    time_series=time_series, legend_fmt=lambda n1: f"{n1/TOTAL_N*100:.0f}%",
    title=r"Phase Transition vs. Experienced Leadership Ratio ($N_1/N$) - starling parameters",
    param_text=param_text,
    table_cols=[("N1", "N1_Wise", "{}"), ("Ratio", "Wise_Ratio", "{:.0%}"),
                ("eta_e", "Effective_Noise", "{:.3f}"), ("<Omega>", "Omega_Avg", "{:.3f}"),
                ("StdErr", "Omega_StdErr", "{:.3f}"), ("Chi", "Chi", "{:.2f}"),
                ("U4", "Binder_U4", "{:.3f}")],
    png_path=png, steps=steps, tail=tail,
)
m, t = vr.export_csv(df, time_series, out_dir, tag, ts_prefix="N1_")
print(f"Susceptibility peak at wise ratio ~ {f_peak*100:.1f} %")
print(f"Figure  -> {png}\nMetrics -> {m}\nSeries  -> {t}\nALL DONE.")
