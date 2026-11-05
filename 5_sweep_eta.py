# =============================================================================
# 5_sweep_eta.py - Noise-driven phase transition of a HOMOGENEOUS flock
#
# The classic Vicsek experiment with starling parameters: every bird has the
# same noise eta (no wise / naive split). This sweep is the calibration baseline
# of the whole series: it locates the critical noise eta_c and shows where the
# two bird types of sweeps 1-4 sit:
#   eta1 = 0.10 (wise)  -> deep in the ordered phase, Omega ~ observed starlings
#   eta2 = 0.50 (naive) -> at eta_c (GPU test run: chi peak at 0.50, U4 ~ 0.42)
#
# Usage:
#   python 5_sweep_eta.py                         # test profile (3000 steps, 3 seeds)
#   python 5_sweep_eta.py --profile paper
#   python 5_sweep_eta.py --steps 5000 --seeds 4
#   python 5_sweep_eta.py --fine --profile paper  # 0.44-0.56 in steps of 0.01 (locate eta_c)
# =============================================================================
import time

import numpy as np
import pandas as pd

import vicsek_core as vc
import vicsek_report as vr

args, prof = vr.parse_args(
    default_N=3000,
    extra=lambda p: p.add_argument("--fine", action="store_true",
                                   help="fine grid 0.44-0.56 (step 0.01) around eta_c"),
)

# -----------------------------------------------------------------------------
# PARAMETERS (starling calibration)
# -----------------------------------------------------------------------------
TOTAL_N = args.N
rho = 2.2
r0 = 1.0
deltat = 1.0
factor = 0.4
v0 = (r0 / deltat) * factor
ETA_WISE = 0.10
ETA_NAIVE = 0.50

ETA_VALUES = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40,
                       0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.70])
if args.fine:
    ETA_VALUES = np.round(np.arange(0.44, 0.5601, 0.01), 2)

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"5_sweep_eta_{args.profile}{'_fine' if args.fine else ''}_N{TOTAL_N}"
L = float(np.sqrt(TOTAL_N / rho))

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N={TOTAL_N} homogeneous | rho={rho} L={L:.2f} r0={r0} v0={v0:.2f} "
      f"| steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------
rows, time_series = [], {}
t_global = time.time()
for eta in ETA_VALUES:
    t0 = time.time()
    eta = float(eta)
    noise_profile = np.full(TOTAL_N, eta, dtype=np.float32)
    series = []
    for s in range(n_seeds):
        print(f"\r >> eta={eta:<5} (eta_std={2*np.pi*eta:4.2f}) seed {s+1}/{n_seeds} ...", end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0, steps, seed=s))
    time_series[eta] = series[0]
    st = vc.steady_state_stats(series, tail, TOTAL_N)
    rows.append(dict(Eta=eta, Eta_standard=2 * np.pi * eta, Rho=rho, R0=r0, V0=v0,
                     Sim_Steps=steps, Seeds=n_seeds, **st))
    print(f"\r >> eta={eta:<5} (eta_std={2*np.pi*eta:4.2f}) | <Omega>={st['Omega_Avg']:.4f}"
          f" ± {st['Omega_StdErr']:.4f} | Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f}"
          f" | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# FIGURE + EXPORT
# -----------------------------------------------------------------------------
param_text = (f"N = {TOTAL_N} (homogeneous) | rho = {rho} | L = {L:.2f} | r0 = {r0} | v0 = {v0:.2f} | "
              f"steps = {steps} (tail {tail}) | seeds = {n_seeds} | backend = {backend} | "
              r"noise = $\eta\,U(-\pi,\pi)$, $\eta_{std}=2\pi\eta$")
png = f"{out_dir}/{tag}.png"
eta_peak = vr.make_sweep_figure(
    df, x_col="Eta", x_label=r"Noise $\eta$  (noise term $\eta\,\xi$, $\xi\sim U(-\pi,\pi)$)", x_fmt="{:.2f}",
    time_series=time_series, legend_fmt=lambda e: f"eta={e:g}",
    title=r"Noise-Driven Phase Transition of a Homogeneous Flock ($\eta$) - starling parameters",
    param_text=param_text,
    table_cols=[("eta", "Eta", "{:.2f}"), ("eta_std", "Eta_standard", "{:.2f}"),
                ("<Omega>", "Omega_Avg", "{:.3f}"), ("StdErr", "Omega_StdErr", "{:.3f}"),
                ("Chi", "Chi", "{:.2f}"), ("U4", "Binder_U4", "{:.3f}")],
    png_path=png, steps=steps, tail=tail,
    extra_vlines=[(ETA_WISE, r"$\eta_1$ wise", "green"), (ETA_NAIVE, r"$\eta_2$ naive", "black")],
)
m, t = vr.export_csv(df, time_series, out_dir, tag, ts_prefix="eta_")
print(f"Susceptibility peak (eta_c estimate) at eta ~ {eta_peak:g} (eta_std ~ {2*np.pi*eta_peak:.2f})")
print(f"Figure  -> {png}\nMetrics -> {m}\nSeries  -> {t}\nALL DONE.")
