# =============================================================================
# 4_sweep_factor.py - Flight-speed driven phase transition (starling parameters)
#
# Fixed: 10 % wise birds, rho = 2.2, r0 = 1 (k ~ 7 neighbours), eta1 = 0.1, eta2 = 0.5.
# Swept: speed factor = v0*dt / r0 (displacement per update in units of r0).
#
# Conversion to real starlings (same assumptions as the calibration):
#   factor 0.4 <-> mean speed ~11.8 m/s, so 1 factor unit ~ 29.5 m/s.
#   Observed event-mean speeds 8.1-19.2 m/s  ->  factor ~ 0.27-0.65 (green band).
# Limits: factor -> 0 approaches the static (XY-like) limit; factor > 1 means a
# bird moves farther than its perception range per update (mean-field-like
# regime; the original script used 1.5).
#
# Usage:
#   python 4_sweep_factor.py                      # test profile (3000 steps, 3 seeds)
#   python 4_sweep_factor.py --profile paper
#   python 4_sweep_factor.py --steps 5000 --seeds 4 --wise 0.10
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
# PARAMETERS
# -----------------------------------------------------------------------------
TOTAL_N = args.N
rho = 2.2
r0 = 1.0
deltat = 1.0
eta1 = 0.10
eta2 = 0.50                   # naive: at eta_c (critical on its own)
WISE_FRACTION = args.wise

FACTOR_STARLING = 0.4                       # ~11.8 m/s mean starling speed
MS_PER_FACTOR = 11.8 / FACTOR_STARLING      # ~29.5 m/s per factor unit (assumption-based)
SPEED_BAND_MS = vc.STARLING_REF["speed_m_s"]
FACTOR_BAND = (SPEED_BAND_MS[0] / MS_PER_FACTOR, SPEED_BAND_MS[1] / MS_PER_FACTOR)

FACTOR_VALUES = np.array([0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                          0.40, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00])

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"4_sweep_factor_{args.profile}_N{TOTAL_N}"
L = float(np.sqrt(TOTAL_N / rho))

N1 = int(round(WISE_FRACTION * TOTAL_N))
noise_profile = np.full(TOTAL_N, eta2, dtype=np.float32)
noise_profile[:N1] = eta1
eta_e = (N1 * eta1 + (TOTAL_N - N1) * eta2) / TOTAL_N

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N={TOTAL_N} wise={WISE_FRACTION*100:.0f}% (N1={N1}) rho={rho} L={L:.2f} "
      f"r0={r0} eta1={eta1} eta2={eta2} | steps={steps} tail={tail} seeds={n_seeds}")
print(f"Starling speed band {SPEED_BAND_MS[0]}-{SPEED_BAND_MS[1]} m/s -> factor "
      f"{FACTOR_BAND[0]:.2f}-{FACTOR_BAND[1]:.2f}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------
rows, time_series = [], {}
t_global = time.time()
for factor in FACTOR_VALUES:
    t0 = time.time()
    factor = float(factor)
    v0 = (r0 / deltat) * factor
    series = []
    for s in range(n_seeds):
        print(f"\r >> factor={factor:<5} (~{factor*MS_PER_FACTOR:4.1f} m/s) seed {s+1}/{n_seeds} ...",
              end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0, steps, seed=s))
    time_series[factor] = series[0]
    st = vc.steady_state_stats(series, tail, TOTAL_N)
    rows.append(dict(Factor=factor, V0=v0, Speed_ms_approx=factor * MS_PER_FACTOR, Rho=rho, R0=r0,
                     N1_Wise=N1, Wise_Ratio=WISE_FRACTION, Effective_Noise=eta_e,
                     Sim_Steps=steps, Seeds=n_seeds, **st))
    print(f"\r >> factor={factor:<5} (~{factor*MS_PER_FACTOR:4.1f} m/s) | <Omega>={st['Omega_Avg']:.4f}"
          f" ± {st['Omega_StdErr']:.4f} | Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f}"
          f" | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# FIGURE + EXPORT
# -----------------------------------------------------------------------------
param_text = (f"N = {TOTAL_N} | wise = {WISE_FRACTION*100:.0f}% | rho = {rho} | L = {L:.2f} | r0 = {r0} | "
              f"eta1 = {eta1} | eta2 = {eta2} | steps = {steps} (tail {tail}) | seeds = {n_seeds} | "
              f"backend = {backend}")
png = f"{out_dir}/{tag}.png"
f_peak = vr.make_sweep_figure(
    df, x_col="Factor", x_label=r"Speed factor $v_0\Delta t/r_0$  (log scale)", x_fmt="{:.2f}",
    time_series=time_series, legend_fmt=lambda f: f"f={f:g}",
    title=rf"Flight-Speed Driven Phase Transition - starling parameters, {WISE_FRACTION*100:.0f}% wise",
    param_text=param_text,
    table_cols=[("factor", "Factor", "{:.2f}"), ("~m/s", "Speed_ms_approx", "{:.1f}"),
                ("<Omega>", "Omega_Avg", "{:.3f}"), ("StdErr", "Omega_StdErr", "{:.3f}"),
                ("Chi", "Chi", "{:.2f}"), ("U4", "Binder_U4", "{:.3f}")],
    png_path=png, steps=steps, tail=tail,
    ref_vline=FACTOR_STARLING, ref_vline_label="Starling mean speed",
    ref_band=FACTOR_BAND, ref_band_label="Observed starling speeds",
    log_x=True,
)
m, t = vr.export_csv(df, time_series, out_dir, tag, ts_prefix="factor_")
print(f"Susceptibility peak at factor ~ {f_peak:g} (~{f_peak*MS_PER_FACTOR:.1f} m/s)")
print(f"Figure  -> {png}\nMetrics -> {m}\nSeries  -> {t}\nALL DONE.")
