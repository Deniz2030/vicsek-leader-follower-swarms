# =============================================================================
# 7_finite_size.py - Finite-size scaling of the leadership-ratio transition
#
# Repeats the wise-fraction sweep of 1sweep_n1_ratio.py for several flock sizes
# N at fixed density (rho = 2.2, so L = sqrt(N/rho) grows with N).
#   * Binder cumulant U4(f) curves for different N should cross near the
#     thermodynamic-limit transition f_c  (crossing = size-independent point).
#   * The susceptibility peak chi_max should grow with N; its log-log slope
#     estimates gamma/nu (in units of L, chi_max ~ L^(gamma/nu)).
# Observed starling flocks span 122-4268 birds, so the default N list covers
# that range; the paper profile extends beyond it.
#
# Usage:
#   python 7_finite_size.py                               # test: N = 1000, 2000, 4000
#   python 7_finite_size.py --profile paper               # N = 1000 ... 16000
#   python 7_finite_size.py --N_list 1000 2000 4000 8000 --steps 5000 --seeds 4
#   python 7_finite_size.py --N_list 8000 --steps 6000 --tail 3000 --seeds 5
#       (one size per sitting; then combine sittings with merge_finite_size.py)
#
# Resume after an interruption (closed terminal, Windows restart, driver reset):
#   every finished seed is saved at once to
#   data/checkpoints_7_finite_size_steps<steps>/N<N>_f<f>_seed<s>.npy
#   Starting the SAME command again reuses these files and only computes what
#   is missing; the final line reports how many seed runs were reused.
#   Delete that folder to force a fresh run.
#
# With a single N there is no Binder crossing to compute; the crossings CSV is
# then empty and the figure shows that one size (merge_finite_size.py does the
# cross-size analysis).
#
# Outputs (data/): 7_finite_size_<profile>[_N...].png, *_metrics.csv,
#   *_binder_crossings.csv, *_chi_peaks.csv
#
# Revision history:
#   - per-seed checkpoints and resume added after an overnight N = 8000 run was
#     lost to a closed terminal
#   - fixed crash at the end of single-N runs (empty crossings table)
# =============================================================================
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vicsek_core as vc
import vicsek_report as vr

args, prof = vr.parse_args(
    default_N=3000,
    extra=lambda p: p.add_argument("--N_list", type=int, nargs="+", default=None,
                                   help="flock sizes (overrides profile default)"),
)
N_LIST = args.N_list or ([1000, 2000, 4000, 8000, 16000] if args.profile == "paper"
                         else [1000, 2000, 4000])

# -----------------------------------------------------------------------------
# PARAMETERS (same starling calibration as sweep 1)
# -----------------------------------------------------------------------------
rho = 2.2
r0 = 1.0
v0 = 0.4
eta1 = 0.10
eta2 = 0.50                   # naive: at eta_c (critical on its own)
WISE_FRACTIONS = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12,
                           0.14, 0.16, 0.18, 0.20, 0.25, 0.30])

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"7_finite_size_{args.profile}" + (f"_N{'-'.join(map(str, N_LIST))}" if args.N_list else "")

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N_list={N_LIST} | rho={rho} r0={r0} v0={v0} eta1={eta1} eta2={eta2} "
      f"| steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# SWEEP over N and wise fraction
# -----------------------------------------------------------------------------
# Checkpoints: every finished seed is saved at once, so an interrupted run
# (Windows restart, driver reset, closed terminal) resumes where it stopped
# when the SAME command is started again. Delete the folder to start fresh.
ckpt_dir = os.path.join(out_dir, f"checkpoints_7_finite_size_steps{steps}")
os.makedirs(ckpt_dir, exist_ok=True)
n_resumed = 0

rows = []
t_global = time.time()
for N in N_LIST:
    L = float(np.sqrt(N / rho))
    for f in WISE_FRACTIONS:
        t0 = time.time()
        N1 = int(round(f * N))
        prof_noise = np.full(N, eta2, dtype=np.float32)
        prof_noise[:N1] = eta1
        series = []
        for s in range(n_seeds):
            ck = os.path.join(ckpt_dir, f"N{N}_f{f:.3f}_seed{s}.npy")
            if os.path.exists(ck):
                series.append(np.load(ck))
                n_resumed += 1
                continue
            print(f"\r >> N={N:<6} f={f:4.2f} seed {s+1}/{n_seeds} ...", end="", flush=True)
            om = vc.simulate(xp, prof_noise, N, rho, r0, v0, steps, seed=s)
            np.save(ck + ".tmp.npy", om)
            os.replace(ck + ".tmp.npy", ck)          # atomic: no half-written checkpoint
            series.append(om)
        st = vc.steady_state_stats(series, tail, N)
        rows.append(dict(N=N, L=L, Wise_Ratio=f, N1_Wise=N1, Sim_Steps=steps, Seeds=n_seeds, **st))
        print(f"\r >> N={N:<6} f={f:4.2f} | <Omega>={st['Omega_Avg']:.4f} | Chi={st['Chi']:.2f}"
              f" | U4={st['Binder_U4']:.3f} | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}"
      f" ({n_resumed} seed runs reused from checkpoints in {ckpt_dir})")
df = pd.DataFrame(rows)

# Binder crossings between consecutive sizes (linear interpolation of U4 difference)
def crossing(fa, ua, ub):
    d = ua - ub
    for i in range(1, len(d)):
        if np.sign(d[i - 1]) != np.sign(d[i]) and d[i - 1] != 0:
            return float(fa[i - 1] + (fa[i] - fa[i - 1]) * d[i - 1] / (d[i - 1] - d[i]))
    return float("nan")

cross_rows = []
for Na, Nb in zip(N_LIST[:-1], N_LIST[1:]):
    a = df[df.N == Na].sort_values("Wise_Ratio")
    b = df[df.N == Nb].sort_values("Wise_Ratio")
    cross_rows.append(dict(N_a=Na, N_b=Nb, f_cross=crossing(a.Wise_Ratio.values, a.Binder_U4.values,
                                                            b.Binder_U4.values)))
df_cross = pd.DataFrame(cross_rows, columns=["N_a", "N_b", "f_cross"])   # empty when only one N

peaks = df.loc[df.groupby("N")["Chi"].idxmax(), ["N", "L", "Wise_Ratio", "Chi"]].rename(
    columns={"Wise_Ratio": "f_peak", "Chi": "Chi_max"}).reset_index(drop=True)
if len(peaks) >= 2:
    slope, intercept = np.polyfit(np.log(peaks.L), np.log(peaks.Chi_max), 1)
else:
    slope, intercept = float("nan"), float("nan")

# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
plt.suptitle("Finite-Size Scaling of the Leadership-Ratio Transition - starling parameters",
             fontsize=14, fontweight="bold", y=0.975)
cmap = plt.get_cmap("plasma")
cols = {N: cmap(i / max(1, len(N_LIST) - 1) * 0.85) for i, N in enumerate(N_LIST)}

for N in N_LIST:
    d = df[df.N == N].sort_values("Wise_Ratio")
    axs[0, 0].errorbar(d.Wise_Ratio, d.Omega_Avg, yerr=d.Omega_StdErr, fmt="-o", ms=4, capsize=3,
                       color=cols[N], label=f"N={N}")
    axs[0, 1].plot(d.Wise_Ratio, d.Chi, "-s", ms=4, color=cols[N], label=f"N={N}")
    axs[1, 0].plot(d.Wise_Ratio, d.Binder_U4, "-d", ms=4, color=cols[N], label=f"N={N}")

axs[0, 0].set_title(r"A) Order parameter $\langle\Omega\rangle$", fontsize=11, fontweight="bold")
axs[0, 1].set_title(r"B) Susceptibility $\chi = N\,\mathrm{Var}(\Omega)$", fontsize=11, fontweight="bold")
axs[1, 0].set_title(r"C) Binder cumulant $U_4$ (crossing $\approx f_c$)", fontsize=11, fontweight="bold")
for ax, yl in ((axs[0, 0], r"$\langle\Omega\rangle$"), (axs[0, 1], r"$\chi$"), (axs[1, 0], r"$U_4$")):
    ax.set_xlabel(r"Wise ratio $f = N_1/N$")
    ax.set_ylabel(yl)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=8)
axs[1, 0].axhline(2 / 3, color="gray", ls=":", lw=0.8)
for fc in df_cross.f_cross.dropna():
    axs[1, 0].axvline(fc, color="crimson", ls="--", lw=0.9)

ax = axs[1, 1]
ax.loglog(peaks.L, peaks.Chi_max, "o", color="darkblue", ms=7, label=r"$\chi_{max}$")
if np.isfinite(slope):
    Lfit = np.linspace(peaks.L.min(), peaks.L.max(), 50)
    ax.loglog(Lfit, np.exp(intercept) * Lfit ** slope, "--", color="crimson",
              label=rf"fit: $\chi_{{max}}\propto L^{{{slope:.2f}}}$")
for r in peaks.itertuples():
    ax.annotate(f"N={r.N}\nf={r.f_peak:.2f}", (r.L, r.Chi_max), textcoords="offset points",
                xytext=(6, -14), fontsize=7)
ax.set_title(r"D) Peak susceptibility vs box size $L$", fontsize=11, fontweight="bold")
ax.set_xlabel(r"$L=\sqrt{N/\rho}$")
ax.set_ylabel(r"$\chi_{max}$")
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(fontsize=8)

cross_txt = " | ".join(f"U4 cross {r.N_a}/{r.N_b}: f={r.f_cross:.3f}" for r in df_cross.itertuples())
fig.text(0.5, 0.015, f"rho = {rho} | r0 = {r0} | v0 = {v0} | eta1 = {eta1} | eta2 = {eta2} | steps = {steps} "
         f"(tail {tail}) | seeds = {n_seeds} | backend = {backend}\n{cross_txt}",
         ha="center", fontsize=8.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.92, wspace=0.28, hspace=0.32)
png = os.path.join(out_dir, f"{tag}.png")
plt.savefig(png, dpi=300)
plt.close(fig)

# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------
df.to_csv(os.path.join(out_dir, f"{tag}_metrics.csv"), index=False, float_format="%.6f")
df_cross.to_csv(os.path.join(out_dir, f"{tag}_binder_crossings.csv"), index=False, float_format="%.6f")
peaks.assign(Chi_max_slope_vs_L=slope).to_csv(os.path.join(out_dir, f"{tag}_chi_peaks.csv"),
                                               index=False, float_format="%.6f")
print(df_cross.to_string(index=False))
print(f"chi_max ~ L^{slope:.3f}")
print(f"Figure  -> {png}\nCSV     -> {out_dir}/{tag}_*.csv\nALL DONE.")
