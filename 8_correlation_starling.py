# =============================================================================
# 8_correlation_starling.py - Scale-free velocity correlations (starling test)
#
# Field studies of starling flocks found that velocity fluctuations are
# correlated over distances that grow with flock size (scale-free correlations,
# Cavagna et al. 2010). This script measures the same quantity in the model:
#   C(r)  = connected correlation of direction fluctuations
#   xi    = first zero crossing of C(r)  (correlation length)
# for several flock sizes N (box size L = sqrt(N/rho)) and two regimes:
#   "starling" : homogeneous eta = 0.10, Omega ~ 0.97 like real flocks
#   "critical" : 10 % wise (eta1 = 0.1) + 90 % naive (eta2 = 0.5), chi peak of sweep 1
# Scale-free behaviour <=> xi grows linearly with L and C(r) curves collapse
# when plotted against r/L.
#
# Usage:
#   python 8_correlation_starling.py                     # test: N = 500 ... 4000
#   python 8_correlation_starling.py --profile paper     # N = 500 ... 8000
#   python 8_correlation_starling.py --N_list 500 1000 2000 4000 8000 --seeds 3
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
    extra=lambda p: (p.add_argument("--N_list", type=int, nargs="+", default=None),
                     p.add_argument("--snap_every", type=int, default=100,
                                    help="steps between correlation snapshots in the tail")),
)
N_LIST = args.N_list or ([500, 1000, 2000, 4000, 8000] if args.profile == "paper"
                         else [500, 1000, 2000, 4000])

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
rho = 2.2
r0 = 1.0
v0 = 0.4
DR = 0.25                     # radial bin width (units of r0)
REGIMES = {
    "starling": dict(label=r"Starling-like order ($\eta=0.10$, homogeneous)", eta1=0.10, eta2=0.10, wise=1.0),
    "critical": dict(label=r"Critical mix (10% wise $\eta_1=0.1$, naive $\eta_2=0.5$)", eta1=0.10, eta2=0.50,
                     wise=0.10),
}

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
snap_every = max(1, min(args.snap_every, tail))
out_dir = vr.output_dir_for(__file__)
tag = f"8_correlation_starling_{args.profile}"

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | N_list={N_LIST} | rho={rho} r0={r0} v0={v0} | steps={steps} "
      f"tail={tail} seeds={n_seeds} snapshots every {snap_every}")
print(f"Output: {out_dir}")
print("=" * 78)

# -----------------------------------------------------------------------------
# RUNS
# -----------------------------------------------------------------------------
curves = {}           # (regime, N) -> (r, C_mean)
rows = []
example = {}          # regime -> (pos, th, L) of the largest N, seed 0, last snapshot
t_global = time.time()
for reg, cfg in REGIMES.items():
    for N in N_LIST:
        t0 = time.time()
        L = float(np.sqrt(N / rho))
        N1 = int(round(cfg["wise"] * N))
        noise = np.full(N, cfg["eta2"], dtype=np.float32)
        noise[:N1] = cfg["eta1"]
        C_acc, xis, omegas = [], [], []
        for s in range(n_seeds):
            print(f"\r >> {reg:<8} N={N:<6} seed {s+1}/{n_seeds} ...", end="", flush=True)
            om, snaps = vc.simulate(xp, noise, N, rho, r0, v0, steps, seed=s,
                                    snapshot_every=snap_every, snapshot_from=steps - tail)
            omegas.append(om[-tail:].mean())
            for pos, th in snaps:
                r, C, _ = vc.velocity_correlation(xp, pos, th, L, dr=DR)
                C_acc.append(C)
                xis.append(vc.zero_crossing(r, C))
            if s == 0 and N == N_LIST[-1]:
                example[reg] = (snaps[-1][0], snaps[-1][1], L)
        C_mean = np.nanmean(np.array(C_acc), axis=0)
        curves[(reg, N)] = (r, C_mean)
        xi_mean_curve = vc.zero_crossing(r, C_mean)
        xis = np.array(xis, dtype=float)
        rows.append(dict(Regime=reg, N=N, L=L, Omega_Avg=float(np.mean(omegas)),
                         Xi_from_mean_C=xi_mean_curve, Xi_mean=float(np.nanmean(xis)),
                         Xi_std=float(np.nanstd(xis)), Xi_over_L=xi_mean_curve / L,
                         Snapshots=len(C_acc), Sim_Steps=steps, Seeds=n_seeds))
        print(f"\r >> {reg:<8} N={N:<6} L={L:6.2f} | <Omega>={np.mean(omegas):.3f} | xi={xi_mean_curve:6.2f}"
              f" | xi/L={xi_mean_curve/L:.3f} | {time.time()-t0:.1f}s")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}")
df = pd.DataFrame(rows)

fits = {}
for reg in REGIMES:
    d = df[(df.Regime == reg) & np.isfinite(df.Xi_from_mean_C)]
    if len(d) >= 2:
        fits[reg] = np.polyfit(d.L, d.Xi_from_mean_C, 1)

# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
plt.suptitle("Velocity-Fluctuation Correlations: Scale-Free Test (starling parameters)",
             fontsize=14, fontweight="bold", y=0.975)
cmap = plt.get_cmap("plasma")
cols = {N: cmap(i / max(1, len(N_LIST) - 1) * 0.85) for i, N in enumerate(N_LIST)}
styles = {"starling": "-", "critical": "--"}

for reg in REGIMES:
    for N in N_LIST:
        r, C = curves[(reg, N)]
        L = float(np.sqrt(N / rho))
        lab = f"N={N}" if reg == "starling" else None
        axs[0, 0].plot(r, C, styles[reg], color=cols[N], lw=1.3, label=lab)
        axs[0, 1].plot(r / L, C, styles[reg], color=cols[N], lw=1.3, label=lab)
for ax in (axs[0, 0], axs[0, 1]):
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_ylabel(r"$C(r)$")
axs[0, 0].plot([], [], "k-", label="starling-like")
axs[0, 0].plot([], [], "k--", label="critical mix")
axs[0, 0].legend(fontsize=7.5, ncol=2)
axs[0, 0].set_xlabel(r"$r$  (units of $r_0$)")
axs[0, 0].set_title(r"A) Correlation function $C(r)$", fontsize=11, fontweight="bold")
axs[0, 1].set_xlabel(r"$r/L$")
axs[0, 1].set_title(r"B) Rescaled: $C$ vs $r/L$ (collapse = scale-free)", fontsize=11, fontweight="bold")

ax = axs[1, 0]
for reg, mk in (("starling", "o"), ("critical", "s")):
    d = df[df.Regime == reg]
    eb = ax.errorbar(d.L, d.Xi_from_mean_C, yerr=d.Xi_std, fmt=mk, ms=6, capsize=3,
                     label=REGIMES[reg]["label"])
    if reg in fits:
        a, b = fits[reg]
        Lf = np.linspace(0, d.L.max() * 1.05, 20)
        ax.plot(Lf, a * Lf + b, ":", lw=1.2, color=eb[0].get_color(), label=rf"fit {reg}: $\xi \approx {a:.3f}\,L {b:+.2f}$")
ax.set_xlabel(r"Box size $L=\sqrt{N/\rho}$")
ax.set_ylabel(r"Correlation length $\xi$ (zero crossing)")
ax.set_title(r"C) $\xi$ grows with system size?", fontsize=11, fontweight="bold")
ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=7.5)

ax = axs[1, 1]
reg_show = "starling" if "starling" in example else next(iter(example))
pos, th, L = example[reg_show]
dux, duy = np.cos(th) - np.cos(th).mean(), np.sin(th) - np.sin(th).mean()
ang = np.arctan2(duy, dux)
sub = np.arange(0, len(th), max(1, len(th) // 2500))
ax.quiver(pos[sub, 0], pos[sub, 1], dux[sub], duy[sub], ang[sub], cmap="hsv", scale=None,
          width=0.003, headwidth=3)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect("equal")
ax.set_title(f"D) Direction fluctuations $\\delta u_i$ ({reg_show}, N={N_LIST[-1]})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.text(0.5, 0.015, f"rho = {rho} | r0 = {r0} | v0 = {v0} | dr = {DR} | steps = {steps} (tail {tail}) | "
         f"seeds = {n_seeds} | snapshots every {snap_every} | backend = {backend}",
         ha="center", fontsize=8.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.92, wspace=0.28, hspace=0.32)
png = os.path.join(out_dir, f"{tag}.png")
plt.savefig(png, dpi=300)
plt.close(fig)

# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------
df.to_csv(os.path.join(out_dir, f"{tag}_xi.csv"), index=False, float_format="%.6f")
# curves of all sizes on the r grid of the largest box (smaller boxes padded with NaN)
r_long = curves[(next(iter(REGIMES)), N_LIST[-1])][0]
curve_df = pd.DataFrame({"r": r_long})
for (reg, N), (r, C) in curves.items():
    col = np.full(len(r_long), np.nan)
    col[:len(C)] = C
    curve_df[f"C_{reg}_N{N}"] = col
curve_df.to_csv(os.path.join(out_dir, f"{tag}_Cr.csv"), index=False, float_format="%.6f")
for reg, (a, b) in fits.items():
    print(f"{reg}: xi ~ {a:.3f} L {b:+.2f}")
print(f"Figure  -> {png}\nCSV     -> {out_dir}/{tag}_*.csv\nALL DONE.")
