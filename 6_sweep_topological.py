# =============================================================================
# 6_sweep_topological.py - Topological interaction (k nearest neighbours)
#
# Field data show that each starling interacts with a fixed NUMBER of
# neighbours (~6-7), not with all birds inside a fixed DISTANCE (Ballerini et
# al. 2008). This script runs two experiments with the topological model
# (each bird aligns with itself + its k nearest neighbours, exact kNN):
#
#   Part A  k sweep      : rho = 2.2, 10 % wise; k = 1 ... 25 (6-7 band shaded)
#   Part B  robustness   : density rho = 0.5 ... 6 for METRIC (r0 = 1) and
#                          TOPOLOGICAL (k = 7) interaction, same 10 % wise flock.
#                          The topological flock should be far less sensitive
#                          to density changes (Ballerini's cohesion argument).
#
# Usage:
#   python 6_sweep_topological.py                   # both parts, test profile
#   python 6_sweep_topological.py --part A
#   python 6_sweep_topological.py --part B --k_robust 7 --profile paper
#   python 6_sweep_topological.py --part B --scale_speed   # control: v0*sqrt(rho) constant
# =============================================================================
import os
import time

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vicsek_core as vc
import vicsek_report as vr

args, prof = vr.parse_args(
    default_N=3000,
    extra=lambda p: (p.add_argument("--wise", type=float, default=0.10, help="wise fraction N1/N"),
                     p.add_argument("--part", choices=["A", "B", "both"], default="both"),
                     p.add_argument("--k_robust", type=int, default=7,
                                    help="k used for the topological curve in part B"),
                     p.add_argument("--scale_speed", action="store_true",
                                    help="part B control: v0 = 0.4*sqrt(2.2/rho), so the step per "
                                         "neighbour spacing (v0*sqrt(rho)) stays constant")),
)

# -----------------------------------------------------------------------------
# PARAMETERS (starling calibration)
# -----------------------------------------------------------------------------
TOTAL_N = args.N
rho_ref = 2.2
r0 = 1.0
v0 = 0.4
eta1 = 0.10
eta2 = 0.50                   # naive: at eta_c (critical on its own)
WISE_FRACTION = args.wise
K_STARLING_BAND = vc.STARLING_REF["topological_neighbours"]       # (6, 7)

K_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25])
RHO_ROBUST = np.array([0.5, 0.75, 1.0, 1.5, 2.2, 3.0, 4.0, 6.0])

steps, tail, n_seeds = prof["steps"], prof["tail"], prof["seeds"]
out_dir = vr.output_dir_for(__file__)
tag = f"6_sweep_topological_{args.profile}_N{TOTAL_N}"
tag_B = tag + ("_scaledv" if args.scale_speed else "")


def v0_at(rho):
    """Speed used at density rho (constant, or scaled for the part-B control)."""
    return v0 * np.sqrt(rho_ref / rho) if args.scale_speed else v0


N1 = int(round(WISE_FRACTION * TOTAL_N))
noise_profile = np.full(TOTAL_N, eta2, dtype=np.float32)
noise_profile[:N1] = eta1

print("=" * 78)
xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)
print(f"Profile={args.profile} | part={args.part} | N={TOTAL_N} wise={WISE_FRACTION*100:.0f}% (N1={N1}) "
      f"v0={v0} eta1={eta1} eta2={eta2} | steps={steps} tail={tail} seeds={n_seeds}")
print(f"Output: {out_dir}")
print("=" * 78)


def run_point(rho, interaction, k=7, label="", speed=None):
    t0 = time.time()
    series = []
    for s in range(n_seeds):
        print(f"\r >> {label} seed {s+1}/{n_seeds} ...", end="", flush=True)
        series.append(vc.simulate(xp, noise_profile, TOTAL_N, rho, r0, v0 if speed is None else speed, steps, seed=s,
                                  interaction=interaction, k_top=k))
    st = vc.steady_state_stats(series, tail, TOTAL_N)
    print(f"\r >> {label} | <Omega>={st['Omega_Avg']:.4f} ± {st['Omega_StdErr']:.4f} | "
          f"Chi={st['Chi']:.2f} | U4={st['Binder_U4']:.3f} | {time.time()-t0:.1f}s")
    return series[0], st


t_global = time.time()

# -----------------------------------------------------------------------------
# PART A: k sweep
# -----------------------------------------------------------------------------
if args.part in ("A", "both"):
    print("\n--- PART A: topological neighbour number k ---")
    rows, time_series = [], {}
    for k in K_VALUES:
        k = int(k)
        ts, st = run_point(rho_ref, "topological", k, label=f"k={k:<3}")
        time_series[k] = ts
        rows.append(dict(K=k, Rho=rho_ref, Interaction="topological", N1_Wise=N1,
                         Wise_Ratio=WISE_FRACTION, Sim_Steps=steps, Seeds=n_seeds, **st))
    dfA = pd.DataFrame(rows)
    param_text = (f"N = {TOTAL_N} | wise = {WISE_FRACTION*100:.0f}% | rho = {rho_ref} | v0 = {v0} | "
                  f"eta1 = {eta1} | eta2 = {eta2} | steps = {steps} (tail {tail}) | seeds = {n_seeds} | "
                  f"backend = {backend}")
    pngA = os.path.join(out_dir, f"{tag}_partA_k.png")
    k_peak = vr.make_sweep_figure(
        dfA, x_col="K", x_label="Topological neighbours $k$", x_fmt="{:d}",
        time_series=time_series, legend_fmt=lambda k: f"k={k}",
        title=rf"Topological Interaction: Neighbour-Number Driven Transition ({WISE_FRACTION*100:.0f}% wise)",
        param_text=param_text,
        table_cols=[("k", "K", "{}"), ("<Omega>", "Omega_Avg", "{:.3f}"), ("StdErr", "Omega_StdErr", "{:.3f}"),
                    ("Chi", "Chi", "{:.2f}"), ("U4", "Binder_U4", "{:.3f}")],
        png_path=pngA, steps=steps, tail=tail,
        ref_band=(K_STARLING_BAND[0] - 0.5, K_STARLING_BAND[1] + 0.5),
        ref_band_label="Starlings: 6-7 neighbours",
    )
    vr.export_csv(dfA, time_series, out_dir, f"{tag}_partA_k", ts_prefix="k_")
    print(f"Part A: susceptibility peak at k ~ {k_peak:g}  -> {pngA}")

# -----------------------------------------------------------------------------
# PART B: density robustness, metric vs topological
# -----------------------------------------------------------------------------
if args.part in ("B", "both"):
    print(f"\n--- PART B: density robustness, metric (r0={r0}) vs topological (k={args.k_robust}) ---")
    rows = []
    for rho in RHO_ROBUST:
        rho = float(rho)
        for inter in ("metric", "topological"):
            _, st = run_point(rho, inter, args.k_robust, label=f"rho={rho:<5} v0={v0_at(rho):.3f} {inter:<11}",
                              speed=v0_at(rho))
            rows.append(dict(Rho=rho, V0=v0_at(rho), Step_per_spacing=v0_at(rho) * np.sqrt(rho), Interaction=inter, K=args.k_robust if inter == "topological" else np.nan,
                             Mean_Neighbours_metric=np.pi * r0 ** 2 * rho, Wise_Ratio=WISE_FRACTION,
                             Sim_Steps=steps, Seeds=n_seeds, **st))
    dfB = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    plt.suptitle(rf"Robustness to Density: Metric vs Topological Interaction ({WISE_FRACTION*100:.0f}% wise)"
                 + (r" - speed scaled, $v_0\sqrt{\rho}$ fixed" if args.scale_speed else ""),
                 fontsize=14, fontweight="bold", y=0.99)
    sty = {"metric": dict(color="darkblue", marker="o", label=rf"Metric ($r_0$={r0})"),
           "topological": dict(color="crimson", marker="s", label=rf"Topological ($k$={args.k_robust})")}
    for inter in ("metric", "topological"):
        d = dfB[dfB.Interaction == inter].sort_values("Rho")
        axs[0].errorbar(d.Rho, d.Omega_Avg, yerr=d.Omega_StdErr, capsize=3, lw=2, **sty[inter])
        axs[1].plot(d.Rho, d.Chi, lw=2, **{k: v for k, v in sty[inter].items()})
        axs[2].plot(d.Rho, d.Binder_U4, lw=2, ls="--", **{k: v for k, v in sty[inter].items()})
    for ax, yl, ttl in ((axs[0], r"$\langle\Omega\rangle$", r"A) Order parameter"),
                        (axs[1], r"$\chi$", r"B) Susceptibility"),
                        (axs[2], r"$U_4$", r"C) Binder cumulant")):
        ax.axvline(rho_ref, color="darkorange", ls="-.", lw=1.2, label=r"Starling $\rho$")
        ax.set_xscale("log")
        ax.set_xticks(RHO_ROBUST)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"Density $\rho$ (log scale)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    axs[0].set_ylim(-0.05, 1.05)

    spread = dfB.groupby("Interaction").Omega_Avg.agg(lambda s: s.max() - s.min())
    fig.text(0.5, 0.01, f"N = {TOTAL_N} | v0 = {('0.4*sqrt(2.2/rho)' if args.scale_speed else v0)} | eta1 = {eta1} | eta2 = {eta2} | steps = {steps} "
             f"(tail {tail}) | seeds = {n_seeds} | backend = {backend} | range of <Omega> over rho: "
             f"metric {spread.get('metric', np.nan):.3f}, topological {spread.get('topological', np.nan):.3f}",
             ha="center", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.86, wspace=0.28)
    pngB = os.path.join(out_dir, f"{tag_B}_partB_robustness.png")
    plt.savefig(pngB, dpi=300)
    plt.close(fig)
    dfB.to_csv(os.path.join(out_dir, f"{tag_B}_partB_robustness_metrics.csv"), index=False, float_format="%.6f")
    print(f"Part B: <Omega> range over rho -> metric {spread.get('metric'):.3f}, "
          f"topological {spread.get('topological'):.3f}  -> {pngB}")

print(f"\nSIMULATION COMPLETED in {time.time()-t_global:.1f} s on {backend}\nALL DONE.")
