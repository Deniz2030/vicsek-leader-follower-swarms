# =============================================================================
# merge_finite_size.py - Combine finite-size runs of 7_finite_size.py
#
# 7_finite_size.py can be run in several sittings (e.g. N = 1000/2000/4000 one
# night, N = 8000 the next with --N_list 8000). This script reads all metrics
# CSVs, merges them (one row per N and wise ratio; a later file wins on
# duplicates), recomputes Binder crossings and susceptibility peaks, estimates
# the infinite-size transition f_c(inf) by a linear fit in 1/L, and redraws the
# 4-panel figure. No simulation is run.
#
# Usage:
#   python merge_finite_size.py                    # all data/7_finite_size_test*_metrics.csv
#   python merge_finite_size.py --profile paper
#   python merge_finite_size.py --inputs data/a_metrics.csv data/b_metrics.csv
# =============================================================================
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

p = argparse.ArgumentParser()
p.add_argument("--profile", default="test", help="profile part of the file names to merge")
p.add_argument("--inputs", nargs="+", default=None, help="explicit metrics CSV files")
p.add_argument("--data_dir", default=None, help="folder with the CSVs (default: ./data next to this script)")
args = p.parse_args()

try:
    here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    here = os.getcwd()
data_dir = args.data_dir or os.path.join(here, "data")

if args.inputs:
    files = args.inputs
else:
    files = sorted(glob.glob(os.path.join(data_dir, f"7_finite_size_{args.profile}*_metrics.csv")))
    files = [f for f in files if "_merged" not in os.path.basename(f)]
if not files:
    raise SystemExit(f"No metrics CSV found in {data_dir} for profile '{args.profile}'.")

frames = []
for f in files:
    d = pd.read_csv(f)
    d["Source_file"] = os.path.basename(f)
    frames.append(d)
    print(f"read {os.path.basename(f)}: N = {sorted(d.N.unique().tolist())}")
df = (pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["N", "Wise_Ratio"], keep="last")
        .sort_values(["N", "Wise_Ratio"]).reset_index(drop=True))
N_LIST = sorted(df.N.unique().tolist())
print(f"merged sizes: {N_LIST}")

# -----------------------------------------------------------------------------
# Binder crossings (consecutive sizes) and chi peaks
# -----------------------------------------------------------------------------
def crossing(f, ua, ub):
    """Crossing of two U4 curves; when several sign changes exist, take the one
    where the two curves are steepest together (least sensitive to noise)."""
    d = ua - ub
    best, best_slope = float("nan"), -1.0
    for i in range(1, len(d)):
        if d[i - 1] == 0 or np.sign(d[i - 1]) != np.sign(d[i]):
            fc = f[i - 1] + (f[i] - f[i - 1]) * d[i - 1] / (d[i - 1] - d[i]) if d[i - 1] != d[i] else f[i - 1]
            slope = abs(ua[i] - ua[i - 1]) + abs(ub[i] - ub[i - 1])
            if slope > best_slope:
                best, best_slope = float(fc), slope
    return best


# Smoothed U4(f): raw U4 points are noisy (+-0.03) near the transition, so raw
# curve crossings can land on noise. Each size is fitted with
#   U4(f) = a + (2/3 - a) / (1 + exp(-(f - f0)/w))
# (a: low-f plateau, f0: midpoint, w: transition width) and the fitted curves
# are crossed. The width shrinking with L (w ~ L^-x) shows the transition
# sharpening with flock size.
def sigmoid(f, a, f0, w):
    return a + (2 / 3 - a) / (1 + np.exp(-(f - f0) / w))


fits = {}
for N in N_LIST:
    d = df[df.N == N]
    try:
        p, _ = curve_fit(sigmoid, d.Wise_Ratio.values, d.Binder_U4.values, p0=[0.4, 0.08, 0.03],
                         bounds=([0.2, -0.2, 0.003], [0.66, 0.5, 0.3]))
    except (RuntimeError, ValueError):
        p = np.array([np.nan, np.nan, np.nan])
    fits[N] = p
fit_tab = pd.DataFrame([dict(N=N, L=df[df.N == N].L.iloc[0], U4_low=p[0], f_mid=p[1], width=p[2])
                        for N, p in fits.items()])


def fit_crossing(pa, pb, lo, hi):
    """Crossing of two fitted U4 curves inside [lo, hi] (the one nearest the mean midpoint)."""
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        return float("nan")
    xs = np.linspace(lo, hi, 2001)
    v = sigmoid(xs, *pa) - sigmoid(xs, *pb)
    idx = np.where(np.sign(v[1:]) != np.sign(v[:-1]))[0]
    if len(idx) == 0:
        return float("nan")
    roots = xs[idx] - v[idx] * (xs[idx + 1] - xs[idx]) / (v[idx + 1] - v[idx])
    mid = 0.5 * (pa[1] + pb[1])
    return float(roots[np.argmin(abs(roots - mid))])


rows = []
for Na, Nb in zip(N_LIST[:-1], N_LIST[1:]):
    a = df[df.N == Na].set_index("Wise_Ratio")
    b = df[df.N == Nb].set_index("Wise_Ratio")
    common = a.index.intersection(b.index).sort_values()
    fc = crossing(common.values, a.loc[common, "Binder_U4"].values, b.loc[common, "Binder_U4"].values)
    La, Lb = a.L.iloc[0], b.L.iloc[0]
    fc_fit = fit_crossing(fits[Na], fits[Nb], float(df.Wise_Ratio.min()), float(df.Wise_Ratio.max()))
    rows.append(dict(N_a=Na, N_b=Nb, L_mean=0.5 * (La + Lb), inv_L_mean=2 / (La + Lb),
                     f_cross=fc_fit, f_cross_raw=fc))
cross = pd.DataFrame(rows)

peaks = df.loc[df.groupby("N")["Chi"].idxmax(), ["N", "L", "Wise_Ratio", "Chi"]].rename(
    columns={"Wise_Ratio": "f_peak", "Chi": "Chi_max"}).reset_index(drop=True)
peaks["inv_L"] = 1 / peaks.L


def lin_extrap(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return float(intercept), float(slope)


fc_cross_inf, _ = lin_extrap(cross.inv_L_mean.values, cross.f_cross.values)
fc_peak_inf, _ = lin_extrap(peaks.inv_L.values, peaks.f_peak.values)
gamma_nu, logA = (np.polyfit(np.log(peaks.L), np.log(peaks.Chi_max), 1) if len(peaks) >= 2
                  else (float("nan"), float("nan")))

m = np.isfinite(fit_tab.width)
width_exp = (np.polyfit(np.log(fit_tab.L[m]), np.log(fit_tab.width[m]), 1)[0] if m.sum() >= 2
             else float("nan"))
print(fit_tab.to_string(index=False))
print(f"transition width w ~ L^{width_exp:.3f}  (negative = sharpens with size)")
print(cross.to_string(index=False))
print(peaks.to_string(index=False))
print(f"f_c(inf) from Binder crossings of fitted U4 (linear in 1/L): {fc_cross_inf:.4f}")
print(f"f_c(inf) from chi peaks        (linear in 1/L): {fc_peak_inf:.4f}")
# Small systems sit outside the scaling regime (N = 1000 never reaches the
# disordered plateau U4 = 1/3), so the crossings of the largest pairs are the
# most reliable direct estimate.
big = cross.f_cross.dropna().values[-2:]
fc_big = float(np.mean(big)) if len(big) else float("nan")
fc_big_spread = float(np.ptp(big) / 2) if len(big) > 1 else float("nan")
print(f"f_c from the two largest-size crossings: {fc_big:.4f} +- {fc_big_spread:.4f}")
print(f"chi_max ~ L^{gamma_nu:.3f}")

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
plt.suptitle(f"Finite-Size Scaling of the Leadership-Ratio Transition (merged, N = "
             f"{', '.join(map(str, N_LIST))})", fontsize=14, fontweight="bold", y=0.975)
cmap = plt.get_cmap("plasma")
cols = {N: cmap(i / max(1, len(N_LIST) - 1) * 0.85) for i, N in enumerate(N_LIST)}
for N in N_LIST:
    d = df[df.N == N]
    axs[0, 0].errorbar(d.Wise_Ratio, d.Omega_Avg, yerr=d.Omega_StdErr, fmt="-o", ms=4, capsize=3,
                       color=cols[N], label=f"N={N}")
    axs[0, 1].plot(d.Wise_Ratio, d.Chi, "-s", ms=4, color=cols[N], label=f"N={N}")
    axs[1, 0].plot(d.Wise_Ratio, d.Binder_U4, "d", ms=4, color=cols[N], label=f"N={N}")
    if np.all(np.isfinite(fits[N])):
        ff = np.linspace(d.Wise_Ratio.min(), d.Wise_Ratio.max(), 200)
        axs[1, 0].plot(ff, sigmoid(ff, *fits[N]), "-", lw=1.2, color=cols[N])
for ax, yl, t in ((axs[0, 0], r"$\langle\Omega\rangle$", r"A) Order parameter $\langle\Omega\rangle$"),
                  (axs[0, 1], r"$\chi$", r"B) Susceptibility $\chi = N\,\mathrm{Var}(\Omega)$"),
                  (axs[1, 0], r"$U_4$", r"C) Binder cumulant $U_4$ with fitted curves (crossings $\approx f_c$)")):
    ax.set_title(t, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Wise ratio $f = N_1/N$")
    ax.set_ylabel(yl)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=8)
axs[1, 0].axhline(2 / 3, color="gray", ls=":", lw=0.8)
axs[1, 0].axhline(1 / 3, color="gray", ls=":", lw=0.8)
for fc in cross.f_cross.dropna():
    axs[1, 0].axvline(fc, color="crimson", ls="--", lw=0.9)
if np.isfinite(fc_cross_inf):
    axs[1, 0].axvline(fc_cross_inf, color="black", ls="-.", lw=1.2, label=rf"$f_c(\infty)\approx${fc_cross_inf:.3f}")
    axs[1, 0].legend(fontsize=8)

# D: f_c(L) extrapolation to 1/L -> 0
ax = axs[1, 1]
ax.plot(cross.inv_L_mean, cross.f_cross, "o", color="crimson", ms=7, label="Binder crossings (fitted U4)")
ax.plot(peaks.inv_L, peaks.f_peak, "s", color="darkblue", ms=7, label=r"$\chi$ peaks")
xs = np.linspace(0, max(peaks.inv_L.max(), cross.inv_L_mean.max() if len(cross) else 0) * 1.1, 20)
for x, y, colr in ((cross.inv_L_mean.values, cross.f_cross.values, "crimson"),
                   (peaks.inv_L.values, peaks.f_peak.values, "darkblue")):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 2:
        s, c = np.polyfit(x[m], y[m], 1)
        ax.plot(xs, s * xs + c, ":", color=colr, lw=1.2)
ax.set_xlim(left=0)
ax.set_title(r"D) Extrapolation of $f_c(L)$ to $1/L \to 0$", fontsize=11, fontweight="bold")
ax.set_xlabel(r"$1/L$")
ax.set_ylabel(r"$f_c(L)$")
ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=8)

steps = sorted(df.Sim_Steps.unique().tolist())
seeds = sorted(df.Seeds.unique().tolist())
fig.text(0.5, 0.015,
         f"steps = {steps} | seeds = {seeds} | f_c: largest-pair crossings {fc_big:.3f}+-{fc_big_spread:.3f}, chi peaks->inf {fc_peak_inf:.3f} "
         f"(linear in 1/L) | chi_max ~ L^{gamma_nu:.2f} | width ~ L^{width_exp:.2f}\nsources: {', '.join(sorted(df.Source_file.unique()))}",
         ha="center", fontsize=8.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.92, wspace=0.28, hspace=0.32)

tag = f"7_finite_size_{args.profile}_merged_N{'-'.join(map(str, N_LIST))}"
out_dir = os.path.dirname(files[0]) or data_dir
png = os.path.join(out_dir, f"{tag}.png")
plt.savefig(png, dpi=300)
plt.close(fig)
df.drop(columns=["Source_file"]).to_csv(os.path.join(out_dir, f"{tag}_metrics.csv"), index=False, float_format="%.6f")
cross.to_csv(os.path.join(out_dir, f"{tag}_binder_crossings.csv"), index=False, float_format="%.6f")
peaks.assign(Chi_max_slope_vs_L=gamma_nu, fc_inf_binder=fc_cross_inf, fc_inf_chi=fc_peak_inf,
             fc_largest_pairs=fc_big, fc_largest_pairs_halfspread=fc_big_spread, width_exponent=width_exp).to_csv(
    os.path.join(out_dir, f"{tag}_chi_peaks.csv"), index=False, float_format="%.6f")
fit_tab.assign(width_exponent=width_exp).to_csv(os.path.join(out_dir, f"{tag}_U4_fits.csv"), index=False,
                                                float_format="%.6f")
print(f"Figure -> {png}\nCSV    -> {out_dir}/{tag}_*.csv\nALL DONE.")
