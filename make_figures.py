# =============================================================================
# make_figures.py - Publication figures (MDPI Applied Sciences) from saved CSVs
#
# Reads the metrics files written by the sweep scripts in data/ and draws
# Figures 1-9 of the manuscript. No simulation is run, so the figures can be
# restyled at any time in seconds.
#   * 600 dpi PNG (journal upload) + vector PDF, 17.8 cm full-page width
#   * panel letters (a), (b), (c); no in-figure titles (captions go in the text)
#   * one y-axis per panel (Omega, chi and U4 are separate panels)
#   * colour: blue = metric / starling-like, orange = topological / critical;
#     system sizes in shades of one hue; reference values in grey
#
# Usage:
#   python make_figures.py                 # all figures, test-profile files
#   python make_figures.py --profile paper # use *_paper_* files where present
#   python make_figures.py --only 3 7      # selected figures
# Output: figures/Fig1.png, figures/Fig1.pdf, ...
# =============================================================================
import argparse
import glob
import json
import os

import matplotlib
import matplotlib.ticker
import matplotlib.ticker as mticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--profile", default="test")
p.add_argument("--data_dir", default=None)
p.add_argument("--out_dir", default=None)
p.add_argument("--only", type=int, nargs="+", default=None)
p.add_argument("--dpi", type=int, default=600)
args = p.parse_args()

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DATA = args.data_dir or os.path.join(HERE, "data")
OUT = args.out_dir or os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
PROF = args.profile

# ----------------------------------------------------------------------------- style
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 3.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
W = 7.0                                        # inches = 17.8 cm


def shades(n, color, lo=0.35, hi=1.0):
    """n shades of one hue, light -> dark (system sizes in order)."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return [tuple(1 - a * (1 - rgb)) for a in np.linspace(lo, hi, n)]


def find(pattern):
    """First existing file for a pattern; prefers the requested profile."""
    for prof in (PROF, "test", "paper"):
        hits = sorted(glob.glob(os.path.join(DATA, pattern.format(prof=prof))))
        if hits:
            return hits[0]
    return None


def critical(param):
    f = find(f"find_critical_{param}_{{prof}}_N*.json")
    if not f:
        return None
    with open(f) as fh:
        d = json.load(fh)
    return d.get("p_c"), d.get("p_c_err")


def letter(ax, s):
    ax.text(-0.16, 1.04, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=args.dpi)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def ordered_band(ax):
    ax.axhline(2 / 3, color=REF, ls=":", lw=0.8)
    ax.axhline(1 / 3, color=REF, ls=":", lw=0.8)
    ax.text(1.0, 2 / 3, " ordered", transform=ax.get_yaxis_transform(), color=INK2, fontsize=6, va="center")
    ax.text(1.0, 1 / 3, " disordered", transform=ax.get_yaxis_transform(), color=INK2, fontsize=6, va="center")


def sweep_figure(name, df, x, xlabel, refs=(), band=None, logx=False, color=BLUE):
    """Three panels: (a) <Omega> +- SE, (b) chi, (c) U4 against one control parameter.
    refs: list of (x, label, linestyle) reference lines; band: (lo, hi, label)."""
    df = df.sort_values(x)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.15))
    cols = [("Omega_Avg", r"Order parameter $\langle\Omega\rangle$"),
            ("Chi", r"Susceptibility $\chi$"),
            ("Binder_U4", r"Binder cumulant $U_4$")]
    for i, (ax, (c, yl)) in enumerate(zip(axs, cols)):
        if band:
            ax.axvspan(band[0], band[1], color=REF, alpha=0.15, lw=0)
        for ref in refs:
            ax.axvline(ref[0], color=REF, ls=ref[2], lw=0.9)
        if c == "Omega_Avg":
            ax.errorbar(df[x], df[c], yerr=df["Omega_StdErr"], color=color, marker="o", capsize=1.5,
                        elinewidth=0.7, capthick=0.7)
            ax.set_ylim(0, 1.0)
        else:
            ax.plot(df[x], df[c], color=color, marker="o")
        if c == "Binder_U4":
            ordered_band(ax)
            ax.set_ylim(0.25, 0.70)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(yl)
        letter(ax, "abc"[i])
    # reference labels once, on panel (a), clear of the data
    ytxt = 0.97
    for ref in refs:
        xv, lab = ref[0], ref[1]
        side = ref[3] if len(ref) > 3 else "top"
        if lab:
            if side == "bottom":
                axs[0].text(xv, 0.03, f" {lab}", transform=axs[0].get_xaxis_transform(), rotation=90,
                            va="bottom", ha="left", fontsize=6, color=INK2)
            else:
                left = side == "left"
                axs[0].text(xv, ytxt, f"{lab} " if left else f" {lab}", transform=axs[0].get_xaxis_transform(),
                            rotation=90, va="top", ha="right" if left else "left", fontsize=6, color=INK2)
    if band and band[2]:
        pos = band[3] if len(band) > 3 else "bottom"
        if pos == "topleft":      # just left of the band, at the top (keeps clear of data and ref labels)
            axs[0].text(band[0], 0.97, f"{band[2]} ", transform=axs[0].get_xaxis_transform(),
                        ha="right", va="top", fontsize=6, color=INK2)
        else:
            axs[0].text((band[0] * band[1]) ** 0.5 if logx else 0.5 * (band[0] + band[1]), 0.03, band[2],
                        transform=axs[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=INK2)
    fig.tight_layout(w_pad=1.6)
    save(fig, name)


figs = {}


def fig(n):
    def deco(fn):
        figs[n] = fn
        return fn
    return deco


@fig(1)
def fig1():
    f = find("5_sweep_eta_{prof}_N*_metrics.csv")
    df = pd.read_csv(f)
    refs = [(0.10, r"$\eta_1$ informed", "--", "bottom"), (0.50, r"$\eta_2$ uninformed", "--")]
    c = critical("eta")
    if c:
        refs.append((c[0], rf"$\eta_c$ = {c[0]:.3f}", "-", "left"))
    sweep_figure("Fig1", df, "Eta", r"Noise $\eta$", refs=refs)


@fig(2)
def fig2():
    df = pd.read_csv(find("1sweep_n1_ratio_{prof}_N*_metrics.csv"))
    sweep_figure("Fig2", df, "Wise_Ratio", r"Informed fraction $f = N_1/N$")


@fig(3)
def fig3():
    df = pd.read_csv(find("2_sweep_rho_{prof}_N*_metrics.csv"))
    sweep_figure("Fig3", df, "Rho", r"Density $\rho$ (agents per $r_0^2$)", refs=[(2.2, "starling", "--")])


@fig(4)
def fig4():
    df = pd.read_csv(find("3_sweep_r0_{prof}_N*_metrics.csv"))
    sweep_figure("Fig4", df, "R0", r"Perception range $r_0$", refs=[(1.0, "starling", "--")])


@fig(5)
def fig5():
    df = pd.read_csv(find("4_sweep_factor_{prof}_N*_metrics.csv"))
    sweep_figure("Fig5", df, "Factor", r"Speed $v_0\Delta t/r_0$", refs=[(0.4, "starling mean", "--")],
                 band=(8.1 / 29.5, 19.2 / 29.5, "observed"), logx=True)


@fig(6)
def fig6():
    df = pd.read_csv(find("6_sweep_topological_{prof}_N*_partA_k_metrics.csv"))
    refs = []
    c = critical("k")
    if c:
        refs.append((c[0], rf"$k_c$ = {c[0]:.1f}", "-"))
    sweep_figure("Fig6", df, "K", r"Topological neighbours $k$", refs=refs, band=(5.5, 7.5, "starling", "topleft"),
                 color=ORANGE)


@fig(7)
def fig7():
    fixed = pd.read_csv(find("6_sweep_topological_{prof}_N*_partB_robustness_metrics.csv"))
    scaled_f = find("6_sweep_topological_{prof}_N*_scaledv_partB_robustness_metrics.csv")
    panels = [(fixed, r"Fixed speed, $v_0\Delta t/r_0 = 0.4$")]
    if scaled_f:
        panels.append((pd.read_csv(scaled_f), r"Scaled speed, $v_0\sqrt{\rho}$ constant"))
    fig, axs = plt.subplots(1, len(panels), figsize=(W * len(panels) / 3 * 1.25, 2.3), sharey=True)
    axs = np.atleast_1d(axs)
    for i, (ax, (d, sub)) in enumerate(zip(axs, panels)):
        d = d[~d.Rho.isna()]
        ax.axvline(2.2, color=REF, ls="--", lw=0.9)
        for inter, colr, lab, mk in (("metric", BLUE, r"metric ($r_0 = 1$)", "o"),
                                     ("topological", ORANGE, r"topological ($k = 7$)", "s")):
            s = d[d.Interaction == inter].sort_values("Rho")
            ax.errorbar(s.Rho, s.Omega_Avg, yerr=s.Omega_StdErr, color=colr, marker=mk, capsize=1.5,
                        elinewidth=0.7, capthick=0.7, label=lab)
            rng = s.Omega_Avg.max() - s.Omega_Avg.min()
            ax.annotate(f"range {rng:.3f}", (s.Rho.iloc[-1], s.Omega_Avg.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=6, color=colr, va="center")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 4])
        ax.set_xticklabels(["0.5", "1", "2", "4"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"Density $\rho$")
        ax.set_ylim(0, 0.7)
        ax.set_title(sub, fontsize=7.5, color=INK, loc="left")
        letter(ax, "abc"[i])
    axs[0].set_ylabel(r"Order parameter $\langle\Omega\rangle$")
    axs[0].text(2.2, 0.97, " starling", transform=axs[0].get_xaxis_transform(), rotation=90, va="top",
                fontsize=6, color=INK2)
    axs[-1].legend(loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "Fig7")


@fig(8)
def fig8():
    from scipy.optimize import curve_fit
    f = find("7_finite_size_{prof}_merged_*_metrics.csv") or find("7_finite_size_{prof}_metrics.csv")
    df = pd.read_csv(f)
    Ns = sorted(df.N.unique())
    cols = shades(len(Ns), BLUE)

    def sig(x, a, x0, w):          # smoothed U4(f), same form as merge_finite_size.py
        return a + (2 / 3 - a) / (1 + np.exp(-(x - x0) / w))

    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2))
    rows = []
    for N, colr in zip(Ns, cols):
        d = df[df.N == N].sort_values("Wise_Ratio")
        axs[0].plot(d.Wise_Ratio, d.Binder_U4, color=colr, marker="o", ls="none", label=f"N = {N:,}")
        try:
            p, _ = curve_fit(sig, d.Wise_Ratio, d.Binder_U4, p0=[0.4, 0.08, 0.03],
                             bounds=([0.2, -0.2, 0.003], [0.66, 0.5, 0.3]))
            ff = np.linspace(d.Wise_Ratio.min(), d.Wise_Ratio.max(), 200)
            axs[0].plot(ff, sig(ff, *p), color=colr, lw=1.0)
            rows.append((d.L.iloc[0], p[2]))
        except (RuntimeError, ValueError):
            pass
        axs[1].plot(d.Wise_Ratio, d.Chi, color=colr, marker="o", label=f"N = {N:,}")
    ordered_band(axs[0])
    axs[0].set_ylim(0.3, 0.70)
    axs[0].set_ylabel(r"Binder cumulant $U_4$")
    axs[1].set_ylabel(r"Susceptibility $\chi$")
    for ax in axs[:2]:
        ax.set_xlabel(r"Informed fraction $f$")
    axs[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))

    # (c) growth of the susceptibility peak with system size (log-log)
    ax = axs[2]
    peaks = df.loc[df.groupby("N").Chi.idxmax()].sort_values("L")
    L, cm = peaks.L.values, peaks.Chi.values
    ax.loglog(L, cm, marker="o", ls="none", color=BLUE,
              label=r"$\chi_{\max}$")
    if len(L) >= 2:
        k, c0 = np.polyfit(np.log(L), np.log(cm), 1)
        xx = np.linspace(L.min() * 0.9, L.max() * 1.1, 20)
        ax.loglog(xx, np.exp(c0) * xx ** k, color=BLUE, ls=":", lw=1.0,
                  label=rf"fit $\propto L^{{{k:.2f}}}$")
    for r in peaks.itertuples():
        ax.annotate(f"f = {r.Wise_Ratio:.2f}", (r.L, r.Chi), xytext=(4, -9), textcoords="offset points",
                    fontsize=6, color=INK2)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([20, 30, 40, 60])
    ax.set_yticks([10, 20, 40])
    ax.set_xlim(17, 75)
    ax.set_ylim(7, 60)
    ax.set_xlabel(r"System size $L=\sqrt{N/\rho}$")
    ax.set_ylabel(r"Peak susceptibility $\chi_{\max}$")
    ax.legend(loc="upper left")
    for i, a_ in enumerate(axs):
        letter(a_, "abc"[i])
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig8")


@fig(9)
def fig9():
    cr = pd.read_csv(find("8_correlation_starling_{prof}_Cr.csv"))
    xi = pd.read_csv(find("8_correlation_starling_{prof}_xi.csv"))
    Ns = sorted(xi.N.unique())
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2))
    for reg, base, ls in (("starling", BLUE, "-"), ("critical", ORANGE, "--")):
        for N, colr in zip(Ns, shades(len(Ns), base)):
            col = f"C_{reg}_N{N}"
            if col not in cr:
                continue
            m = cr[col].notna()
            L = xi[(xi.Regime == reg) & (xi.N == N)].L.iloc[0]
            axs[0].plot(cr.r[m], cr[col][m], color=colr, ls=ls, lw=1.1)
            axs[1].plot(cr.r[m] / L, cr[col][m], color=colr, ls=ls, lw=1.1)
    for ax in axs[:2]:
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel(r"Correlation $C(r)$")
    axs[0].set_xlabel(r"Distance $r$ (units of $r_0$)")
    axs[1].set_xlabel(r"Rescaled distance $r/L$")
    # legend: regime by colour/line, size by shade
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, ls="-", label=r"starling-like ($\Omega \approx 0.97$)"),
         Line2D([], [], color=ORANGE, ls="--", label="critical mixture")]
    h += [Line2D([], [], color=c, label=f"N = {N:,}") for N, c in zip(Ns, shades(len(Ns), INK2))]
    axs[0].legend(handles=h, loc="upper right", fontsize=6)
    ax = axs[2]
    for reg, colr, mk, lab in (("starling", BLUE, "o", "starling-like"), ("critical", ORANGE, "s", "critical mixture")):
        d = xi[xi.Regime == reg].sort_values("L")
        k, c0 = np.polyfit(d.L, d.Xi_from_mean_C, 1)
        ax.errorbar(d.L, d.Xi_from_mean_C, yerr=d.Xi_std, color=colr, marker=mk, ls="none", capsize=1.5,
                    elinewidth=0.7, capthick=0.7, label=rf"{lab}: $\xi \approx {k:.2f}\,L {c0:+.1f}$")
        xx = np.linspace(0, d.L.max() * 1.05, 20)
        ax.plot(xx, k * xx + c0, color=colr, ls=":", lw=1.0)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(r"System size $L = \sqrt{N/\rho}$")
    ax.set_ylabel(r"Correlation length $\xi$")
    ax.legend(loc="upper left", fontsize=6)
    for i, a in enumerate(axs):
        letter(a, "abc"[i])
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig9")


print(f"data: {DATA}\nout:  {OUT}")
for n in sorted(figs):
    if args.only and n not in args.only:
        continue
    try:
        figs[n]()
    except (TypeError, FileNotFoundError, ValueError) as e:
        print(f"  Fig{n}: skipped ({e})")
print("ALL DONE.")
