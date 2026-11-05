# =============================================================================
# vicsek_report.py - Shared figure / CSV export / CLI helpers for the sweep series
# =============================================================================
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vicsek_core as vc

PROFILES = {
    #             steps   tail   seeds
    "smoke": dict(steps=60, tail=30, seeds=2),       # pipeline check only
    "test":  dict(steps=3000, tail=1500, seeds=3),
    "paper": dict(steps=10000, tail=5000, seeds=5),
}


def parse_args(default_N=3000, extra=None):
    """Common CLI: --profile, --N, --steps, --tail, --seeds, --backend."""
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(PROFILES), default="test")
    p.add_argument("--N", type=int, default=default_N, help="total flock size")
    p.add_argument("--steps", type=int, default=None, help="override profile steps")
    p.add_argument("--tail", type=int, default=None, help="override steady-state window")
    p.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    p.add_argument("--backend", choices=["auto", "cupy", "numpy"], default=None)
    if extra:
        extra(p)
    a, _ = p.parse_known_args()
    prof = dict(PROFILES[a.profile])
    for k in ("steps", "tail", "seeds"):
        if getattr(a, k) is not None:
            prof[k] = getattr(a, k)
    prof["tail"] = min(prof["tail"], prof["steps"])
    return a, prof


def output_dir_for(script_file):
    try:
        d = os.path.dirname(os.path.abspath(script_file))
    except (NameError, TypeError):
        d = os.getcwd()
    out = os.path.join(d, "data")
    os.makedirs(out, exist_ok=True)
    return out


def make_sweep_figure(df, x_col, x_label, x_fmt, time_series, legend_fmt, title, param_text,
                      table_cols, png_path, steps, tail, ref_vline=None, ref_vline_label=None,
                      log_x=False, ref_band=None, ref_band_label=None, extra_vlines=None):
    """
    Standard 4-panel figure used by every sweep:
      A) Omega(t) for seed 0   B) steady-state window
      C) <Omega> vs x with observed-starling band   D) Chi and Binder U4 vs x
    table_cols: list of (header, df_column, format) for the embedded table.
    """
    i_peak = int(df["Chi"].idxmax())
    x_peak = float(df.loc[i_peak, x_col])

    fig, axs = plt.subplots(2, 2, figsize=(12, 16))
    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
    cmap = plt.get_cmap("viridis")
    keys = list(time_series)
    col = lambda k: cmap(k / max(1, len(keys) - 1))  # noqa: E731

    # A
    for k, key in enumerate(keys):
        axs[0, 0].plot(time_series[key], color=col(k), lw=0.8, alpha=0.85, label=legend_fmt(key))
    axs[0, 0].axvline(steps - tail, color="k", ls=":", lw=1)
    axs[0, 0].set_title(r"A) Time Evolution of $\Omega(t)$ (seed 0)", fontsize=11, fontweight="bold", pad=12)
    axs[0, 0].set_xlabel("Time step $t$")
    axs[0, 0].set_ylabel(r"$\Omega(t)$")
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].grid(True, ls="--", alpha=0.5)
    axs[0, 0].legend(fontsize=6.5, ncol=3, loc="lower right", framealpha=0.6)

    # B
    show = min(500, tail)
    t_tail = np.arange(steps - show, steps)
    for k, key in enumerate(keys):
        axs[0, 1].plot(t_tail, time_series[key][-show:], color=col(k), lw=0.9)
    axs[0, 1].set_title(f"B) Steady-State Fluctuations (last {show} steps)", fontsize=11, fontweight="bold", pad=12)
    axs[0, 1].set_xlabel("Time step $t$")
    axs[0, 1].set_ylabel(r"$\Omega(t)$")
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].grid(True, ls="--", alpha=0.5)

    # C
    lo, hi = vc.STARLING_REF["polarization_range"]
    ax = axs[1, 0]
    ax.axhspan(lo, hi, color="orange", alpha=0.18, label="Observed starlings")
    ax.errorbar(df[x_col], df["Omega_Avg"], yerr=df["Omega_StdErr"], fmt="-o", color="darkblue",
                ecolor="black", capsize=4, lw=2, label=r"$\langle\Omega\rangle$")
    ax.axvline(x_peak, color="crimson", ls="--", lw=1, label=r"$\chi$ peak")
    if ref_band is not None:
        ax.axvspan(ref_band[0], ref_band[1], color="green", alpha=0.10, label=ref_band_label)
    if ref_vline is not None:
        ax.axvline(ref_vline, color="darkorange", ls="-.", lw=1.3, label=ref_vline_label)
    for xv, lab, colr in (extra_vlines or []):
        ax.axvline(xv, color=colr, ls=":", lw=1.4, label=lab)
    ax.set_title(r"C) Stationary Order Parameter $\langle\Omega\rangle$", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\langle\Omega\rangle$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=8, loc="best")

    # D
    ax_c = axs[1, 1]
    ax_u = ax_c.twinx()
    p1, = ax_c.plot(df[x_col], df["Chi"], "-s", color="crimson", lw=2, label=r"$\chi = N\,\mathrm{Var}(\Omega)$")
    p2, = ax_u.plot(df[x_col], df["Binder_U4"], "--d", color="purple", lw=1.5, label=r"Binder $U_4$")
    ax_u.axhline(2 / 3, color="purple", ls=":", lw=0.8)
    ax_u.axhline(1 / 3, color="purple", ls=":", lw=0.8)
    if ref_band is not None:
        ax_c.axvspan(ref_band[0], ref_band[1], color="green", alpha=0.10)
    if ref_vline is not None:
        ax_c.axvline(ref_vline, color="darkorange", ls="-.", lw=1.3)
    for xv, _, colr in (extra_vlines or []):
        ax_c.axvline(xv, color=colr, ls=":", lw=1.4)
    ax_c.set_title(r"D) Susceptibility $\chi$ & Binder Cumulant $U_4$", fontsize=11, fontweight="bold", pad=12)
    ax_c.set_xlabel(x_label)
    ax_c.set_ylabel(r"$\chi$", color="crimson")
    ax_u.set_ylabel(r"$U_4$  (1/3 disordered, 2/3 ordered)", color="purple")
    ax_c.grid(True, ls="--", alpha=0.5)
    ax_c.legend(handles=[p1, p2], loc="center right", fontsize=8)

    if log_x:
        for a in (axs[1, 0], ax_c):
            a.set_xscale("log")

    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.34, top=0.935, wspace=0.32, hspace=0.30)
    fig.text(0.5, 0.295, param_text, ha="center", va="center", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

    rows = [[fmt.format(getattr(r, c)) for _, c, fmt in table_cols] for r in df.itertuples()]
    ax_t = fig.add_axes([0.08, 0.01, 0.84, 0.26])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=[h for h, _, _ in table_cols],
                     loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    return x_peak


def export_csv(df, time_series, out_dir, tag, ts_prefix):
    m = os.path.join(out_dir, f"{tag}_metrics.csv")
    df.to_csv(m, index=False, float_format="%.6f")
    t = os.path.join(out_dir, f"{tag}_timeseries.csv")
    pd.DataFrame({f"{ts_prefix}{k}": v for k, v in time_series.items()}).rename_axis("Step").to_csv(
        t, float_format="%.6f")
    return m, t
