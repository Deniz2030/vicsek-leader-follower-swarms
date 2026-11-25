# =============================================================================
# run_sensitivity.py - Sensitivity of Algorithm 1 (Binder bisection) to its own
# settings, for the noise transition of the homogeneous flock (Reviewer 1, C5).
#
# Runs find_critical.py once per setting, one after the other, and then writes
# a summary table. Each run saves its own JSON in data/, so if the run is
# interrupted, starting this script again skips the settings already done.
#
# Usage:   python run_sensitivity.py            (about 1.5-2 h on an RTX 4060)
#          python run_sensitivity.py --summary  (only rebuild the table)
# =============================================================================
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = ["--param", "eta", "--lo", "0.30", "--hi", "0.70"]

# tag -> extra arguments (baseline: U*=0.5, tol=0.01, seeds=3, max_seeds=8, 3000 steps)
RUNS = {
    "ustar045": ["--u_star", "0.45"],
    "ustar055": ["--u_star", "0.55"],
    "tol002":   ["--tol", "0.02"],
    "tol0005":  ["--tol", "0.005"],
    "smax3":    ["--max_seeds", "3"],          # no adaptive sampling
    "smax12":   ["--max_seeds", "12"],
    "steps6000": ["--steps", "6000", "--tail", "3000"],
}


def json_path(tag):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(DATA, f"find_critical_eta_test_N3000{suffix}.json")


def run_all():
    for tag, extra in RUNS.items():
        if os.path.exists(json_path(tag)):
            print(f"[skip] {tag}: result already in data/")
            continue
        cmd = [sys.executable, os.path.join(HERE, "find_critical.py")] + BASE + extra + ["--tag", tag]
        print("\n" + "=" * 78 + f"\n[run] {tag}: {' '.join(cmd[1:])}\n" + "=" * 78, flush=True)
        subprocess.run(cmd, check=True, cwd=HERE)


def summary():
    rows = []
    for tag in [""] + list(RUNS):
        p = json_path(tag)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        rows.append(dict(
            setting=tag or "baseline",
            u_star=d.get("u_star", 0.5), tol=d.get("tol", 0.01),
            seeds=d.get("seeds", 3), max_seeds=d.get("max_seeds", 8), steps=d.get("steps", 3000),
            p_c=d["p_c"], p_c_err=d["p_c_err"], evaluations=d["evaluations"],
            seeds_used=d.get("seeds_used", sum(r.get("seeds", 0) for r in d.get("log", []))),
            runtime_min=round(d.get("runtime_s", float("nan")) / 60, 1)))
    if not rows:
        print("No results found in data/.")
        return
    keys = list(rows[0])
    out = os.path.join(DATA, "sensitivity_algorithm1_eta.csv")
    with open(out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"\n{'setting':<10} {'U*':>5} {'tol':>6} {'Smax':>4} {'steps':>6} {'eta_c':>8} {'err':>7} "
          f"{'evals':>5} {'seeds':>5} {'min':>6}")
    for r in rows:
        print(f"{r['setting']:<10} {r['u_star']:>5.2f} {r['tol']:>6.3f} {r['max_seeds']:>4} {r['steps']:>6} "
              f"{r['p_c']:>8.4f} {r['p_c_err']:>7.4f} {r['evaluations']:>5} {r['seeds_used']:>5} "
              f"{r['runtime_min']:>6}")
    print(f"\nTable -> {out}\nALL DONE.")


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        run_all()
    summary()
