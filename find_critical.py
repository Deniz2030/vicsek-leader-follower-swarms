# =============================================================================
# find_critical.py - Automatic location of the critical point (Binder bisection)
#
# Algorithm (single flock size N):
#   The Binder cumulant U4 = 1 - <O^4>/(3<O^2>^2) of the 2D Vicsek order
#   parameter tends to 2/3 in the ordered phase and 1/3 in the disordered
#   phase. The transition point p_c of a control parameter p is defined as the
#   value where U4 crosses the midpoint U* = 1/2. Because U4 is monotonic in p
#   across the transition, p_c is found by bisection:
#
#     1. Evaluate U4 at the bracket ends p_lo, p_hi (must lie on opposite
#        sides of U*; otherwise the bracket is widened).
#     2. Repeat until p_hi - p_lo <= tol:
#          p_m = (p_lo + p_hi)/2 ; evaluate U4(p_m) with S seeds
#          if |U4(p_m) - U*| < 2*SE(U4): add seeds (adaptive, up to max_seeds)
#          keep the half-interval whose ends straddle U*
#     3. p_c = linear interpolation of U4 between the final bracket ends;
#        uncertainty from the bracket width and the bootstrap SE of U4.
#   Cost: 2 + ceil(log2((p_hi - p_lo)/tol)) parameter evaluations.
#
# The result is a finite-size estimate p_c(N); 7_finite_size.py extrapolates
# to N -> infinity.
#
# Usage (starling calibration by default):
#   python find_critical.py --param eta  --lo 0.30 --hi 0.70 --tol 0.01
#   python find_critical.py --param wise --lo 0.00 --hi 0.40 --tol 0.01
#   python find_critical.py --param rho  --lo 0.5  --hi 6.0  --tol 0.05
#   python find_critical.py --param r0 | factor | k   (k = topological neighbours)
#   add --steps / --tail / --seeds / --N / --profile as in the other scripts
# =============================================================================
import json
import os
import time

import numpy as np

import vicsek_core as vc
import vicsek_report as vr

U_STAR = 0.5   # midpoint between ordered (2/3) and disordered (1/3) limits


def extra(p):
    p.add_argument("--param", choices=["eta", "wise", "rho", "r0", "factor", "k"], required=True)
    p.add_argument("--lo", type=float, required=True)
    p.add_argument("--hi", type=float, required=True)
    p.add_argument("--tol", type=float, default=0.01, help="bracket width at which to stop")
    p.add_argument("--max_seeds", type=int, default=8, help="adaptive seed cap near U*")
    p.add_argument("--wise_fixed", type=float, default=0.10, help="wise fraction when not swept")
    p.add_argument("--max_iter", type=int, default=20)


args, prof = vr.parse_args(default_N=3000, extra=extra)

# -----------------------------------------------------------------------------
# Starling calibration (defaults for every parameter that is not swept)
# -----------------------------------------------------------------------------
BASE = dict(N=args.N, rho=2.2, r0=1.0, factor=0.4, eta1=0.10, eta2=0.50,
            wise=args.wise_fixed, k=7)
steps, tail, seeds0 = prof["steps"], prof["tail"], prof["seeds"]
INTEGER = args.param == "k"

xp, backend = vc.get_backend(args.backend)
vc.self_test(xp)


def config(p):
    c = dict(BASE)
    if args.param == "eta":          # homogeneous flock, every bird has noise eta
        c.update(eta1=p, eta2=p, wise=1.0)
    elif args.param == "wise":
        c["wise"] = p
    elif args.param in ("rho", "r0", "factor"):
        c[args.param] = p
    elif args.param == "k":
        c["k"] = int(round(p))
    return c


cache = {}   # p -> list of Omega(t) series (one per seed)


def binder(p, n_seeds):
    """U4(p) and bootstrap SE using n_seeds independent runs (cached, extended on demand)."""
    c = config(p)
    N = c["N"]
    N1 = int(round(c["wise"] * N))
    noise = np.full(N, c["eta2"], dtype=np.float32)
    noise[:N1] = c["eta1"]
    inter = "topological" if args.param == "k" else "metric"
    runs = cache.setdefault(p, [])
    while len(runs) < n_seeds:
        s = len(runs)
        runs.append(vc.simulate(xp, noise, N, c["rho"], c["r0"], c["factor"], steps,
                                seed=s, interaction=inter, k_top=c["k"]))
    tails = [r[-tail:] for r in runs[:n_seeds]]
    st = vc.steady_state_stats(tails, tail, N)
    # bootstrap over blocks (5 blocks per run) -> SE of U4
    blocks = np.concatenate([np.array_split(t, 5) for t in tails])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        sel = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        m2, m4 = np.mean(sel ** 2), np.mean(sel ** 4)
        boots.append(1 - m4 / (3 * m2 ** 2))
    return st["Binder_U4"], float(np.std(boots)), st


def evaluate(p, log):
    t0 = time.time()
    n = seeds0
    u, se, st = binder(p, n)
    while abs(u - U_STAR) < 2 * se and n < args.max_seeds:   # adaptive: too close to call
        n += 1
        u, se, st = binder(p, n)
    side = "ordered" if u > U_STAR else "disordered"
    log.append(dict(p=p, U4=u, U4_SE=se, seeds=n, Omega=st["Omega_Avg"], Chi=st["Chi"], side=side))
    print(f" >> {args.param}={p:<8.4g} U4={u:.3f} ± {se:.3f} ({n} seeds) <Omega>={st['Omega_Avg']:.3f}"
          f" Chi={st['Chi']:.2f} -> {side} | {time.time()-t0:.1f}s")
    return u


print("=" * 78)
print(f"Binder bisection | param={args.param} bracket=[{args.lo}, {args.hi}] tol={args.tol} | U*={U_STAR}")
print(f"Fixed: {BASE} | steps={steps} tail={tail} seeds={seeds0} (max {args.max_seeds}) | backend={backend}")
print("=" * 78)

log = []
t_global = time.time()
lo, hi = float(args.lo), float(args.hi)
if INTEGER:
    lo, hi = float(round(lo)), float(round(hi))
u_lo, u_hi = evaluate(lo, log), evaluate(hi, log)
if (u_lo - U_STAR) * (u_hi - U_STAR) > 0:
    raise SystemExit(f"Bracket does not straddle U*={U_STAR}: U4({lo})={u_lo:.3f}, U4({hi})={u_hi:.3f}. "
                     f"Widen --lo/--hi.")

it = 0
while (hi - lo) > args.tol and it < args.max_iter:
    mid = (lo + hi) / 2
    if INTEGER:
        mid = float(round(mid))
        if mid in (lo, hi):
            break
    u_mid = evaluate(mid, log)
    if (u_mid - U_STAR) * (u_lo - U_STAR) > 0:
        lo, u_lo = mid, u_mid
    else:
        hi, u_hi = mid, u_mid
    it += 1

# final estimate: linear interpolation of U4 between the bracket ends
p_c = lo + (hi - lo) * (u_lo - U_STAR) / (u_lo - u_hi) if u_lo != u_hi else (lo + hi) / 2
se_lo = next(r["U4_SE"] for r in log if r["p"] == lo)
se_hi = next(r["U4_SE"] for r in log if r["p"] == hi)
slope = abs(u_hi - u_lo) / max(hi - lo, 1e-12)
err = 0.5 * (hi - lo) + (np.hypot(se_lo, se_hi) / slope if slope > 0 else 0.0)
err = min(err, hi - lo)

result = dict(param=args.param, p_c=p_c, p_c_err=err, bracket=[lo, hi], evaluations=len(log),
              fixed=BASE, steps=steps, tail=tail, backend=backend,
              runtime_s=round(time.time() - t_global, 1), log=log)
if args.param == "eta":
    result["eta_c_standard"] = 2 * np.pi * p_c

out_dir = vr.output_dir_for(__file__)
path = os.path.join(out_dir, f"find_critical_{args.param}_{args.profile}_N{args.N}.json")
with open(path, "w") as f:
    json.dump(result, f, indent=2, default=float)

print("-" * 78)
print(f"p_c({args.param}) = {p_c:.4f} ± {err:.4f}   bracket [{lo:.4f}, {hi:.4f}]   "
      f"{len(log)} evaluations, {result['runtime_s']} s")
if args.param == "eta":
    print(f"   eta_c (standard Vicsek, 2*pi*eta) = {2*np.pi*p_c:.3f}")
print(f"Log -> {path}\nALL DONE.")
