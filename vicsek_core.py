# =============================================================================
# vicsek_core.py - Shared Vicsek simulation core for the sweep series
#   1sweep_n1_ratio.py, 2_sweep_rho.py, 3_sweep_r0.py, 4_sweep_factor.py,
#   5_sweep_eta.py, 6_sweep_topological.py, ...
#
# * GPU (CuPy/CUDA) when available, NumPy fallback otherwise (same code path).
# * Neighbour search with a periodic CELL LIST -> memory O(N), scales to large N
#   (the old N x N matrix needs N^2 memory: N=50k would need ~20 GB).
# * Built-in self-test compares the cell list with a brute-force sum at start-up,
#   so a broken CUDA install or a wrong result is caught before a long run.
#
# Noise convention (kept from the original scripts):
#   theta_i(t+1) = arg( sum_{j in N_i} e^{i theta_j} ) + eta_i * xi,  xi ~ U(-pi, pi)
#   -> equivalent to the standard Vicsek eta_std = 2*pi*eta.
# =============================================================================
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")


# -----------------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------------
def _windows_cuda_dll_setup():
    """Bind CUDA DLLs shipped inside the venv (Windows pip-installed CuPy)."""
    if not sys.platform.startswith("win"):
        return
    venv = sys.prefix
    cupy_dir = os.path.join(venv, "Lib", "site-packages", "cupy")
    cuda_bin = os.path.join(venv, "Lib", "site-packages", "cupy_backends", "cuda")
    os.environ.setdefault("CUDA_PATH", cupy_dir)
    for d in (cupy_dir, cuda_bin):
        if os.path.exists(d):
            os.add_dll_directory(d)
    # nvidia-* wheels (cuda-nvrtc, cublas, ...) put DLLs under site-packages/nvidia/*/bin
    nv_root = os.path.join(venv, "Lib", "site-packages", "nvidia")
    if os.path.isdir(nv_root):
        for sub in os.listdir(nv_root):
            b = os.path.join(nv_root, sub, "bin")
            if os.path.isdir(b):
                os.add_dll_directory(b)


def get_backend(force=None, verbose=True):
    """Return (xp, name). force: None | 'cupy' | 'numpy'. Env VICSEK_BACKEND overrides."""
    choice = (force or os.environ.get("VICSEK_BACKEND", "auto")).lower()
    if choice in ("auto", "cupy", "gpu", "cuda"):
        try:
            _windows_cuda_dll_setup()
            import cupy as cp
            n_dev = cp.cuda.runtime.getDeviceCount()
            if n_dev < 1:
                raise RuntimeError("no CUDA device found")
            # small kernel to verify that the CUDA toolchain (NVRTC) really works
            _ = float((cp.arange(8, dtype=cp.float32) ** 2).sum())
            if verbose:
                props = cp.cuda.runtime.getDeviceProperties(0)
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                print(f"[backend] CuPy {cp.__version__} | GPU: {name} | "
                      f"VRAM free {free_b/2**30:.1f}/{total_b/2**30:.1f} GB")
            return cp, "cupy"
        except Exception as exc:  # noqa: BLE001
            if choice in ("cupy", "gpu", "cuda"):
                raise RuntimeError(f"CuPy/CUDA requested but not usable: {exc}") from exc
            if verbose:
                print(f"[backend] CuPy not usable ({exc}); falling back to NumPy (CPU).")
    if verbose:
        print(f"[backend] NumPy {np.__version__} (CPU)")
    return np, "numpy"


def to_host(a):
    """Device array -> NumPy array (no-op on NumPy)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def free_gpu_memory(xp):
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def _mem_budget_bytes(xp, fraction=0.35, cpu_bytes=512 * 2**20):
    """Working-memory budget for one neighbour-sum chunk."""
    if xp.__name__ == "cupy":
        free_b, _ = xp.cuda.runtime.memGetInfo()
        free_b += xp.get_default_memory_pool().free_bytes()
        return max(int(free_b * fraction), 32 * 2**20)
    return cpu_bytes


# -----------------------------------------------------------------------------
# Neighbour sums  S_i = sum_{|r_ij|<r0} exp(i theta_j)   (self included)
# -----------------------------------------------------------------------------
def neighbor_sum_dense(xp, pos, z, L, r0):
    """Brute force O(N^2) reference (used for the self-test / tiny systems)."""
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    dx -= L * xp.round(dx / L)
    dy -= L * xp.round(dy / L)
    adj = ((dx * dx + dy * dy) < (r0 * r0)).astype(z.dtype)
    return adj @ z


_OFFSETS = [(ox, oy) for oy in (-1, 0, 1) for ox in (-1, 0, 1)]


def neighbor_sum_cells(xp, pos, z, L, r0, mem_budget=None):
    """Periodic cell-list neighbour sum, fully vectorised (works on NumPy and CuPy)."""
    N = pos.shape[0]
    nc = int(np.floor(L / r0))
    if nc < 3:                      # box too small for a 3x3 stencil
        return neighbor_sum_dense(xp, pos, z, L, r0)
    cs = L / nc
    cx = xp.floor(pos[:, 0] / cs).astype(xp.int32) % nc
    cy = xp.floor(pos[:, 1] / cs).astype(xp.int32) % nc
    cid = cx + nc * cy

    order = xp.argsort(cid).astype(xp.int32)
    cid_s = cid[order]
    counts = xp.bincount(cid_s, minlength=nc * nc)
    start = xp.cumsum(counts) - counts
    rank = xp.arange(N, dtype=xp.int32) - start[cid_s].astype(xp.int32)
    max_occ = int(counts.max())

    table = xp.full((nc * nc, max_occ), -1, dtype=xp.int32)
    table[cid_s, rank] = order

    # neighbour cell ids for every particle: (N, 9)
    nb = xp.stack([((cx + ox) % nc) + nc * ((cy + oy) % nc) for ox, oy in _OFFSETS], axis=1)

    width = 9 * max_occ
    per_particle = width * 40                 # bytes per particle of temporaries (generous)
    budget = mem_budget or _mem_budget_bytes(xp)
    chunk = int(max(256, min(N, budget // per_particle)))

    S = xp.empty(N, dtype=z.dtype)
    r0sq = r0 * r0
    px, py = pos[:, 0], pos[:, 1]
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        cand = table[nb[a:b]].reshape(b - a, width)       # (n, 9*max_occ)
        valid = cand >= 0
        cidx = xp.where(valid, cand, 0)
        ddx = px[cidx] - px[a:b, None]
        ddy = py[cidx] - py[a:b, None]
        ddx -= L * xp.round(ddx / L)
        ddy -= L * xp.round(ddy / L)
        inside = valid & ((ddx * ddx + ddy * ddy) < r0sq)
        S[a:b] = (z[cidx] * inside.astype(xp.float32)).sum(axis=1)
    return S


# -----------------------------------------------------------------------------
# Topological interaction: each bird aligns with itself + its k nearest neighbours
# (Ballerini et al. 2008: starlings track ~6-7 neighbours regardless of distance)
# -----------------------------------------------------------------------------
def _cell_table(xp, pos, L, cs_target):
    N = pos.shape[0]
    nc = max(1, int(np.floor(L / cs_target)))
    cs = L / nc
    cx = xp.floor(pos[:, 0] / cs).astype(xp.int32) % nc
    cy = xp.floor(pos[:, 1] / cs).astype(xp.int32) % nc
    cid = cx + nc * cy
    order = xp.argsort(cid).astype(xp.int32)
    cid_s = cid[order]
    counts = xp.bincount(cid_s, minlength=nc * nc)
    start = xp.cumsum(counts) - counts
    rank = xp.arange(N, dtype=xp.int32) - start[cid_s].astype(xp.int32)
    max_occ = int(counts.max())
    table = xp.full((nc * nc, max_occ), -1, dtype=xp.int32)
    table[cid_s, rank] = order
    return nc, cs, cx, cy, table, max_occ


def _topo_rows_dense(xp, pos, z, L, k, rows, mem_budget):
    """Exact kNN sum for a subset of rows against all particles (fallback)."""
    N = pos.shape[0]
    out = xp.empty(rows.shape[0], dtype=z.dtype)
    chunk = int(max(1, min(rows.shape[0], mem_budget // max(1, N * 24))))
    for a in range(0, rows.shape[0], chunk):
        r = rows[a:a + chunk]
        dx = pos[None, :, 0] - pos[r, 0][:, None]
        dy = pos[None, :, 1] - pos[r, 1][:, None]
        dx -= L * xp.round(dx / L)
        dy -= L * xp.round(dy / L)
        d2 = dx * dx + dy * dy
        d2[xp.arange(r.shape[0]), r] = -1.0              # self always included
        kk = min(k, N - 1)
        sel = xp.argpartition(d2, kk, axis=1)[:, :kk + 1]
        out[a:a + chunk] = z[sel].sum(axis=1)
    return out


def neighbor_sum_topological_dense(xp, pos, z, L, k):
    rows = xp.arange(pos.shape[0], dtype=xp.int32)
    return _topo_rows_dense(xp, pos, z, L, k, rows, 256 * 2**20)


def neighbor_sum_topological(xp, pos, z, L, k, rho, mem_budget=None):
    """
    Exact k-nearest-neighbour sum with an expanding cell stencil.
    A particle's result is accepted only when its k-th neighbour lies inside the
    radius s*cs that the (2s+1)^2 stencil is guaranteed to cover; otherwise it is
    retried with a larger stencil, and finally with an all-pairs fallback.
    """
    N = pos.shape[0]
    budget = mem_budget or _mem_budget_bytes(xp)
    cs_target = float(np.sqrt(2.0 * (k + 1) / (np.pi * rho)))
    nc, cs, cx, cy, table, max_occ = _cell_table(xp, pos, L, cs_target)
    if nc < 5 or N <= k + 1:
        return neighbor_sum_topological_dense(xp, pos, z, L, k)

    S = xp.empty(N, dtype=z.dtype)
    todo = xp.arange(N, dtype=xp.int32)
    px, py = pos[:, 0], pos[:, 1]
    srad = 1
    while todo.shape[0] > 0:
        if 2 * srad + 1 > nc:
            S[todo] = _topo_rows_dense(xp, pos, z, L, k, todo, budget)
            break
        offs = [(ox, oy) for oy in range(-srad, srad + 1) for ox in range(-srad, srad + 1)]
        width = len(offs) * max_occ
        if width < k + 1:
            srad += 1
            continue
        chunk = int(max(256, min(todo.shape[0], budget // (width * 48))))
        lim = np.float32((srad * cs) ** 2)
        failed = []
        for a in range(0, todo.shape[0], chunk):
            rows = todo[a:a + chunk]
            n = rows.shape[0]
            rcx, rcy = cx[rows], cy[rows]
            nb = xp.stack([((rcx + ox) % nc) + nc * ((rcy + oy) % nc) for ox, oy in offs], axis=1)
            cand = table[nb].reshape(n, width)
            valid = cand >= 0
            cidx = xp.where(valid, cand, 0)
            ddx = px[cidx] - px[rows, None]
            ddy = py[cidx] - py[rows, None]
            ddx -= L * xp.round(ddx / L)
            ddy -= L * xp.round(ddy / L)
            d2 = xp.where(valid, ddx * ddx + ddy * ddy, xp.float32(np.inf))
            d2 = xp.where(valid & (cand == rows[:, None]), xp.float32(-1.0), d2)   # self first
            sel = xp.argpartition(d2, k, axis=1)[:, :k + 1]
            kth = xp.take_along_axis(d2, sel, axis=1).max(axis=1)
            ok = kth <= lim
            okr = rows[ok]
            if okr.shape[0] > 0:
                S[okr] = z[xp.take_along_axis(cidx, sel, axis=1)[ok]].sum(axis=1)
            bad = rows[~ok]
            if bad.shape[0] > 0:
                failed.append(bad)
        todo = xp.concatenate(failed) if failed else xp.zeros(0, dtype=xp.int32)
        srad += 1
    return S


def self_test(xp, n=1500, L=20.0, r0=1.0, seed=123, verbose=True):
    """Cell list / kNN must equal brute force (catches CUDA/driver or logic errors)."""
    rs = xp.random.RandomState(seed)
    pos = rs.uniform(0, L, size=(n, 2)).astype(xp.float32)
    # add a dense cluster and an empty region to exercise stencil expansion
    pos[: n // 5] = (pos[: n // 5] * 0.15).astype(xp.float32)
    th = rs.uniform(-np.pi, np.pi, size=n).astype(xp.float32)
    z = xp.exp(1j * th).astype(xp.complex64)
    s1 = to_host(neighbor_sum_dense(xp, pos, z, L, r0))
    s2 = to_host(neighbor_sum_cells(xp, pos, z, L, r0, mem_budget=4 * 2**20))  # force chunking
    err_m = float(np.max(np.abs(s1 - s2)))
    rho = n / L ** 2
    t1 = to_host(neighbor_sum_topological_dense(xp, pos, z, L, 7))
    t2 = to_host(neighbor_sum_topological(xp, pos, z, L, 7, rho, mem_budget=4 * 2**20))
    err_t = float(np.max(np.abs(t1 - t2)))
    ok = err_m < 1e-3 and err_t < 1e-3
    if verbose:
        print(f"[self-test] metric cell list max|dS|={err_m:.2e} | topological kNN max|dS|={err_t:.2e}"
              f" -> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("Neighbour-search self-test failed; results would be wrong.")
    return max(err_m, err_t)


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def simulate(xp, noise_profile, N, rho, r0, v0, steps, seed=0, return_state=False,
             interaction="metric", k_top=7, snapshot_every=None, snapshot_from=0):
    """
    Heterogeneous-noise Vicsek model in 2D, periodic box L = sqrt(N/rho).
    noise_profile : array (N,) with each bird's eta (same convention as above).
    interaction   : "metric" (radius r0) or "topological" (self + k_top nearest).
    snapshot_every: if set, store (pos, theta) every that many steps from
                    snapshot_from on; returned as a list after Omega(t).
    Returns Omega(t) as a NumPy float64 array (length = steps).
    """
    L = float(np.sqrt(N / rho))
    rs = xp.random.RandomState(seed)
    eta = xp.asarray(noise_profile, dtype=xp.float32)
    pos = rs.uniform(0, L, size=(N, 2)).astype(xp.float32)
    th = rs.uniform(-np.pi, np.pi, size=N).astype(xp.float32)
    omega = xp.zeros(steps, dtype=xp.float32)
    v0 = np.float32(v0)
    snaps = []

    for t in range(steps):
        z = xp.exp(1j * th).astype(xp.complex64)
        if interaction == "metric":
            S = neighbor_sum_cells(xp, pos, z, L, r0)
        elif interaction == "topological":
            S = neighbor_sum_topological(xp, pos, z, L, int(k_top), rho)
        else:
            raise ValueError(f"unknown interaction {interaction!r}")
        xi = rs.uniform(-np.pi, np.pi, size=N).astype(xp.float32)
        th = (xp.angle(S) + eta * xi).astype(xp.float32)
        pos[:, 0] = xp.mod(pos[:, 0] + xp.cos(th) * v0, L)
        pos[:, 1] = xp.mod(pos[:, 1] + xp.sin(th) * v0, L)
        omega[t] = xp.abs(xp.mean(xp.exp(1j * th)))
        if snapshot_every and t >= snapshot_from and (t - snapshot_from) % snapshot_every == 0:
            snaps.append((to_host(pos).copy(), to_host(th).copy()))

    out = to_host(omega).astype(np.float64)
    result = [out]
    if return_state:
        result += [to_host(pos), to_host(th)]
    if snapshot_every:
        result.append(snaps)
    del pos, th, eta, omega
    free_gpu_memory(xp)
    return result[0] if len(result) == 1 else tuple(result)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def steady_state_stats(series_list, tail, N):
    """
    series_list: list of Omega(t) arrays (one per independent seed).
    Uses the last `tail` steps of each run.
      Omega_avg  : mean over seeds of time averages
      Omega_err  : standard error across seeds (independent runs)
      Chi        : N * <var_t(Omega)>   (standard Vicsek susceptibility)
      Binder_U4  : 1 - <O^4> / (3 <O^2>^2) over all steady-state samples
    """
    tails = np.array([np.asarray(s)[-tail:] for s in series_list])
    run_means = tails.mean(axis=1)
    n_runs = tails.shape[0]
    omega_avg = float(run_means.mean())
    omega_err = float(run_means.std(ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else float("nan")
    chi = float(N * tails.var(axis=1).mean())
    m2 = float(np.mean(tails ** 2))
    m4 = float(np.mean(tails ** 4))
    u4 = 1.0 - m4 / (3.0 * m2 ** 2) if m2 > 0 else 0.0
    return dict(Omega_Avg=omega_avg, Omega_StdErr=omega_err, Chi=chi, Binder_U4=u4)


# -----------------------------------------------------------------------------
# Velocity-fluctuation correlation (Cavagna et al. 2010 style)
# -----------------------------------------------------------------------------
def velocity_correlation(xp, pos, th, L, dr=0.25, r_max=None, mem_budget=None):
    """
    Connected correlation of direction fluctuations
        du_i = u_i - <u>,  u_i = (cos th_i, sin th_i),  normalised so <du^2> = 1
        C(r) = sum_{i!=j} du_i . du_j  delta(r - r_ij) / sum_{i!=j} delta(r - r_ij)
    with periodic minimum-image distances up to r_max (default L/2).
    Returns (r_centres, C, pair_counts) as NumPy arrays.
    The correlation length xi is the first zero crossing of C(r).
    """
    pos = xp.asarray(pos, dtype=xp.float32)
    th = xp.asarray(th, dtype=xp.float32)
    N = pos.shape[0]
    r_max = float(r_max or L / 2.0)
    nb = int(np.ceil(r_max / dr))
    ux, uy = xp.cos(th), xp.sin(th)
    dux, duy = ux - ux.mean(), uy - uy.mean()
    norm = xp.sqrt(xp.mean(dux * dux + duy * duy))
    dux, duy = dux / norm, duy / norm

    num = xp.zeros(nb, dtype=xp.float64)
    cnt = xp.zeros(nb, dtype=xp.float64)
    budget = mem_budget or _mem_budget_bytes(xp)
    chunk = int(max(1, min(N, budget // max(1, N * 40))))
    idx_all = xp.arange(N)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        dx = pos[None, :, 0] - pos[a:b, 0][:, None]
        dy = pos[None, :, 1] - pos[a:b, 1][:, None]
        dx -= L * xp.round(dx / L)
        dy -= L * xp.round(dy / L)
        d = xp.sqrt(dx * dx + dy * dy)
        dot = dux[a:b, None] * dux[None, :] + duy[a:b, None] * duy[None, :]
        keep = (d < r_max) & (idx_all[None, :] != xp.arange(a, b)[:, None])
        bins = xp.floor(d[keep] / dr).astype(xp.int64)
        num += xp.bincount(bins, weights=dot[keep].astype(xp.float64), minlength=nb)[:nb]
        cnt += xp.bincount(bins, minlength=nb)[:nb].astype(xp.float64)
    num, cnt = to_host(num), to_host(cnt)
    C = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    r = (np.arange(nb) + 0.5) * dr
    return r, C, cnt


def zero_crossing(r, C):
    """First zero crossing of C(r) by linear interpolation (NaN if none)."""
    for i in range(1, len(C)):
        if np.isfinite(C[i - 1]) and np.isfinite(C[i]) and C[i - 1] > 0 >= C[i]:
            return float(r[i - 1] + (r[i] - r[i - 1]) * C[i - 1] / (C[i - 1] - C[i]))
    return float("nan")


# -----------------------------------------------------------------------------
# Starling (Sturnus vulgaris) reference values used to set dimensionless params
# -----------------------------------------------------------------------------
STARLING_REF = dict(
    flock_size_range=(122, 4268),      # STARFLAG field events
    nn_distance_m=(0.68, 1.51),        # mean nearest-neighbour distance
    speed_m_s=(8.1, 19.2),             # event-mean speeds, overall ~11.8 m/s
    polarization_range=(0.844, 0.992), # observed order parameter, mean ~0.97
    topological_neighbours=(6, 7),     # interaction with ~6-7 nearest neighbours
)
