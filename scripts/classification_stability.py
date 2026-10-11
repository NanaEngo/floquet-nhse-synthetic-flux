#!/usr/bin/env python3
"""
Track C audit C-P1-11: seed-dependence of the deep/marginal solution classes.

For each lattice size N in the published ladder, rerun the production DOC
protocol (identical hyperparameters: 500 Adam steps, lr 0.003, stability
barrier max Re lambda < -0.05) from 10 independent random initializations,
classify every converged solution (F_C^norm >= 0.5 -> deep_stability, else
marginal_stability; unstable if the final spectral margin fails), and build
the confusion matrix against the published classes.

Model conventions are byte-identical to scripts/doc_nscaling_hpc.py (kappa=1,
theta0=0.1, d-theta probe, record Fisher information normalized by ||mu||^2).
Before optimizing, the script re-evaluates the published optima through the
model as a self-check and aborts on mismatch.

Results checkpoint to data/classification_stability.json after each N
(interrupt/resume safe); log mirrors stdout.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import json
import sys
import time
import math
import argparse

import jax
import jax.numpy as jnp
from jax import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
REF_JSON = os.path.join(DATA_DIR, "n_scaling_results_v3.json")
ENRICHED_JSON = os.path.join(DATA_DIR, "n_scaling_results_v3_enriched.json")
OUT_JSON = os.path.join(DATA_DIR, "classification_stability.json")

KAPPA = 1.0
THETA0 = 0.1
DTHETA = 1e-4
Ns = [6, 8, 10, 12, 14, 16]
SEEDS_PER_N = 10
N_OPTIM = 500          # ponytail: production protocol kept identical on purpose
LR = 0.003
DEEP_THRESHOLD = 0.5
BASE_SEED = 20260823   # audit run; distinct from production seed 42

# Bounds must match doc_nscaling_hpc.py exactly (self-check inverts unmap).
J_LO, J_HI = 0.05, 0.5
g_LO, g_HI = 0.5, 3.0
L_LO, L_HI = 0.005, 0.1
PHI_LO, PHI_HI = 0.0, 6.2832


def unmap(x):
    """Map unconstrained optimizer variables to physical parameters."""
    def sig(z):
        """Logistic sigmoid mapping a real scalar to (0, 1).

        Args:
            z: Real scalar, dimensionless.

        Returns:
            Sigmoid value in (0, 1), dimensionless.
        """
        return 1.0 / (1.0 + jnp.exp(-z))
    return jnp.array([
        J_LO + (J_HI - J_LO) * sig(x[0]),
        g_LO + (g_HI - g_LO) * sig(x[1]),
        L_LO + (L_HI - L_LO) * sig(x[2]),
        PHI_LO + (PHI_HI - PHI_LO) * sig(x[3]),
    ])


def map_back(p):
    """Invert unmap: physical parameters -> unconstrained variables."""
    def logit(lo, hi, v):
        """Map a bounded physical value back to an unconstrained variable.

        Args:
            lo: Lower box bound (same unit as v).
            hi: Upper box bound (same unit as v).
            v: Physical value inside (lo, hi).

        Returns:
            Unconstrained logit value, dimensionless.
        """
        u = min(max((v - lo) / (hi - lo), 1e-9), 1.0 - 1e-9)
        return math.log(u / (1.0 - u))
    return [
        logit(J_LO, J_HI, p[0]),
        logit(g_LO, g_HI, p[1]),
        logit(L_LO, L_HI, p[2]),
        logit(PHI_LO, PHI_HI, p[3]),
    ]


def make_drift(n, J, g, lam, phi, theta):
    """Build the 2n x 2n non-Hermitian double-chain drift matrix."""
    A = -KAPPA * jnp.eye(n, dtype=jnp.complex128)
    A = A.at[n - 1, n - 1].add(-1j * theta)
    for j in range(n - 1):
        A = A.at[j + 1, j].add(J * jnp.exp(g))
        A = A.at[j, j + 1].add(J * jnp.exp(-g))
    B = -KAPPA * jnp.eye(n, dtype=jnp.complex128)
    for j in range(n - 1):
        B = B.at[j + 1, j].add(J * jnp.exp(-g))
        B = B.at[j, j + 1].add(J * jnp.exp(g))
    L = lam * jnp.exp(1j * phi) * jnp.eye(n, dtype=jnp.complex128)
    return jnp.block([[A, L], [L.conj().T, B]])


def realify(A):
    """Embed a complex matrix into a real matrix of doubled dimension."""
    return jnp.block([[A.real, -A.imag], [A.imag, A.real]])


def solve_lyap(Ar):
    """Solve the continuous-time Lyapunov equation for the covariance."""
    m = Ar.shape[0]
    Lop = jnp.kron(jnp.eye(m), Ar) + jnp.kron(Ar, jnp.eye(m))
    return jnp.linalg.solve(Lop, -(2 * KAPPA * jnp.eye(m)).flatten()).reshape(m, m)


def compute_all(x, N):
    """Evaluate record Fisher information metrics for parameters x."""
    p = unmap(x)
    J, g, lam, phi = p
    n = int(N)
    A0 = make_drift(n, J, g, lam, phi, THETA0)
    Ar0 = realify(A0)
    max_re = jnp.max(jnp.linalg.eigvals(A0).real)
    Sig = solve_lyap(Ar0)
    nn = Ar0.shape[0]
    u = jnp.zeros(nn).at[2 * (n - 1)].set(1.0)
    mu = -jnp.linalg.solve(Ar0, u)
    norm2 = jnp.sum(jnp.abs(mu) ** 2)
    dAr = jnp.zeros((nn, nn))
    idx = 2 * (n - 1)
    dAr = dAr.at[idx, idx + 1].set(-1.0)
    dAr = dAr.at[idx + 1, idx].set(1.0)
    dmu = -jnp.linalg.solve(Ar0, dAr @ jnp.linalg.solve(Ar0, u))
    Sig_inv = jnp.linalg.inv(Sig)
    F_mean = dmu @ Sig_inv @ dmu
    Lm = jnp.kron(jnp.eye(nn), Ar0) + jnp.kron(Ar0, jnp.eye(nn))
    dSig = jnp.linalg.solve(Lm, -(dAr @ Sig + Sig @ dAr.T).flatten()).reshape(nn, nn)
    X = Sig_inv @ dSig
    F_cov = 0.5 * jnp.trace(X @ X)
    F_C = F_mean + F_cov
    F_norm = F_C / norm2
    T = jnp.linalg.inv(-A0)
    Sf = jnp.abs(T[n - 1, 0]) ** 2 + jnp.abs(T[n - 1 + n, 0]) ** 2
    Sr = jnp.abs(T[0, n - 1]) ** 2 + jnp.abs(T[0 + n, n - 1]) ** 2
    gf = 10.0 * jnp.log10(jnp.maximum(Sf, 1e-30))
    gr = 10.0 * jnp.log10(jnp.maximum(Sr, 1e-30))
    return F_C, norm2, F_norm, max_re, gf, gr, gf - gr


def loss(x, N):
    """Production loss: negative normalized FI plus stability barrier."""
    _, _, F_norm, max_re, *_ = compute_all(x, N)
    pen = jnp.maximum(max_re + 0.05, 0.0) * 1e8
    return -F_norm + pen


def adam_run(grad_fn, loss_fn, key):
    """Run one full production-protocol Adam optimization from a random start."""
    x = random.uniform(key, (4,), minval=-2.0, maxval=2.0)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    best_x, best_f = x, 1e30
    for step in range(N_OPTIM):
        g = grad_fn(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (step + 1))
        vh = v / (1 - 0.999 ** (step + 1))
        x = x - LR * mh / (jnp.sqrt(vh) + 1e-8)
        if (step + 1) % 100 == 0:
            lv = float(loss_fn(x))
            if lv < best_f:
                best_f = lv
                best_x = x.copy()
    return best_x, best_f


def classify(F_norm, max_re):
    """Assign a solution class from its normalized FI and spectral margin."""
    if max_re >= -0.01:
        return "unstable"
    return "deep_stability" if F_norm >= DEEP_THRESHOLD else "marginal_stability"


def self_check():
    """Verify the model reproduces the published optima before any optimization."""
    refs = json.load(open(REF_JSON))
    ok = True
    for r in refs:
        x = jnp.array(map_back([r["J"], r["g"], r["lam_inter"], r["phi"]]))
        _, _, F_norm, max_re, gf, gr, iso = [float(v) for v in compute_all(x, r["N"])]
        checks = [
            ("F_C_norm", F_norm, r["F_C_norm"], 1e-3),
            ("max_Re_lam", max_re, r["max_Re_lam"], 1e-4),
            ("gain_fwd", gf, r["gain_fwd_dB"], 5e-2),
            ("gain_rev", gr, r["gain_rev_dB"], 5e-2),
            ("isolation", iso, r["isolation_dB"], 5e-2),
        ]
        for name, got, want, rtol in checks:
            if abs(got - want) > rtol * (abs(want) + 1e-12):
                print(f"SELF-CHECK FAIL N={r['N']} {name}: got {got}, want {want}")
                ok = False
    if not ok:
        raise SystemExit("Self-check failed: model does not match published v3.")
    print("Self-check OK: all 6 published optima reproduced within tolerance.")


def main():
    """Run the seed-dependence study (optionally a subset of N) with checkpoints."""
    import argparse
    global OUT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=",".join(map(str, Ns)), help="comma-separated lattice sizes")
    ap.add_argument("--out", default=OUT_JSON, help="output/checkpoint JSON path")
    args = ap.parse_args()
    ns = [int(v) for v in args.ns.split(",")]
    OUT_JSON = args.out
    self_check()
    try:
        enriched = {r["N"]: r.get("solution_class") for r in json.load(open(ENRICHED_JSON))}
    except FileNotFoundError:
        # published classes of tab:nscaling (manuscript facts; enriched JSON lost in
        # the 2026-08-23 workspace restructure)
        enriched = {6: "deep_stability", 8: "marginal_stability", 10: "deep_stability",
                    12: "marginal_stability", 14: "deep_stability", 16: "marginal_stability"}
    results = []
    done = set()
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON)).get("per_N", [])
            results = [e for e in prev if e["N"] in ns]
            done = {e["N"] for e in results}
        except Exception:
            pass
        print(f"Resuming: {len(done)}/{len(Ns)} sizes already complete.")
    t_start = time.time()
    for N in ns:
        if N not in Ns:
            raise SystemExit(f'refusing unexpected N={N}')
        if N in done:
            continue
        import functools
        grad_fn = jax.jit(jax.grad(functools.partial(loss, N=N)))
        loss_fn = jax.jit(functools.partial(loss, N=N))
        fisher_fn = jax.jit(functools.partial(compute_all, N=N))
        print(f'PROCESS START N={N} -> {OUT_JSON}', flush=True)
        t0 = time.time()
        seeds = []
        for k in range(SEEDS_PER_N):
            key = random.PRNGKey(BASE_SEED + 1000 * N + k)
            bx, _bf = adam_run(grad_fn, loss_fn, key)
            _, _, F_norm, max_re, gf, gr, iso = [float(v) for v in fisher_fn(bx)]
            p = unmap(bx)
            seeds.append({
                "seed_index": k,
                "J": float(p[0]), "g": float(p[1]),
                "lam_inter": float(p[2]), "phi": float(p[3]),
                "F_C_norm": F_norm, "max_Re_lam": float(max_re),
                "gain_fwd_dB": gf, "gain_rev_dB": gr, "isolation_dB": iso,
                "class": classify(F_norm, float(max_re)),
            })
        counts = {"deep_stability": 0, "marginal_stability": 0, "unstable": 0}
        for s in seeds:
            counts[s["class"]] += 1
        entry = {
            "N": int(N),
            "reference_class": enriched.get(N),
            "counts": counts,
            "seeds": seeds,
            "time_s": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"N={N}: ref={entry['reference_class']} deep={counts['deep_stability']} "
              f"marginal={counts['marginal_stability']} unstable={counts['unstable']} "
              f"[{entry['time_s']}s]", flush=True)
        # checkpoint after each N (interrupt/resume safe)
        payload = {
            "protocol": {
                "seeds_per_N": SEEDS_PER_N, "adam_steps": N_OPTIM, "lr": LR,
                "deep_threshold": DEEP_THRESHOLD, "base_seed": BASE_SEED,
                "barrier": "max Re lambda < -0.05 (production)",
            },
            "per_N": results,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, indent=2)

    matrix = {rc: {c: 0 for c in ("deep_stability", "marginal_stability", "unstable")}
              for rc in ("deep_stability", "marginal_stability")}
    for e in results:
        for s in e["seeds"]:
            if e["reference_class"] in matrix:
                matrix[e["reference_class"]][s["class"]] += 1
    print("\nConfusion matrix (rows = published class, cols = re-optimized class):")
    header = "".join(f"{c:>20}" for c in ("deep", "marginal", "unstable"))
    print(f"{'':>22}{header}")
    for rc, row in matrix.items():
        cells = "".join(f"{row[c]:>20}" for c in ("deep_stability", "marginal_stability", "unstable"))
        print(f"{rc:>22}{cells}")
    payload = json.load(open(OUT_JSON))
    payload["confusion_matrix"] = matrix
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT_JSON}  (total {round(time.time() - t_start, 1)}s)")


if __name__ == "__main__":
    main()
