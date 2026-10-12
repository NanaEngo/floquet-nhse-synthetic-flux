#!/usr/bin/env python3
"""
Track C audit C-P1-11: seed-dependence of the deep/marginal solution classes.

For each lattice size N in the published ladder, rerun the production DOC
protocol (identical hyperparameters: 500 Adam steps, lr 0.003, stability
barrier max Re lambda < -0.05) from 10 independent random initializations,
classify every converged solution (F_C^norm >= 0.5 -> deep_stability, else
marginal_stability; unstable if the final spectral margin fails), and build
the confusion matrix against the published classes.

Model conventions live in scripts/doc_core.py (kappa=1, theta0=0.1,
d-theta probe, record Fisher information normalized by ||mu||^2) and are
imported here, so producers and gates share one implementation.
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
import argparse

import jax
import jax.numpy as jnp
from jax import random

from doc_core import (
    N_OPTIM,
    LR,
    unmap,
    map_back,
    compute_all,
    loss,
    published_gains,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
REF_JSON = os.path.join(DATA_DIR, "n_scaling_results_v3.json")
ENRICHED_JSON = os.path.join(DATA_DIR, "n_scaling_results_v3_enriched.json")
OUT_JSON = os.path.join(DATA_DIR, "classification_stability.json")

Ns = [6, 8, 10, 12, 14, 16]
SEEDS_PER_N = 10
# N_OPTIM/LR: production protocol, single source in doc_core (imported above).
DEEP_THRESHOLD = 0.5
BASE_SEED = 20260823   # audit run; distinct from production seed 42


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
    """Verify the model reproduces the published optima before any optimization.

    F_C_norm/max_Re_lam come from the record-Fisher model; gains use the
    reported scattering-matrix convention (doc_core.s_gains), matching
    every published gain. The T-matrix transmission inside compute_all is
    a diagnostic only.
    """
    refs = json.load(open(REF_JSON))
    ok = True
    for r in refs:
        x = jnp.array(map_back([r["J"], r["g"], r["lam_inter"], r["phi"]]))
        _, _, F_norm, max_re, *_ = [float(v) for v in compute_all(x, r["N"])]
        gf, gr, iso = [float(v) for v in published_gains(
            r["J"], r["g"], r["lam_inter"], r["phi"], r["N"])]
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
            # NOTE convention split (frozen with the published analysis):
            # per-seed gains stay in the T-matrix diagnostic convention
            # because the published ensemble medians (marginal Gfwd
            # +11.5 dB, IQR 7.7-19.7) and the tradeoff cloud were computed
            # from it. Only the six optima rows use the reported S-matrix
            # convention (see published_gains/self_check). Do not mix
            # without updating the manuscript medians.
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
