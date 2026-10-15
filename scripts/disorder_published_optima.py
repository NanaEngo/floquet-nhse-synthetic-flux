#!/usr/bin/env python3
"""Disorder re-optimization on the six published optima (C-S2).

The v2 disorder study stress-tested a dedicated reference configuration;
this script applies the same question to the actually published optima:
perturb each optimum by uncorrelated Gaussian disorder (RMS 5% on the
hopping/inter-chain parameters, the documented proxy for detuning and
hopping fluctuations), Adam-refine with a few restarts through the
shared doc_core loss, and record retention = best Fnorm / nominal Fnorm
per realization. Reports the count of realizations retaining >= 80%.

Reuses classification_stability.adam_run/classify and doc_core model
bit-identically; the only new code is the disorder loop. HPC-scale at
full realizations/restarts; --smoke runs 1 realization x N=6 x 1 restart.
Output: data/disorder_published_optima.json (checkpointed per size).
"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doc_core  # noqa: E402
import classification_stability as cs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REF_JSON = os.path.join(DATA, "n_scaling_results_v3.json")
OUT_JSON = os.path.join(DATA, "disorder_published_optima.json")

SIGMA = 0.05
RETAIN = 0.80


def main():
    """Run the disorder-on-optima study with per-size checkpointing."""
    import argparse
    import jax
    import jax.numpy as jnp
    from jax import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    n_real = 1 if args.smoke else args.realizations
    n_rest = 1 if args.smoke else args.restarts
    refs = json.load(open(REF_JSON))
    try:
        results = json.load(open(args.out))["per_N"]
        done = {e["N"] for e in results}
    except (FileNotFoundError, KeyError):
        results, done = [], set()
    for ref in refs:
        N = ref["N"]
        if N in done:
            print(f"skip complete N={N}")
            continue
        nominal = float(ref["F_C_norm"])
        grad_fn = jax.jit(jax.grad(functools.partial(doc_core.loss, N=N)))
        loss_fn = jax.jit(functools.partial(doc_core.loss, N=N))
        fisher_fn = jax.jit(functools.partial(doc_core.compute_all, N=N))
        key = random.PRNGKey(args.seed + N)
        retentions, details = [], []
        import time
        t0 = time.time()
        for m in range(n_real):
            key, sub = random.split(key)
            # disorder proxy: 5% Gaussian on (J, g, lam); phi untouched
            base = jnp.array([ref["J"], ref["g"], ref["lam_inter"]])
            pert = base * (1.0 + SIGMA * random.normal(sub, (3,)))
            x_init = jnp.array(doc_core.map_back(
                [float(pert[0]), float(pert[1]), float(pert[2]), ref["phi"]]))
            best_f, best_info = 1e30, None
            for rr in range(n_rest):
                key, sub = random.split(key)
                if rr == 0:
                    x = x_init
                    m0, v0 = jnp.zeros(4), jnp.zeros(4)
                else:
                    x = random.uniform(sub, (4,), minval=-2.0, maxval=2.0)
                    m0, v0 = jnp.zeros(4), jnp.zeros(4)
                # short Adam refinement honoring doc_core LR/N protocol shape
                # (moments named mm/vv: m is the realization counter)
                mm, vv = m0, v0
                bx, bf = x, 1e30
                for step in range(args.steps if not args.smoke else 50):
                    g = grad_fn(x)
                    mm = 0.9 * mm + 0.1 * g
                    vv = 0.999 * vv + 0.001 * g ** 2
                    mh = mm / (1 - 0.9 ** (step + 1))
                    vh = vv / (1 - 0.999 ** (step + 1))
                    x = x - doc_core.LR * mh / (jnp.sqrt(vh) + 1e-8)
                    if (step + 1) % 50 == 0:
                        lv = float(loss_fn(x))
                        if lv < bf:
                            bf, bx = lv, x.copy()
                if bf < best_f:
                    best_f = bf
                    _, _, fn_, mr_, *_ = [float(v) for v in fisher_fn(bx)]
                    best_info = (float(fn_), float(mr_))
            ret = best_info[0] / nominal if nominal > 0 else 0.0
            retentions.append(ret)
            details.append({"realization": m, "retention": ret,
                            "Fnorm": best_info[0], "max_re": best_info[1]})
        passed = sum(1 for r in retentions if r >= RETAIN)
        results.append({"N": N, "nominal_Fnorm": nominal,
                        "n_realizations": n_real, "n_restarts": n_rest,
                        "adam_steps": args.steps if not args.smoke else 50,
                        "retentions": retentions,
                        "pass_ge_80pct": passed,
                        "details": details,
                        "time_s": round(time.time() - t0, 1)})
        with open(args.out, "w") as f:
            json.dump({"per_N": results}, f, indent=2)
        print(f"N={N}: {passed}/{n_real} retain >=80% "
              f"(min {min(retentions):.2f})", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
