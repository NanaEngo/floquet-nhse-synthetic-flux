#!/usr/bin/env python3
"""Basin predictability for the Track C DOC landscape (C-S1, C-S3).

Regenerates the deterministic initial points of the 60-seed
classification ensemble (PRNGKey(BASE_SEED + 1000*N + k), uniform(-2, 2))
and correlates basin outcome (deep/marginal/unstable) with initial
features: spectral margin, normalized Fisher information, loss, and
barrier distance. Also profiles the unstable draws (initial vs final)
and records the N=11 negative-Fnorm anomaly from the intermediate scan.

Gate: recomputed final Fnorm from stored final parameters must match
the archived values (validates the doc_core path + parameter record).
Aborts on mismatch. No optimization is run; pure post-processing.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doc_core  # noqa: E402
from doc_core import THETA0  # noqa: E402,F401 (protocol constant, documented)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SEEDS_JSON = os.path.join(DATA, "classification_stability.json")
INTER_JSON = os.path.join(DATA, "n_scaling_intermediate.json")
OUT_JSON = os.path.join(DATA, "basin_predictability.json")

BASE_SEED = 20260823


def quartiles(xs):
    """Median and interquartile interval of a sample (dimensionless stats).

    Args:
        xs: Flat sequence of floats.

    Returns:
        Tuple (median, q25, q75).
    """
    s = sorted(xs)
    n = len(s)
    med = statistics.median(s)
    q25 = statistics.median(s[: n // 2])
    q75 = statistics.median(s[(n + 1) // 2:])
    return med, q25, q75


def main():
    """Correlate basin outcomes with deterministic initial features."""
    import functools
    import jax
    import jax.numpy as jnp
    from jax import random

    ensemble = json.load(open(SEEDS_JSON))["per_N"]
    feats = []  # one row per seed
    for entry in ensemble:
        N = entry["N"]
        import time as _time
        _t0 = _time.time()

        def _init_feats(x):
            _, _, fn_, mr_, *_ = doc_core.compute_all(x, N)
            return fn_, mr_, doc_core.loss(x, N)

        eval_init = jax.jit(_init_feats)
        fisher = jax.jit(functools.partial(doc_core.compute_all, N=N))
        for s in entry["seeds"]:
            key = random.PRNGKey(BASE_SEED + 1000 * N + s["seed_index"])
            x0 = random.uniform(key, (4,), minval=-2.0, maxval=2.0)
            f0, mr0, l0 = [float(v) for v in eval_init(x0)]
            # gate: stored final params reproduce stored final Fnorm
            xf = jnp.array(doc_core.map_back(
                [s["J"], s["g"], s["lam_inter"], s["phi"]]))
            _, _, ff, *_ = [float(v) for v in fisher(xf)]
            if abs(ff - s["F_C_norm"]) > 1e-9 * (abs(s["F_C_norm"]) + 1e-12):
                raise SystemExit(
                    f"PARAM RECORD FAIL N={N} seed={s['seed_index']}: "
                    f"recomputed {ff}, stored {s['F_C_norm']}")
            feats.append({
                "N": N, "seed_index": s["seed_index"],
                "class": s["class"],
                "init_max_re": float(mr0), "init_Fnorm": float(f0),
                "init_loss": float(l0),
                "barrier_gap": float(-0.05 - mr0),
                "final_Fnorm": float(s["F_C_norm"]),
                "final_max_re": float(s["max_Re_lam"]),
            })
        print(f"N={N}: 10 seeds evaluated [{_time.time() - _t0:.0f}s]",
              flush=True)
    by_cls = {}
    for f in feats:
        by_cls.setdefault(f["class"], []).append(f)
    summary = {}
    for cls, rows in by_cls.items():
        block = {"n": len(rows)}
        for feat in ("init_max_re", "init_Fnorm", "init_loss", "barrier_gap"):
            med, q25, q75 = quartiles([r[feat] for r in rows])
            block[feat] = {"median": med, "q25": q25, "q75": q75}
        summary[cls] = block
    # quartile rule: P(deep | init_loss in best quartile)
    losses = sorted(feats, key=lambda r: r["init_loss"])
    q = len(losses) // 4
    for name, subset in (("best_loss_quartile", losses[:q]),
                         ("worst_loss_quartile", losses[-q:])):
        counts = {}
        for r in subset:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        summary[name] = {"n": len(subset), "classes": counts}
    # unstable profile (C-S3): init vs final + N=11 negatives note
    unst = [f for f in feats if f["class"] == "unstable"]
    summary["unstable_profile"] = {
        "n": len(unst),
        "rows": unst,
        "note": "N=11 intermediate scan returned two near-zero negative "
                "Fnorm (-0.0021, -0.0015); per-restart parameters were not "
                "recorded there, so no initial-feature profile exists for "
                "them.",
    }
    summary["early_stop"] = {
        "status": "not_validated",
        "design": "abort a restart if loss after 100 steps exceeds a "
                  "calibrated threshold; requires instrumented reruns "
                  "(loss trajectories are not archived).",
    }
    payload = {
        "status": "BASIN_PREDICTABILITY_COMPUTED",
        "method": "deterministic initial-point regeneration "
                  "(BASE_SEED=20260823) + doc_core evaluation; no optimization",
        "classes": summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    for cls, block in summary.items():
        if cls in by_cls:
            print(f"{cls}: n={block['n']} "
                  f"init_max_re med {block['init_max_re']['median']:.3f} "
                  f"init_Fnorm med {block['init_Fnorm']['median']:.4f} "
                  f"init_loss med {block['init_loss']['median']:.3f}")
    for name in ("best_loss_quartile", "worst_loss_quartile"):
        print(f"{name}: {summary[name]['classes']}")
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
