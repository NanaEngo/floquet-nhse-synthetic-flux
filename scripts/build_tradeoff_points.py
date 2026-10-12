#!/usr/bin/env python3
"""Build the unified S-convention tradeoff cloud (Option A unification).

Reads the 60 DOC seeds from data/classification_stability.json and the six
published optima from data/n_scaling_results_v3.json, evaluates seed gains
with doc_core.published_gains (reported S-matrix convention) and writes
data/tradeoff_points.json deterministically. Optima rows are copied
verbatim (already S). Prints per-class medians/IQRs for the manuscript.

Seed F_C_norm/max_re are convention-free and copied as stored; only
Gfwd/Iso move from the legacy T diagnostic to S. The raw T seed record
(classification_stability.json) is left untouched.
"""
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doc_core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SEEDS_JSON = os.path.join(DATA, "classification_stability.json")
OPTS_JSON = os.path.join(DATA, "n_scaling_results_v3.json")
OUT_JSON = os.path.join(DATA, "tradeoff_points.json")

CLS = {"deep_stability": "deep", "marginal_stability": "marg", "unstable": "unst"}


def main():
    """Rebuild data/tradeoff_points.json with S-convention seed gains."""
    seeds = json.load(open(SEEDS_JSON))["per_N"]
    opts = json.load(open(OPTS_JSON))
    points = []
    for entry in seeds:
        N = entry["N"]
        for s in sorted(entry["seeds"], key=lambda r: r["seed_index"]):
            gf, gr, iso = [float(v) for v in doc_core.published_gains(
                s["J"], s["g"], s["lam_inter"], s["phi"], N)]
            points.append({
                "max_re": float(s["max_Re_lam"]),
                "Fnorm": float(s["F_C_norm"]),
                "Gfwd": round(gf, 2),
                "Iso": round(iso, 2),
                "cls": CLS[s["class"]],
                "N": int(N),
                "src": "seed",
            })
    for r in opts:
        cls = "deep" if r["F_C_norm"] >= 0.5 else "marg"
        points.append({
            "max_re": float(r["max_Re_lam"]),
            "Fnorm": float(r["F_C_norm"]),
            "Gfwd": float(r["gain_fwd_dB"]),
            "Iso": float(r["isolation_dB"]),
            "cls": cls,
            "N": int(r["N"]),
            "src": "published",
        })
    with open(OUT_JSON, "w") as f:
        json.dump(points, f, indent=2)
    print(f"wrote {OUT_JSON} ({len(points)} points)")
    by_cls = {}
    for p in points:
        if p["src"] == "seed":
            by_cls.setdefault(p["cls"], {"gf": [], "iso": [], "fn": []})
            by_cls[p["cls"]]["gf"].append(p["Gfwd"])
            by_cls[p["cls"]]["iso"].append(p["Iso"])
            by_cls[p["cls"]]["fn"].append(p["Fnorm"])
    for cls, v in by_cls.items():
        print(f"{cls}: n={len(v['gf'])} "
              f"Gfwd med {statistics.median(v['gf']):.2f} "
              f"IQR [{np.percentile(v['gf'], 25):.1f}, {np.percentile(v['gf'], 75):.1f}] "
              f"min {min(v['gf']):.1f} | "
              f"Iso med {statistics.median(v['iso']):.2f} "
              f"IQR [{np.percentile(v['iso'], 25):.1f}, {np.percentile(v['iso'], 75):.1f}] "
              f"min {min(v['iso']):.1f} | "
              f"Fnorm med {statistics.median(v['fn']):.4f}")


if __name__ == "__main__":
    main()
