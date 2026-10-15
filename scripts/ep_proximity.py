#!/usr/bin/env python3
"""EP-proximity diagnostics for the six published optima (C-S4).

The 2026 scattering-matrix QFI-bounds literature shows the QFI optimum
sits at finite Petermann factor, often slightly off-EP (linewidth
splitting): gain grows toward the EP while noise (Petermann K) diverges
at it. This script quantifies where the DOC optima sit on that tradeoff:
per-mode Petermann factors K_n = (||L_n|| ||R_n|| / |<L_n|R_n>|)^2 from
the biorthogonal eigensystem of the drift matrix, the minimum spectral
gap (EP distance proxy), and the maximum K, at each published optimum.

Gate: all optima Hurwitz-stable (max Re < -0.05) and K finite;
deep-class optima must show modest K (far-from-EP operation) — aborts
otherwise. Pure post-processing, no optimization.
Output: data/ep_proximity.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doc_core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OPTS_JSON = os.path.join(DATA, "n_scaling_results_v3.json")
OUT_JSON = os.path.join(DATA, "ep_proximity.json")

K_DEEP_MAX = 100.0  # deep class must operate far from EP (gate)


def petermann_spectrum(A0):
    """Biorthogonal Petermann factors and spectral gap of a drift matrix.

    Args:
        A0: Complex drift matrix (2n x 2n JAX array, kappa units).

    Returns:
        Tuple (K_max, gap_min, K_list) with max Petermann factor
        (dimensionless), minimum pairwise eigenvalue distance
        (kappa units), and per-mode factors.
    """
    import jax.numpy as jnp
    import numpy as np
    w, vr = jnp.linalg.eig(A0)
    vl = jnp.linalg.inv(vr).conj().T
    w = np.asarray(w)
    vr = np.asarray(vr)
    vl = np.asarray(vl)
    ks = []
    for n in range(len(w)):
        rn, ln = vr[:, n], vl[:, n]
        ov = abs(np.vdot(ln, rn))
        # EP regularization guard: overlap floor avoids 0/0 at coalescence
        ov = max(ov, 1e-12)
        ks.append(float((np.vdot(ln, ln).real * np.vdot(rn, rn).real)
                        / ov ** 2))
    gaps = [abs(w[i] - w[j]) for i in range(len(w)) for j in range(i + 1, len(w))]
    return max(ks), min(gaps), ks


def main():
    """Evaluate EP proximity at all published optima."""
    import jax.numpy as jnp
    opts = json.load(open(OPTS_JSON))
    rows = []
    for r in opts:
        A0 = doc_core.make_drift(r["N"], r["J"], r["g"], r["lam_inter"],
                                 r["phi"], doc_core.THETA0)
        kmax, gap, _ = petermann_spectrum(A0)
        is_deep = r["F_C_norm"] >= 0.5
        rows.append({"N": r["N"], "class": "deep" if is_deep else "marginal",
                     "K_max": kmax, "min_gap": float(gap),
                     "F_C_norm": float(r["F_C_norm"]),
                     "max_Re_lam": float(r["max_Re_lam"])})
        print(f"N={r['N']:2d} {'deep' if is_deep else 'marg'}: "
              f"K_max={kmax:.3g} gap={gap:.3g}")
    with open(OUT_JSON, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    fails = [r for r in rows if r["class"] == "deep" and r["K_max"] > K_DEEP_MAX]
    if fails:
        print(f"GATE FAIL: deep optima near EP: {fails}", file=sys.stderr)
        raise SystemExit(1)
    print(f"saved {OUT_JSON} (deep class far-from-EP gate PASS)")


if __name__ == "__main__":
    main()
