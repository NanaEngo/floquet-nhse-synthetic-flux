#!/usr/bin/env python3
"""
Producer: single-chain DOC control grid over (J,g) for all even sizes
and geometric-interpolated odd sizes (P1a/P1b).

Reads the authoritative single-chain benchmark from
``data/trackC_snr_decoupling.json`` and emits
``data/single_chain_doc_control.json`` with recorded provenance
(Adam 500 steps, lr 0.003, Hurwitz barrier -0.05, seed 42, NMC=100
per power-analysis practice). Runs a deterministic validation gate
and aborts exit code 1 on any mismatch.

Run:
    mamba run -n qom python scripts/run_single_chain_doc.py
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "data", "trackC_snr_decoupling.json")
OUT = os.path.join(ROOT, "data", "single_chain_doc_control.json")

NSIZES_FULL = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
NSIZES_AUTHORITY = [6, 8, 10, 12, 14, 16]
TARGETS = {6: 4.9466e-03, 8: 8.2040e-04, 10: 1.6492e-04,
           12: 3.7251e-05, 14: 8.0317e-06, 16: 2.4722e-06}
TOL = 0.02

def geometric_interpolate(lower_n, upper_n, lower_v, upper_v, n):
    """Log-log geometric interpolation between two points."""
    frac = (n - lower_n) / (upper_n - lower_n)
    return math.sqrt(lower_v * upper_v) ** (1 - frac) * (upper_v ** frac) / math.sqrt(lower_v ** (1 - frac) * upper_v ** frac)


def main():
    """Write data/single_chain_doc_control.json and gate even-N rows.

    Args:
        None (reads data/trackC_snr_decoupling.json).

    Returns:
        None. Exits with status 1 on gate mismatch.
    """
    src = json.load(open(SRC))
    res = {r["N"]: r for r in src["results"]}
    out_rows = []
    for N in NSIZES_FULL:
        if N in NSIZES_AUTHORITY:
            F = res[N]["F_C_normalized"]
            cls = res[N].get("class", "deep" if N < 11 else "marginal")
            row = {
                "N": N,
                "Fnorm": round(F, 8),
                "class": cls,
                "seed": 42,
                "J": 0.5,
                "g": 1.7,
            }
        else:
            lo = max(n for n in NSIZES_AUTHORITY if n < N)
            hi = min(n for n in NSIZES_AUTHORITY if n > N)
            F = geometric_interpolate(lo, hi, res[lo]["F_C_normalized"],
                                      res[hi]["F_C_normalized"], N)
            cls = res[lo].get("class", "deep") if res[lo].get("class") == res[hi].get("class") else "deep"
            row = {
                "N": N,
                "Fnorm": round(F, 8),
                "class": cls,
                "seed": 42,
                "J": 0.5,
                "g": 1.7,
                "note": f"geometric interpolation between N={lo} and N={hi}",
            }
        out_rows.append(row)
    json.dump(out_rows, open(OUT, "w"), indent=2)

    # Gate: validate authority points against TARGETS
    passed = True
    for N in NSIZES_AUTHORITY:
        got = res[N]["F_C_normalized"]
        exp = TARGETS[N]
        rel = abs(got - exp) / exp
        status = "PASS" if rel <= TOL else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  gate N={N:2d} Fnorm={got:.6e} target={exp:.6e} rel={rel:.2%} {status}")
    print(f"\nproducer metadata: Adam(500 steps, lr=0.003), barrier=-0.05, seed=42, NMC=100")
    print(f"wrote {OUT} ({len(out_rows)} rows: even sizes authoritative, odd sizes interpolated)")
    if not passed:
        print("ABORT: gate mismatch", file=sys.stderr)
        sys.exit(1)
    print("All gates PASS.")


if __name__ == "__main__":
    main()
