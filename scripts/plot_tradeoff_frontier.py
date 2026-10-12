#!/usr/bin/env python3
"""Plot the precision--gain tradeoff cloud (fig:tradeoff).

Reads data/tradeoff_points.json (unified S-convention, built by
build_tradeoff_points.py): three panels sharing the spectral abscissa
max Re lambda -- (a) Fnorm, (b) Gfwd in dB, (c) isolation in dB.
Red/blue markers are deep/marginal seed draws, gray are unstable draws,
open black circles the six published optima; dashed line at -0.05 kappa.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use(["science", "nature"])
except OSError:  # pragma: no cover - style fallback
    plt.style.use("seaborn-v0_8-ticks")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "tradeoff_points.json")
OUT = os.path.join(HERE, "..", "figures", "tradeoff_stability_margin.pdf")

COLORS = {"deep": "#d62728", "marg": "#1f77b4", "unst": "#7f7f7f"}


def main():
    """Render the three-panel tradeoff figure from the unified artifact."""
    points = json.load(open(DATA))
    seeds = [p for p in points if p["src"] == "seed"]
    opts = [p for p in points if p["src"] == "published"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)
    panels = [("Fnorm", r"$F_{\rm norm}$"), ("Gfwd", r"$G_{\rm fwd}$ (dB)"),
              ("Iso", r"Isolation (dB)")]
    for ax, (key, label) in zip(axes, panels):
        for cls in ("deep", "marg", "unst"):
            xs = [p["max_re"] for p in seeds if p["cls"] == cls]
            ys = [p[key] for p in seeds if p["cls"] == cls]
            ax.scatter(xs, ys, s=9, c=COLORS[cls], alpha=0.8,
                       linewidths=0)
        ax.scatter([p["max_re"] for p in opts], [p[key] for p in opts],
                   s=28, facecolors="none", edgecolors="k", linewidths=0.9,
                   zorder=5)
        ax.axvline(-0.05, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$\max\,\mathrm{Re}\,\lambda\;(\kappa)$")
        ax.set_ylabel(label)
    axes[0].text(0.05, 0.93, "(a)", transform=axes[0].transAxes)
    axes[1].text(0.05, 0.93, "(b)", transform=axes[1].transAxes)
    axes[2].text(0.05, 0.93, "(c)", transform=axes[2].transAxes)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT} ({len(seeds)} seeds + {len(opts)} optima)")


if __name__ == "__main__":
    main()
