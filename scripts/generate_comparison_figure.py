#!/usr/bin/env python3
"""
Generate the single-chain vs double-chain comparison figure for Track C.

Reads both sources of truth instead of hardcoding numbers:
  - ../data/trackC_snr_decoupling.json : single-chain baseline
    (F_C_normalized per N + log-linear fit in 'fisher_normalized')
  - ../data/n_scaling_results_v3.json  : double-chain DOC-optimized ladder,
    split into deep-stability (N = 6, 10, 14) and marginal-stability
    (N = 8, 12, 16) classes.

Outputs figures/single_vs_double_chain_norm.pdf (relative to this script).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

with open(os.path.join(DATA, "trackC_snr_decoupling.json")) as f:
    snr = json.load(f)
with open(os.path.join(DATA, "n_scaling_results_v3.json")) as f:
    ladder = json.load(f)

# --- Single-chain baseline ---
N_single = np.array([r["N"] for r in snr["results"]])
Fnorm_single = np.array([r["F_C_normalized"] for r in snr["results"]])
fit = snr["fisher_normalized"]
a_norm, r2 = fit["slope"], fit["r2"]

# --- Double-chain DOC-optimized ladder, split by solution class ---
deep_N = np.array([r["N"] for r in ladder if r["F_C_norm"] >= 0.5])
deep_F = np.array([r["F_C_norm"] for r in ladder if r["F_C_norm"] >= 0.5])
marg_N = np.array([r["N"] for r in ladder if r["F_C_norm"] < 0.5])
marg_F = np.array([r["F_C_norm"] for r in ladder if r["F_C_norm"] < 0.5])

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

ax.plot(N_single, Fnorm_single, "o-", color="#2166ac", markersize=7, linewidth=1.5,
        label="Single chain (baseline)", zorder=3)
N_fit = np.linspace(5, 17, 100)
ax.plot(N_fit, np.exp(a_norm * N_fit + fit["intercept"]), "--",
        color="#2166ac", linewidth=1.0, alpha=0.6)

ax.plot(deep_N, deep_F, "D", color="#b2182b", markersize=8,
        markeredgecolor="#67001f", markeredgewidth=1.0,
        label="Double chain: deep stability", zorder=4)
ax.plot(marg_N, marg_F, "^", color="#d6604d", markersize=8,
        markeredgecolor="#b2182b", markeredgewidth=1.0,
        label="Double chain: marginal stability", zorder=4)

ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.annotate(rf"$a_{{\rm norm}} = {a_norm:.2f}$", xy=(10, 2e-4), fontsize=9,
            color="#2166ac", ha="center", va="bottom")
ax.annotate(r"$F_C^{\rm norm}$ plateaus", xy=(13.2, 1.4), fontsize=9,
            color="#b2182b", ha="center", va="bottom")

ax.set_xlabel(r"Array size $N$", fontsize=11)
ax.set_ylabel(r"Resource-normalized Fisher $F_C^{\rm norm}$", fontsize=11)
ax.set_yscale("log")
ax.set_xlim(5, 17)
ax.set_ylim(1e-7, 1e1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
os.makedirs(FIGS, exist_ok=True)
out_pdf = os.path.join(FIGS, "single_vs_double_chain_norm.pdf")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"Figure saved: {out_pdf}")
print(f"  single-chain fit: slope = {a_norm:.4f}, R^2 = {r2:.4f}")
for n, fv in zip(deep_N, deep_F):
    print(f"  deep N={n}: F_C^norm = {fv:.4f}")
for n, fv in zip(marg_N, marg_F):
    print(f"  marginal N={n}: F_C^norm = {fv:.4f}")
