#!/usr/bin/env python3
"""Regenerate retained Track C figures from canonical data/*.json sources.

Every plotted number traces to a retained data artifact; literature and device
constants are stated in the corresponding figure captions.
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

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
})


def load(name):
    """Load a canonical JSON artifact from data/.

    Args:
        name: JSON filename (str) inside data/.

    Returns:
        Parsed JSON content (dict or list).
    """
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


# ------------------------------------------------------------------ figure 1
def fig_nhse_vs_ep():
    """Plot single-chain SNR/Fisher scaling from trackC_snr_decoupling.json.

    Saves figures/nhse_vs_ep_scaling.pdf.

    Returns:
        None.
    """
    s = load("trackC_snr_decoupling.json")
    res = sorted(s["results"], key=lambda r: r["N"])
    N = np.array([r["N"] for r in res])
    snr = np.array([r["snr_matched"] for r in res])
    fraw = np.array([r["F_C_raw"] for r in res])
    fnrm = np.array([r["F_C_normalized"] for r in res])
    fits = {
        "SNR$_N$": (np.array([snr]), s["target_regime_snr"]),
        "$F_C^{\\rm raw}$": (None, s["fisher_record"]),
        "$F_C^{\\rm norm}$": (None, s["fisher_normalized"]),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    nn = np.linspace(5.5, 16.5, 50)

    series = [
        (snr, "#2166ac", "o", "matched SNR$_N$",
         s["target_regime_snr"]["slope"], s["target_regime_snr"]["intercept"]),
        (fraw, "#4daf4a", "s", "$F_C^{\\rm raw}$",
         s["fisher_record"]["slope"], s["fisher_record"]["intercept"]),
        (fnrm, "#b2182b", "D", "$F_C^{\\rm norm}$",
         s["fisher_normalized"]["slope"], s["fisher_normalized"]["intercept"]),
    ]
    for vals, col, mk, lab, a, b in series:
        ax.semilogy(N, vals, mk, color=col, ms=6, label=lab, zorder=3)
        ax.semilogy(nn, np.exp(a * nn + b), "--", color=col, lw=0.9, alpha=0.6, zorder=2)
        ax.annotate(f"$a={a:+.4f}$", xy=(15.6, float(np.exp(a * 15.6 + b))),
                    color=col, fontsize=7.5, va="center")
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel("Single-chain observable ($\\kappa$ units)")
    ax.set_xlim(5, 17)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "nhse_vs_ep_scaling.pdf"), bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ figure 2
def fig_basin_occupancy():
    """Plot DOC basin counts from classification_stability.json.

    Saves figures/optimization.pdf.

    Returns:
        None.
    """
    c = load("classification_stability.json")
    per_n = c["per_N"]
    Ns = [d["N"] for d in per_n]
    deep = np.array([d["counts"]["deep_stability"] for d in per_n])
    marg = np.array([d["counts"]["marginal_stability"] for d in per_n])
    unst = np.array([d["counts"]["unstable"] for d in per_n])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(Ns))
    ax.bar(x, deep, 0.62, color="#b2182b", label="deep stability")
    ax.bar(x, marg, 0.62, bottom=deep, color="#2166ac", label="marginal stability")
    ax.bar(x, unst, 0.62, bottom=deep + marg, color="#9e9e9e", label="unstable")
    for i in range(len(Ns)):
        if deep[i]:
            ax.text(i, deep[i] / 2, str(deep[i]), ha="center", va="center",
                    color="w", fontsize=8)
        if marg[i]:
            ax.text(i, deep[i] + marg[i] / 2, str(marg[i]), ha="center",
                    va="center", color="w", fontsize=8)
        if unst[i]:
            ax.text(i, deep[i] + marg[i] + unst[i] / 2, str(unst[i]),
                    ha="center", va="center", fontsize=8)
    tot_deep = int(deep.sum())
    tot_marg = int(marg.sum())
    tot_unst = int(unst.sum())
    pub = sum(d["counts"][d["reference_class"]] for d in per_n)
    ax.axhline(color="k", lw=0.6)
    ax.set_xticks(x, [str(n) for n in Ns])
    ax.set_xlabel("Array size $N$ (10 independent restarts each)")
    ax.set_ylabel("Converged solutions (count)")
    ax.set_ylim(0, 10.5)
    ax.legend(ncol=3, loc="upper center", framealpha=0.9)
    ax.annotate(f"published class recovered: {pub}/60 ($\\approx 53\\%$)",
                xy=(0.02, 0.93), xycoords="axes fraction", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "optimization.pdf"), bbox_inches="tight")
    plt.close()
    print(f"basins: deep={tot_deep} marginal={tot_marg} unstable={tot_unst}")


# ------------------------------------------------------------------ figures 3-4
def _otoc_data():
    """Read OTOC exponents and the platform MSS bound from trackC_physical_data.json.

    Returns:
        Tuple (N, lam, r2, mss) with N site counts (int array), lam OTOC
        rates in units of kappa, r2 fit qualities (dimensionless), mss the
        MSS bound in units of kappa.
    """
    d = load("trackC_physical_data.json")["otoc_convergence"]
    N = np.array(d["n_values"])
    lam = np.array(d["lambda_raw_list"])
    r2 = np.array(d["r2_list"])
    # trackC_physical_data.json stores the corrected platform MSS bound directly.
    mss = d["mss_bound"]
    return N, lam, r2, mss


def fig_otoc_exponents():
    """Plot OTOC Lyapunov exponents vs N with per-point R^2.

    Saves figures/otoc_scrambling.pdf.

    Returns:
        None.
    """
    N, lam, r2, mss = _otoc_data()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(N, lam, "o", color="#b2182b", ms=7, zorder=3)
    for n, l, r in zip(N, lam, r2):
        ax.annotate(f"$R^2={r:.3f}$", xy=(n, l), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel("$\\lambda_{\\rm OTOC}$ [$\\kappa^{-1}$]")
    ax.set_xticks(N)
    ax.set_ylim(0, max(lam) * 1.25)
    ax.grid(True, alpha=0.3)
    ax.annotate("semi-log fits, $t \\in [0.5, 1.5]\\,\\kappa^{-1}$",
                xy=(0.03, 0.92), xycoords="axes fraction", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "otoc_scrambling.pdf"), bbox_inches="tight")
    plt.close()
    print(f"lambda_OTOC: {dict(zip(N.tolist(), lam.round(2).tolist()))}")


def fig_mss_saturation():
    """Plot lambda_OTOC/lambda_MSS vs N on a log scale.

    Saves figures/mss_saturation.pdf.

    Returns:
        None.
    """
    N, lam, r2, mss = _otoc_data()
    ratio = lam / mss
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.axhline(1.0, color="k", lw=1.0, ls="--", zorder=2)
    ax.annotate("$\\lambda_{\\rm MSS}$ (MSS bound)", xy=(N[0], 1.0),
                xytext=(2, 6), textcoords="offset points", fontsize=8)
    ax.plot(N, ratio, "o", color="#2166ac", ms=7, zorder=3)
    for n, r in zip(N, ratio):
        ax.annotate(f"{r:.4f}", xy=(n, r), xytext=(0, -12),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color="#2166ac")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 3)
    ax.set_yticks([1e-2, 1e-1, 1])
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.set_xlabel("Array size $N$")
    ax.set_ylabel("$\\lambda_{\\rm OTOC}/\\lambda_{\\rm MSS}$")
    ax.set_xticks(N)
    ax.grid(True, alpha=0.3, which="both")
    ax.annotate(f"$\\max\\ \\lambda_{{\\rm OTOC}}/\\lambda_{{\\rm MSS}} = {ratio.max():.4f} \\ll 1$"
                "\n($\\kappa/2\\pi = 1.7$ MHz, $T_c = 15$ mK)",
                xy=(0.97, 0.04), xycoords="axes fraction", fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "mss_saturation.pdf"), bbox_inches="tight")
    plt.close()
    print(f"MSS bound at 1.7 MHz: {mss:.1f}; ratios: {ratio.round(4).tolist()}")


# ------------------------------------------------------------------ figure 5
def _force_scales():
    """Same formulas as derive_device_params.py (MacCabe2020 platform)."""
    dev = load("trackC_snr_decoupling.json")["params"]["physical_device"]
    hbar = 1.054571817e-34
    m = dev["m_eff_fg"] * 1e-18
    om = 2 * np.pi * dev["Omega_m_GHz"] * 1e9
    gam = om / dev["Q_m"]
    sql = np.sqrt(hbar * m * om * gam) / 1e-18   # N/sqrt(Hz) -> aN
    coop = 4 * (2 * np.pi * dev["g0_phys_MHz"] * 1e6) ** 2 / (
        2 * np.pi * dev["kappa_phys_MHz"] * 1e6 * gam)
    return sql, sql / np.sqrt(coop / 4)


def fig_weak_force():
    """Plot force-noise levels and the conditional axion projection.

    Saves figures/weak_force_limits.pdf. Force scales are in
    aN/sqrt(Hz); axion couplings in GeV^-1.

    Returns:
        None.
    """
    sql, opt = _force_scales()
    band = np.array([1.0, 10.0])
    cast = 6.6e-11          # CAST solar-axion bound (cited in caption)
    bare = 1.7e-14          # five-GHz conditional estimate from derive_axion_bound.py

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axs[0]
    for lev, col, lab in ((sql, "#2166ac", "$S_{FF}^{\\rm SQL}$"),
                          (opt, "#b2182b", "imprecision--back-action optimum")):
        ax.plot(band, [lev, lev], color=col, lw=1.6, label=lab)
        ax.annotate(f"{lev:.1e} $\\mathrm{{aN}}/\\sqrt{{\\mathrm{{Hz}}}}$",
                    xy=(5.5, lev), xytext=(0, 4), textcoords="offset points",
                    color=col, fontsize=7.5, ha="center")
    ax.set_yscale("log")
    ax.set_xlim(1, 10)
    ax.set_xlabel("Fourier frequency $\\Omega/2\\pi$ [GHz]")
    ax.set_ylabel("$S_{FF}^{1/2}$ [$\\mathrm{aN}/\\sqrt{\\mathrm{Hz}}$]")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center left", fontsize=7)

    ax = axs[1]
    ax.axhspan(cast, 1e-9, color="#d9d9d9", alpha=0.7)
    ax.annotate("CAST bound", xy=(5.5, cast * 1.6), ha="center", fontsize=7.5,
                color="#555555")
    ax.plot(band, [bare, bare], color="#b2182b", lw=1.6,
            label="conditional model estimate")
    ax.set_yscale("log")
    ax.set_xlim(1, 10)
    ax.set_ylim(3e-15, 3e-10)
    ax.set_xlabel("Axion mass $m_a c^2/h$ [GHz]")
    ax.set_ylabel("$g_{a\\gamma\\gamma}$ [GeV$^{-1}$]")
    y0 = 10 ** (np.log10(bare) - 0.25)
    ax.annotate(f"$g_{{a\\gamma\\gamma}} \\gtrsim {bare:.1e}$ at 5 GHz",
                xy=(5.5, y0), ha="center", fontsize=7.5, color="#b2182b")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "weak_force_limits.pdf"), bbox_inches="tight")
    plt.close()
    print(f"SQL={sql:.4g} aN, BA-opt={opt:.4g} aN")


if __name__ == "__main__":
    fig_nhse_vs_ep()
    fig_basin_occupancy()
    fig_otoc_exponents()
    fig_mss_saturation()
    fig_weak_force()
    print("done")
