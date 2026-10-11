#!/usr/bin/env python3
"""
Producer for every Derived row of SI Table tab:si_device_params.

Reads the platform inputs from data/trackC_snr_decoupling.json
(params.physical_device, MacCabe2020 silicon phononic-crystal nanobeam),
derives the device quantities used in mainV2/SIV2, verifies them against
TARGETS with explicit tolerances, and exits non-zero on any mismatch.

Run: mamba run -n qom python scripts/derive_device_params.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "trackC_snr_decoupling.json")

HBAR = 1.054571817e-34
KB = 1.380649e-23

# TARGETS: expected manuscript values (tolerance = relative).
TARGETS = {
    "Gamma_m_per_s": (20.94, 0.02),
    "x_zpf_fm": (41.0, 0.03),
    "Omega_over_kappa": (2941.0, 0.01),
    "g0_over_kappa": (0.53, 0.01),
    "cooperativity_C": (5.72e5, 0.02),
    "n_th_BE": (1.128e-7, 0.02),
    "SQL_aN": (0.00833, 0.02),
    "BA_optimum_aN": (2.20e-5, 0.02),
}


def main():
    """Compute all derived rows, print them, verify against TARGETS."""
    dev = json.load(open(SRC))["params"]["physical_device"]
    kappa = dev["kappa_phys_MHz"] * 1e6 * 2 * math.pi          # rad/s
    g0 = dev["g0_phys_MHz"] * 1e6 * 2 * math.pi                # rad/s
    Omega = dev["Omega_m_GHz"] * 1e9 * 2 * math.pi             # rad/s
    m = dev["m_eff_fg"] * 1e-18                                # kg
    Q = dev["Q_m"]
    T = dev["T_c_mK"] * 1e-3                                   # K

    Gamma = Omega / Q                                          # s^-1
    x_zpf = math.sqrt(HBAR / (2.0 * m * Omega))                # m
    n_th = 1.0 / math.expm1(HBAR * Omega / (KB * T))           # Bose-Einstein
    C = 4.0 * g0 ** 2 / (kappa * Gamma)
    SQL_N = math.sqrt(HBAR * m * Omega * Gamma)                # N/sqrt(Hz)
    BA_N = SQL_N / math.sqrt(C / 4.0)

    out = {
        "Gamma_m_per_s": Gamma,
        "x_zpf_fm": x_zpf * 1e15,
        "Omega_over_kappa": Omega / kappa,
        "g0_over_kappa": g0 / kappa,
        "cooperativity_C": C,
        "n_th_BE": n_th,
        "SQL_aN": SQL_N * 1e18,
        "BA_optimum_aN": BA_N * 1e18,
    }
    for k, v in out.items():
        print(f"{k:>18} = {v:.6g}")

    ok = True
    for k, (want, rtol) in TARGETS.items():
        got = out[k]
        if abs(got - want) > rtol * abs(want):
            print(f"MISMATCH {k}: computed {got:.6g}, target {want} (+-{rtol*100:.0f}%)")
            ok = False
    if not ok:
        sys.exit(1)
    print("All derived rows match tab:si_device_params within tolerance.")


if __name__ == "__main__":
    main()
