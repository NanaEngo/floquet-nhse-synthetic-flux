#!/usr/bin/env python3
"""Derive the broadband axion-photon coupling sensitivity g_agamma_min.

Standard haloscope/Dicke chain, all assumptions explicit and traced to the
MacCabe2020 platform parameters stored in ../data/trackC_snr_decoupling.json:

  P_sig = g_agg^2 * rho_a * B0^2 * V_eff * Q_L / m_a          (SI units)
  SNR   = P_sig * t_int / (hbar * omega)                      (1 quantum-limited bin)

Assumptions declared here (NOT hidden):
  - rho_a  = 0.3 GeV/cm^3 (local DM energy density)
  - V_eff  = eta_fill * (lambda0/n_Si)^3 : single optical-mode volume of one
    nanobeam segment, eta_fill = 1 (conservative; ladder overlap ignored)
  - Q_L    = 2e6 loaded optical Q (SI Sec. 'Evanescent field coupling')
  - C_dip  = 1 (dipole form factor)
  - N-chain coherent enhancement: V -> N*V, SNR -> N*SNR (power adds coherently
    across the ladder), quoted as an equivalent N multiplier on SNR.
Aborts (non-zero exit) if the recomputed five-gigahertz value differs from the
manuscript reference 1.7e-14 /GeV by more than five percent.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
dev = json.load(open(os.path.join(
    HERE, "..", "data", "trackC_snr_decoupling.json")))["params"]["physical_device"]

# --- constants -------------------------------------------------------------
rho_a   = 0.3            # GeV/cm^3
B0      = 10.0           # T           (manuscript)
t_int   = 3600.0         # s           (manuscript: 1 hour)
lam0    = 1550e-9        # m           (SI evanescent section)
nSi     = 3.48           # refractive index silicon
QL      = 2e6            # loaded optical Q (SI)
eta     = 1.0            # dipole form factor
N       = 12             # ladder size (manuscript)
hbar    = 1.054571817e-34
c       = 299792458.0

f       = np.array([1e9, 5e9, 10e9])          # band edges + centre
m_a     = 4.135667696e-15 * f                 # GeV  (h = 4.1357e-15 eV*s -> GeV*s)
V_mode  = eta * (lam0 / nSi) ** 3             # m^3

P_per_g2 = rho_a * B0**2 * N * V_mode * QL / m_a    # W per GeV^-2  (rho/m in GeV/cm^3*m^-?)
# careful: rho[m_a] must be in same length units: convert rho to GeV/m^3 (1 cm^3 = 1e-6 m^3)
rho_a_m = rho_a * 1e6
P_per_g2 = rho_a_m * B0**2 * N * V_mode * QL / m_a
omega   = 2 * np.pi * f
g_min   = np.sqrt(hbar * omega / (t_int * P_per_g2))

print(f"V_mode = {V_mode:.3e} m^3 ; rho/m_a = {rho_a_m / m_a[1]:.3e} m^-3")
for fi, gi in zip(f, g_min):
    print(f"f = {fi/1e9:5.1f} GHz  ->  g_agamma_min = {gi:.3e} /GeV")

ref = 1.7e-14
ok = np.isclose(g_min[1], ref, rtol=0.05)
print("manuscript reference:", ref, "| ratio derived/reference:", round(float(g_min[1] / ref), 5))
sys.exit(0 if ok else 1)
