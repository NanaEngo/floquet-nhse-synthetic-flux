#!/usr/bin/env python3
"""
Validate directional gain using proper input-output S-parameter formalism.
S = I - sqrt(2*kappa) * A^{-1},  G_fwd = |S_{N,1}|^2,  G_rev = |S_{1,N}|^2.
Uses numpy (no JAX needed for validation).
"""
import os, json
import numpy as np

KAPPA = 1.0; THETA0 = 0.1

def make_drift(n, J, g, lam, phi, theta):
    """Build the 2n x 2n non-Hermitian double-chain drift matrix (NumPy).

    Args:
        n: Number of sites per chain (int).
        J: Hopping amplitude in kappa units.
        g: Non-reciprocity, dimensionless.
        lam: Inter-chain coupling in kappa units.
        phi: Inter-chain phase in rad.
        theta: Signal detuning on the last site in rad.

    Returns:
        Complex drift matrix of shape (2*n, 2*n) in kappa units.
    """
    A = -KAPPA*np.eye(n, dtype=np.complex128)
    A[n-1,n-1] += -1j*theta
    for j in range(n-1):
        A[j+1,j] += J*np.exp(g)
        A[j,j+1] += J*np.exp(-g)
    B = -KAPPA*np.eye(n, dtype=np.complex128)
    for j in range(n-1):
        B[j+1,j] += J*np.exp(-g)
        B[j,j+1] += J*np.exp(g)
    L = lam*np.exp(1j*phi)*np.eye(n, dtype=np.complex128)
    return np.block([[A, L],[L.conj().T, B]])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "n_scaling_results_v3.json")) as f:
    results = json.load(f)

print("="*95)
print("Directional Gain Validation via Input-Output S-Parameter Formalism")
print("S = I - sqrt(2*kappa) * A^{-1},  G_fwd = 10*log10(|S_{N,1}|^2)")
print("="*95)
print(f"{'N':>3} {'J':>8} {'g':>8} {'lam':>8} {'phi':>8} | "
      f"{'|S_fwd|^2':>12} {'G_fwd(dB)':>10} {'|S_rev|^2':>12} {'G_rev(dB)':>10} {'Iso(dB)':>8} {'stable':>7}")
print("-"*95)

for r in results:
    N = r["N"]
    J, g, lam, phi = r["J"], r["g"], r["lam_inter"], r["phi"]
    A0 = make_drift(N, J, g, lam, phi, THETA0)
    max_re = np.max(np.linalg.eigvals(A0).real)

    A_inv = np.linalg.inv(A0)
    S = np.eye(2*N, dtype=np.complex128) - np.sqrt(2*KAPPA) * A_inv

    # Forward: Chain A site 1 -> Chain A site N (indices 0 -> N-1 in complex basis)
    S_fwd = np.abs(S[N-1, 0])**2
    G_fwd = 10.0 * np.log10(max(S_fwd, 1e-30))

    # Reverse: Chain A site N -> Chain A site 1 (indices N-1 -> 0)
    S_rev = np.abs(S[0, N-1])**2
    G_rev = 10.0 * np.log10(max(S_rev, 1e-30))

    iso = G_fwd - G_rev
    stable = "YES" if max_re < 0 else "NO"

    print(f"{N:>3} {J:>8.4f} {g:>8.3f} {lam:>8.4f} {phi:>8.3f} | "
          f"{S_fwd:>12.6e} {G_fwd:>10.2f} {S_rev:>12.6e} {G_rev:>10.2f} {iso:>8.2f} {stable:>7}")

print("="*95)

# DOC nominal (N=12)
print("\nDOC nominal (N=12): J=0.138, g=2.055, lam=0.010, phi=1.736")
A0 = make_drift(12, 0.138, 2.055, 0.010, 1.736, THETA0)
A_inv = np.linalg.inv(A0)
S = np.eye(24, dtype=np.complex128) - np.sqrt(2*KAPPA) * A_inv
S_fwd = np.abs(S[11, 0])**2
S_rev = np.abs(S[0, 11])**2
G_fwd = 10.0*np.log10(max(S_fwd, 1e-30))
G_rev = 10.0*np.log10(max(S_rev, 1e-30))
max_re = np.max(np.linalg.eigvals(A0).real)
print(f"  max_Re_lam = {max_re:.4f}")
print(f"  |S_fwd|^2 = {S_fwd:.6e}, G_fwd = {G_fwd:.2f} dB")
print(f"  |S_rev|^2 = {S_rev:.6e}, G_rev = {G_rev:.2f} dB")
print(f"  Isolation = {G_fwd - G_rev:.2f} dB")
