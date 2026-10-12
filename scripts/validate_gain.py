#!/usr/bin/env python3
"""
Validate directional gain using proper input-output S-parameter formalism.
S = I - sqrt(2*kappa) * A^{-1},  G_fwd = |S_{N,1}|^2,  G_rev = |S_{1,N}|^2.
Uses numpy (no JAX needed for validation).

Micro-case (C-2): when JAX is available, re-evaluates the N=6 published
optimum through the shared doc_core (the same code the producers import)
and runs a short deterministic Adam smoke test, so this gate exercises
the producer code path instead of only re-reading JSON. Without JAX the
NumPy S-table above still runs and the micro-case is skipped.
"""
import os, json, sys
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

print("\n" + "="*95)
print("DOC core micro-case (shared producer code path)")
print("="*95)
try:
    import doc_core as dc
    from jax import random as jrandom
    import functools
except ImportError as exc:
    print(f"  SKIP: JAX/doc_core unavailable ({exc}); NumPy table above stands alone.")
else:
    failures = []
    def check(name, got, want, rtol):
        """Compare a micro-case value against its reference (gate semantics).

        Args:
            name: Check label for the report.
            got: Recomputed value (dimensionless/dB as labeled).
            want: Reference value in the same unit.
            rtol: Relative tolerance (fraction).

        Returns:
            None. Records failures for the final gate verdict.
        """
        ok = abs(got - want) <= rtol * (abs(want) + 1e-12)
        print(f"  micro {name}: got {got:.6f} want {want:.6f} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    row6 = next(r for r in results if r["N"] == 6)
    x6 = dc.jnp.array(dc.map_back(
        [row6["J"], row6["g"], row6["lam_inter"], row6["phi"]]))
    _, _, f6, mr6, *_ = [float(v) for v in dc.compute_all(x6, 6)]
    g6 = [float(v) for v in dc.published_gains(
        row6["J"], row6["g"], row6["lam_inter"], row6["phi"], 6)]
    check("F_C_norm", f6, row6["F_C_norm"], 1e-3)
    check("max_Re_lam", mr6, row6["max_Re_lam"], 1e-4)
    check("gain_fwd", g6[0], row6["gain_fwd_dB"], 5e-2)
    check("gain_rev", g6[1], row6["gain_rev_dB"], 5e-2)
    check("isolation", g6[2], row6["isolation_dB"], 5e-2)
    # Adam smoke: 20 deterministic steps from a fixed key must not diverge.
    import jax
    grad_fn = jax.jit(jax.grad(functools.partial(dc.loss, N=6)))
    xs = jrandom.uniform(jrandom.PRNGKey(0), (4,), minval=-2.0, maxval=2.0)
    m = dc.jnp.zeros_like(xs)
    v = dc.jnp.zeros_like(xs)
    f0 = float(jax.jit(functools.partial(dc.loss, N=6))(xs))
    for step in range(20):
        g = grad_fn(xs)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        mh = m / (1 - 0.9 ** (step + 1))
        vh = v / (1 - 0.999 ** (step + 1))
        xs = xs - dc.LR * mh / (dc.jnp.sqrt(vh) + 1e-8)
    f1 = float(jax.jit(functools.partial(dc.loss, N=6))(xs))
    ok = bool(dc.jnp.isfinite(f1)) and f1 <= f0
    print(f"  micro adam_smoke: f0 {f0:.4f} -> f20 {f1:.4f} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("adam_smoke")
    if failures:
        print(f"ABORT: micro-case mismatch {failures}", file=sys.stderr)
        sys.exit(1)
    print("  DOC core micro-case: all PASS.")

print("\n" + "="*95)
print("Single-chain baseline guard (read-only reuse of run_single_chain_doc)")
print("="*95)
import run_single_chain_doc as scd
_src = json.load(open(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "trackC_snr_decoupling.json")))["results"]
_res = {r["N"]: r for r in _src}
if not scd.gate_authority(_res):
    print("ABORT: single-chain authority mismatch", file=sys.stderr)
    sys.exit(1)
print("  Single-chain baseline guard: all PASS.")
