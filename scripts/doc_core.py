#!/usr/bin/env python3
"""Shared differentiable-optimal-control (DOC) core for Track C producers.

Single source of truth for the double-chain drift model, the record Fisher
information, and the production loss. Imported by ``doc_nscaling_hpc.py``,
``doc_nscaling_intermediate.py``, ``classification_stability.py`` (JAX path)
and exercised by the ``validate_gain.py`` gate, so gates test the same code
that produced the published artifacts.

Model conventions (frozen production protocol): ``KAPPA = 1`` frequency unit,
``THETA0 = 0.1`` d-theta probe, Adam 500 steps at lr 0.003, Hurwitz barrier
``max Re lambda < -0.05`` in kappa units. Parameter box bounds are part of
the model: changing them changes published optima.

Requires JAX (CPU, 64-bit). Raises a descriptive ImportError if JAX is
missing instead of failing obscurely.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ImportError as exc:  # ponytail: fail fast with the install hint
    raise ImportError(
        "Track C DOC core requires JAX (e.g. `mamba run -n qom python -m "
        "pip install jax jaxlib`). The NumPy-only validate_gain.py path "
        "does not need it."
    ) from exc

import math

# Production protocol constants (kappa frequency units unless noted).
KAPPA = 1.0
THETA0 = 0.1
N_OPTIM = 500
LR = 0.003
BARRIER = -0.05  # Hurwitz barrier on max Re lambda, in kappa units
BARRIER_PENALTY = 1e8

# Parameter box bounds (J, lam in kappa units; g dimensionless; phi in rad).
J_LO, J_HI = 0.05, 0.5
g_LO, g_HI = 0.5, 3.0
L_LO, L_HI = 0.005, 0.1
PHI_LO, PHI_HI = 0.0, 6.2832


def unmap(x):
    """Map unconstrained optimizer variables to physical parameters.

    Args:
        x: Unconstrained vector of shape (4,), dimensionless.

    Returns:
        Array [J, g, lam, phi] with J in kappa units, g dimensionless,
        lam in kappa units, phi in rad.
    """
    def sig(z):
        """Logistic sigmoid mapping a real scalar to (0, 1).

        Args:
            z: Real scalar, dimensionless.

        Returns:
            Sigmoid value in (0, 1), dimensionless.
        """
        return 1.0 / (1.0 + jnp.exp(-z))
    return jnp.array([
        J_LO + (J_HI - J_LO) * sig(x[0]),
        g_LO + (g_HI - g_LO) * sig(x[1]),
        L_LO + (L_HI - L_LO) * sig(x[2]),
        PHI_LO + (PHI_HI - PHI_LO) * sig(x[3]),
    ])


def map_back(p):
    """Invert unmap: physical parameters -> unconstrained variables.

    Args:
        p: Sequence [J, g, lam, phi] inside the box bounds (units as
            in unmap).

    Returns:
        List of 4 unconstrained logit values, dimensionless.
    """
    def logit(lo, hi, v):
        """Map a bounded physical value back to an unconstrained variable.

        Args:
            lo: Lower box bound (same unit as v).
            hi: Upper box bound (same unit as v).
            v: Physical value inside (lo, hi).

        Returns:
            Unconstrained logit value, dimensionless.
        """
        u = min(max((v - lo) / (hi - lo), 1e-9), 1.0 - 1e-9)
        return math.log(u / (1.0 - u))
    return [
        logit(J_LO, J_HI, p[0]),
        logit(g_LO, g_HI, p[1]),
        logit(L_LO, L_HI, p[2]),
        logit(PHI_LO, PHI_HI, p[3]),
    ]


def make_drift(n, J, g, lam, phi, theta):
    """Build the 2n x 2n non-Hermitian double-chain drift matrix.

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
    A = -KAPPA * jnp.eye(n, dtype=jnp.complex128)
    A = A.at[n - 1, n - 1].add(-1j * theta)
    for j in range(n - 1):
        A = A.at[j + 1, j].add(J * jnp.exp(g))
        A = A.at[j, j + 1].add(J * jnp.exp(-g))
    B = -KAPPA * jnp.eye(n, dtype=jnp.complex128)
    for j in range(n - 1):
        B = B.at[j + 1, j].add(J * jnp.exp(-g))
        B = B.at[j, j + 1].add(J * jnp.exp(g))
    L = lam * jnp.exp(1j * phi) * jnp.eye(n, dtype=jnp.complex128)
    return jnp.block([[A, L], [L.conj().T, B]])


def realify(A):
    """Embed a complex matrix into a real matrix of doubled dimension.

    Args:
        A: Complex array of shape (m, m).

    Returns:
        Real array of shape (2*m, 2*m) with the same spectrum up to doubling.
    """
    return jnp.block([[A.real, -A.imag], [A.imag, A.real]])


def solve_lyap(Ar):
    """Solve the continuous Lyapunov equation for the covariance.

    Args:
        Ar: Real drift matrix of shape (m, m) in kappa units.

    Returns:
        Covariance matrix of shape (m, m) in photon-number units
        (dimensionless).
    """
    m = Ar.shape[0]
    Lop = jnp.kron(jnp.eye(m), Ar) + jnp.kron(Ar, jnp.eye(m))
    return jnp.linalg.solve(Lop, -(2 * KAPPA * jnp.eye(m)).flatten()).reshape(m, m)


def compute_all(x, N):
    """Evaluate record Fisher information metrics for parameters x.

    Args:
        x: Unconstrained parameter vector of shape (4,), dimensionless.
        N: Array size per chain (int).

    Returns:
        Tuple (F_C, norm2, F_norm, max_re, gf, gr, iso) with F_C raw
        Fisher in rad^-2, norm2 steady-state photon number
        (dimensionless), F_norm normalized Fisher in rad^-2 per photon,
        max_re spectral abscissa in kappa units, gf/gr forward/reverse
        gains in dB, iso isolation in dB.
    """
    p = unmap(x)
    J, g, lam, phi = p
    n = int(N)
    A0 = make_drift(n, J, g, lam, phi, THETA0)
    Ar0 = realify(A0)
    max_re = jnp.max(jnp.linalg.eigvals(A0).real)
    Sig = solve_lyap(Ar0)
    nn = Ar0.shape[0]
    u = jnp.zeros(nn).at[2 * (n - 1)].set(1.0)
    mu = -jnp.linalg.solve(Ar0, u)
    norm2 = jnp.sum(jnp.abs(mu) ** 2)
    dAr = jnp.zeros((nn, nn))
    idx = 2 * (n - 1)
    dAr = dAr.at[idx, idx + 1].set(-1.0)
    dAr = dAr.at[idx + 1, idx].set(1.0)
    dmu = -jnp.linalg.solve(Ar0, dAr @ jnp.linalg.solve(Ar0, u))
    Sig_inv = jnp.linalg.inv(Sig)
    F_mean = dmu @ Sig_inv @ dmu
    Lm = jnp.kron(jnp.eye(nn), Ar0) + jnp.kron(Ar0, jnp.eye(nn))
    dSig = jnp.linalg.solve(Lm, -(dAr @ Sig + Sig @ dAr.T).flatten()).reshape(nn, nn)
    X = Sig_inv @ dSig
    F_cov = 0.5 * jnp.trace(X @ X)
    F_C = F_mean + F_cov
    F_norm = F_C / norm2
    T = jnp.linalg.inv(-A0)
    Sf = jnp.abs(T[n - 1, 0]) ** 2 + jnp.abs(T[n - 1 + n, 0]) ** 2
    Sr = jnp.abs(T[0, n - 1]) ** 2 + jnp.abs(T[0 + n, n - 1]) ** 2
    gf = 10.0 * jnp.log10(jnp.maximum(Sf, 1e-30))
    gr = 10.0 * jnp.log10(jnp.maximum(Sr, 1e-30))
    return F_C, norm2, F_norm, max_re, gf, gr, gf - gr


def loss(x, N):
    """Production loss: negative normalized FI plus stability barrier.

    Args:
        x: Unconstrained parameter vector of shape (4,), dimensionless.
        N: Array size per chain (int).

    Returns:
        Scalar loss, dimensionless (barrier active when max_re exceeds
        BARRIER in kappa units).
    """
    _, _, F_norm, max_re, *_ = compute_all(x, N)
    pen = jnp.maximum(max_re - BARRIER, 0.0) * BARRIER_PENALTY
    return -F_norm + pen


def s_gains(A0, n):
    """Directional gains in the reported scattering-matrix convention.

    S = I - sqrt(2*kappa) * A0^{-1}, G_fwd = |S_{N,1}|^2,
    G_rev = |S_{1,N}|^2. This is the convention of every published gain
    (JSON artifacts, manuscript, SI); the T-matrix power transmission
    inside compute_all is a diagnostic only and must not be published.

    Args:
        A0: Complex drift matrix of shape (2*n, 2*n) in kappa units.
        n: Number of sites per chain (int).

    Returns:
        Tuple (gf, gr, iso) with forward/reverse gains and isolation
        in dB.
    """
    S = jnp.eye(2 * n, dtype=jnp.complex128) - jnp.sqrt(2 * KAPPA) * jnp.linalg.inv(A0)
    s_fwd = jnp.abs(S[n - 1, 0]) ** 2
    s_rev = jnp.abs(S[0, n - 1]) ** 2
    gf = 10.0 * jnp.log10(jnp.maximum(s_fwd, 1e-30))
    gr = 10.0 * jnp.log10(jnp.maximum(s_rev, 1e-30))
    return gf, gr, gf - gr


def published_gains(J, g, lam, phi, N):
    """Reported gains for physical parameters at size N.

    Convenience wrapper building the drift matrix at the production probe
    THETA0 and returning s_gains. Producers must record these (not the
    T-matrix diagnostic) so reruns reproduce the published JSON rows.

    Args:
        J: Hopping amplitude in kappa units.
        g: Non-reciprocity, dimensionless.
        lam: Inter-chain coupling in kappa units.
        phi: Inter-chain phase in rad.
        N: Array size per chain (int).

    Returns:
        Tuple (gf, gr, iso) in dB (see s_gains).
    """
    A0 = make_drift(int(N), J, g, lam, phi, THETA0)
    return s_gains(A0, int(N))
