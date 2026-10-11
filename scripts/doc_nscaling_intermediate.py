#!/usr/bin/env python3
"""
Track C DOC intermediate-size scan: N = 7, 9, 11, 13 (P1b).
Same protocol as doc_nscaling_hpc.py (Adam 500 steps, lr 0.003,
Hurwitz barrier -0.05): 10 restarts per size, per-restart Fnorm recorded
for class-coexistence analysis.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from jax import random
import json, time

KAPPA = 1.0; THETA0 = 0.1; DTHETA = 1e-4
import os as _os
Ns = [int(s) for s in _os.environ.get("DOC_SIZES", "7,9,11,13").split(",")]
N_OPTIM = 500; LR = 0.003
N_RESTARTS = int(_os.environ.get("DOC_RESTARTS", "10"))

J_LO, J_HI = 0.05, 0.5
g_LO, g_HI = 0.5, 3.0
L_LO, L_HI = 0.005, 0.1
PHI_LO, PHI_HI = 0.0, 6.2832

def unmap(x):
    """Map unconstrained optimizer variables to physical box bounds.

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
        return 1.0/(1.0+jnp.exp(-z))
    return jnp.array([
        J_LO  + (J_HI  - J_LO)  * sig(x[0]),
        g_LO  + (g_HI  - g_LO)  * sig(x[1]),
        L_LO  + (L_HI  - L_LO)  * sig(x[2]),
        PHI_LO + (PHI_HI - PHI_LO) * sig(x[3])
    ])

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
    A = -KAPPA*jnp.eye(n, dtype=jnp.complex128)
    A = A.at[n-1,n-1].add(-1j*theta)
    for j in range(n-1):
        A = A.at[j+1,j].add(J*jnp.exp(g))
        A = A.at[j,j+1].add(J*jnp.exp(-g))
    B = -KAPPA*jnp.eye(n, dtype=jnp.complex128)
    for j in range(n-1):
        B = B.at[j+1,j].add(J*jnp.exp(-g))
        B = B.at[j,j+1].add(J*jnp.exp(g))
    L = lam*jnp.exp(1j*phi)*jnp.eye(n, dtype=jnp.complex128)
    return jnp.block([[A, L],[L.conj().T, B]])

def realify(A):
    """Embed a complex matrix as a real matrix of doubled dimension.

    Args:
        A: Complex array of shape (m, m).

    Returns:
        Real array of shape (2*m, 2*m) with the same spectrum up to doubling.
    """
    return jnp.block([[A.real,-A.imag],[A.imag,A.real]])

def solve_lyap(Ar):
    """Solve the continuous Lyapunov equation for the steady-state covariance.

    Args:
        Ar: Real drift matrix of shape (n, n) in kappa units.

    Returns:
        Covariance matrix of shape (n, n) in photon-number units (dimensionless).
    """
    n = Ar.shape[0]
    L = jnp.kron(jnp.eye(n),Ar)+jnp.kron(Ar,jnp.eye(n))
    return jnp.linalg.solve(L, -(2*KAPPA*jnp.eye(n)).flatten()).reshape(n,n)

def compute_all(x, N):
    """Evaluate raw/normalized Fisher information, stability and gains.

    Args:
        x: Unconstrained parameter vector of shape (4,), dimensionless.
        N: Array size per chain (int).

    Returns:
        Tuple (F_C, norm2, F_norm, max_re, gf, gr, iso) with F_C raw Fisher
        in rad^-2, norm2 steady-state photon number (dimensionless), F_norm
        normalized Fisher in rad^-2 per photon, max_re spectral abscissa in
        kappa units, gf/gr forward/reverse gains in dB, iso isolation in dB.
    """
    p = unmap(x); J,g,lam,phi = p; n=int(N)
    A0 = make_drift(n,J,g,lam,phi,THETA0)
    Ar0 = realify(A0)
    max_re = jnp.max(jnp.linalg.eigvals(A0).real)
    Sig = solve_lyap(Ar0)
    nn = Ar0.shape[0]
    u = jnp.zeros(nn).at[2*(n-1)].set(1.0)
    mu = -jnp.linalg.solve(Ar0, u)
    norm2 = jnp.sum(jnp.abs(mu)**2)
    dAr = jnp.zeros((nn,nn))
    idx = 2*(n-1)
    dAr = dAr.at[idx,idx+1].set(-1.0)
    dAr = dAr.at[idx+1,idx].set(1.0)
    dmu = -jnp.linalg.solve(Ar0, dAr@jnp.linalg.solve(Ar0,u))
    Sig_inv = jnp.linalg.inv(Sig)
    F_mean = dmu@Sig_inv@dmu
    Lm = jnp.kron(jnp.eye(nn),Ar0)+jnp.kron(Ar0,jnp.eye(nn))
    dSig = jnp.linalg.solve(Lm, -(dAr@Sig+Sig@dAr.T).flatten()).reshape(nn,nn)
    X = Sig_inv@dSig
    F_cov = 0.5*jnp.trace(X@X)
    F_C = F_mean+F_cov
    F_norm = F_C/norm2
    T = jnp.linalg.inv(-A0)
    Sf = jnp.abs(T[n-1,0])**2+jnp.abs(T[n-1+n,0])**2
    Sr = jnp.abs(T[0,n-1])**2+jnp.abs(T[0+n,n-1])**2
    gf = 10.0*jnp.log10(jnp.maximum(Sf,1e-30))
    gr = 10.0*jnp.log10(jnp.maximum(Sr,1e-30))
    return F_C, norm2, F_norm, max_re, gf, gr, gf-gr

def loss(x, N):
    """Production loss: negative normalized Fisher plus stability barrier.

    Args:
        x: Unconstrained parameter vector of shape (4,), dimensionless.
        N: Array size per chain (int).

    Returns:
        Scalar loss, dimensionless (barrier active when max_re > -0.05 kappa).
    """
    F_C, norm2, F_norm, max_re, *_ = compute_all(x, N)
    # Strong barrier: penalize any max_re > -0.05 (stay well inside stable region)
    pen = jnp.maximum(max_re + 0.05, 0.0) * 1e8
    return -F_norm + pen

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "n_scaling_intermediate.json")
RESUME = _os.environ.get("DOC_RESUME", "0") == "1"
results = []
if RESUME and os.path.exists(out):
    try:
        results = json.load(open(out))
        done = {r["N"] for r in results}
        Ns = [N for N in Ns if N not in done]
        print(f"resume: skipping completed sizes {sorted(done)}", flush=True)
    except Exception:
        results = []
key = random.PRNGKey(20260823)
for N in Ns:
    t0 = time.time()
    grad_fn = jax.jit(jax.grad(lambda x: loss(x, N)))
    loss_fn = jax.jit(lambda x: loss(x, N))
    fisher_fn = jax.jit(lambda x: compute_all(x, N))

    best_global_x = None
    best_global_f = 1e30

    restart_fnoms = []
    for restart in range(N_RESTARTS):
        key, sub = random.split(key)
        x = random.uniform(sub, (4,), minval=-2.0, maxval=2.0)
        m=jnp.zeros_like(x); v=jnp.zeros_like(x)
        best_x, best_f = x, 1e30
        for step in range(N_OPTIM):
            g = grad_fn(x)
            m = 0.9*m+0.1*g; v = 0.999*v+0.001*g**2
            mh = m/(1-0.9**(step+1)); vh = v/(1-0.999**(step+1))
            x = x - LR*mh/(jnp.sqrt(vh)+1e-8)
            if (step+1)%100==0:
                lv = float(loss_fn(x))
                if lv < best_f:
                    best_f = lv; best_x = x.copy()
        if best_f < best_global_f:
            best_global_f = best_f; best_global_x = best_x.copy()
        lv_restart = float(loss_fn(best_x))
        rf, _, rf_norm, *_ = fisher_fn(best_x)
        restart_fnoms.append(float(rf_norm))

    F_C,norm2,F_norm,max_re,gf,gr,iso = fisher_fn(best_global_x)
    stable = bool(max_re < -0.01)  # strict stability gate
    p = unmap(best_global_x)
    dt = round(time.time()-t0,1)
    r = {"N":int(N),"J":float(p[0]),"g":float(p[1]),"lam_inter":float(p[2]),"phi":float(p[3]),
         "F_C_raw":float(F_C),"norm2":float(norm2),"F_C_norm":float(F_norm),
         "max_Re_lam":float(max_re),"stable":stable,
         "gain_fwd_dB":float(gf),"gain_rev_dB":float(gr),"isolation_dB":float(iso),"time_s":dt,
     "restart_Fnorms":restart_fnoms,"n_restarts":N_RESTARTS,"seed":20260823}
    results.append(r)
    with open(out,"w") as f: json.dump(results, f, indent=2)
    print(f"N={N}: F_C^norm={r['F_C_norm']:.4f} J={r['J']:.4f} g={r['g']:.4f} "
          f"lam={r['lam_inter']:.4f} phi={r['phi']:.4f} "
          f"max_Re={r['max_Re_lam']:.4f} iso={r['isolation_dB']:.1f}dB "
          f"stable={r['stable']} [{dt}s]", flush=True)

print(f"\nSaved: {out}")
