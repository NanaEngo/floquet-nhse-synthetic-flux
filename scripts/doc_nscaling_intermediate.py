#!/usr/bin/env python3
"""
Track C DOC intermediate-size scan: N = 7, 9, 11, 13 (P1b).
Same protocol as doc_nscaling_hpc.py (Adam 500 steps, lr 0.003,
Hurwitz barrier -0.05): 10 restarts per size, per-restart Fnorm recorded
for class-coexistence analysis.

Model math and reported S-matrix gains are imported from doc_core.py;
recorded rows use the S convention (see published_gains).
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from jax import random
import json, time

from doc_core import (
    N_OPTIM,
    LR,
    unmap,
    compute_all,
    loss,
    published_gains,
)

import os as _os
Ns = [int(s) for s in _os.environ.get("DOC_SIZES", "7,9,11,13").split(",")]
N_RESTARTS = int(_os.environ.get("DOC_RESTARTS", "10"))

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

    F_C,norm2,F_norm,max_re,_,_,_ = fisher_fn(best_global_x)
    stable = bool(max_re < -0.01)  # strict stability gate
    p = unmap(best_global_x)
    # Recorded gains use the reported S-matrix convention so reruns
    # reproduce the published rows (never the T-matrix diagnostic).
    gf, gr, iso = published_gains(p[0], p[1], p[2], p[3], N)
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
