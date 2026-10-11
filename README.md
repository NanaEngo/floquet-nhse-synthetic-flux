# floquet-nhse-synthetic-flux
In single-port non-Hermitian sensors, the Petermann factor that accompanies boundary-mode accumulation cancels the susceptibility gain, imposing a strict resource bound on metrological precision. We examine whether a multi-port architecture can evade this bound under full photon accounting.

---

# Track C — NHSE sensing (double-chain optomechanical ladder)

Goal: test whether a two-port double-chain ladder with opposing
non-reciprocal hoppings evades the single-port Petermann-factor resource
bound, via stability-constrained differentiable optimal control (DOC) of the
resource-normalized Fisher information.

## Canonical files

- `01_track_C_nhse_mainV2.tex` / `.pdf` — main article (source of truth).
- `02_track_C_nhse_SIV2.tex` / `.pdf` — Supplemental Material.
- `03_track_C_nhse_cover_letter.tex` / `.pdf` — PRX cover letter.
- `04_popular_summary.txt` — PRX Popular Summary.
- `references.bib`, `trackC_macros.tex` — bibliography and shared macros.
- `arxiv_submission/` — self-contained arXiv bundle (`main.tex`, `si.tex`
  are byte-synced mirrors of the two root sources; see its `README.md`).
- `PROVENANCE.md` — claim → data → reproducer map.
- `Track_C_data_analysis_report.md` — data analysis report (evidence record).

## Build order

From `arxiv_submission/` (see `arxiv_submission/README.md`):

```bash
export SOURCE_DATE_EPOCH=1788393600
pdflatex -interaction=nonstopmode -halt-on-error -jobname=si si.tex
bibtex si
pdflatex -interaction=nonstopmode -halt-on-error -jobname=si si.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=main main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -jobname=main main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=si si.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=main main.tex
```

After any root edit, re-sync mirrors (`diff -q` root vs `arxiv_submission/`
copies must be silent). Do NOT rebuild PDFs or edit `.tex` without
updating `PROVENANCE.md`/DAR first.

## Key producers (`scripts/` → `data/`)

| Producer | Role | Output |
|---|---|---|
| `doc_nscaling_hpc.py` | DOC N-scaling, N=6..16 even (Adam 500 steps, lr 0.003, barrier max Re λ < −0.05, seed 42) | `data/n_scaling_results_v3.json` |
| `doc_nscaling_intermediate.py` | Same protocol, N=7,9,11,13, 10 restarts each | `data/n_scaling_intermediate.json` |
| `validate_gain.py` | S-parameter gain validation (fail-closed gate) | stdout gate, no JSON |
| `derive_device_params.py` | MacCabe2020 platform → SI device table (fail-closed gate) | stdout gate, no JSON |
| `derive_axion_bound.py` | Conditional axion-photon projection from platform inputs | (see DAR §11 qualifications) |
| `classification_stability.py` | 10-restart seed-dependence per N, class counts + confusion matrix | `data/classification_stability.json` |
| `run_single_chain_doc.py` | Single-chain control grid (even authoritative, odd interpolated) | `data/single_chain_doc_control.json` |
| `regenerate_figures.py` | All retained figures from canonical JSON | `figures/*.pdf` |
| `generate_comparison_figure.py` | Single- vs double-chain comparison | `figures/single_vs_double_chain_norm.pdf` |

## Traceability (values read from JSON on 2026-10-11)

| Claim | Source |
|---|---|
| Single-chain ladder N=6..16, platform κ/2π=1.7 MHz, g₀/2π=0.9 MHz, Ωₘ/2π=5.0 GHz, m=1.0 fg, Qₘ=1.5e9, T_c=15 mK | `data/trackC_snr_decoupling.json` → `params.physical_device` |
| N=12 single-chain F_C^norm = 3.7251e-05 | `data/trackC_snr_decoupling.json` results |
| Fit slopes: SNR +0.172344 (VALIDATED), F_raw +2.461636 (VALIDATED), F_norm −0.762482 (FAIL_NORMALIZATION) | `data/trackC_snr_decoupling.json` fit objects |
| Six DOC optima N=6..16; N=12: F_norm=0.0504, maxRe=−0.0635, G_fwd=15.45 dB, iso=56.94 dB, all stable | `data/n_scaling_results_v3.json` |
| Resampling counts e.g. N=12: deep 2 / marginal 8 / unstable 0 (60 outcomes total) | `data/classification_stability.json` |
| MSS bound 1155.0 (κ units), OTOC N=6,8,10,12, max fraction 0.01737 | `data/trackC_physical_data.json` otoc_convergence |
| Control grid: 10 rows, N=6..14 + 16 (odd interpolated) | `data/single_chain_doc_control.json` |
| Intermediate scan N=7,9,11,13, best Fnorm 0.9861/0.9783/0.9853/0.9718 | `data/n_scaling_intermediate.json` |

Negative/qualified results are part of the evidence: normalized single-chain
slope < 0, bare-grid instability (`trackC_multichain_snr.json`), no JPA-backed
axion curve. See DAR §§10–11 before citing.

## Status / gates

- `mamba run -n qom python scripts/validate_gain.py` → exit 0 required.
- `mamba run -n qom python scripts/derive_device_params.py` → exit 0 required.
- Never commit, never touch `data/*.json`, `figures/`, or `.tex` without a
  verified producer run + DAR update first.
