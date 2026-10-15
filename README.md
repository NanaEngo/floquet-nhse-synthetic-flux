# floquet-nhse-synthetic-flux

Public deposit for the PRX Quantum manuscript on two-port non-Hermitian
skin-effect sensing.

Can a two-port double-chain optomechanical ladder with opposing
non-reciprocal hoppings bypass the single-port Petermann-factor bound?
Yes, within the tested model: stability-constrained differentiable optimal
control finds two solution classes at opposite ends of a precision–gain
frontier set by the spectral margin — deep stability (median
F = 0.928 over 60 restarts) and marginal stability (median forward gain
+14.5 dB at F = 0.063) — and the deep-stability advantage survives
preamplifier noise at F_eff ≈ 0.3–0.5.

## Contents

| Path | Role |
|---|---|
| `01_track_C_nhse_main.tex` / `.pdf` | Main article (`apsrev4-2`) |
| `02_track_C_nhse_SI.tex` / `.pdf` | Supplemental Material |
| `03_track_C_nhse_cover_letter.tex` / `.pdf` | Cover letter |
| `04_popular_summary.txt` | Popular summary |
| `references.bib` | Bibliography |
| `figures/` | Manuscript and SI figures |
| `scripts/` | Deterministic analysis scripts |
| `environment.yml` | Environment specification |

Manuscript sources, deterministic analysis scripts, and the environment
specification are archived here; simulation data files are not uploaded.
