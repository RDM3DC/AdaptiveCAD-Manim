# Adaptive Chern Self-Healing Conductance Law — Verified Benchmark Release

**Submission name:** Adaptive Chern Self-Healing Conductance Law  
**Version:** 1.0.0  
**Repository:** [github.com/RDM3DC/AdaptiveCAD-Manim](https://github.com/RDM3DC/AdaptiveCAD-Manim)  
**Commit:** See `outputs/manifest.json` for the current git hash.  
**Date:** 2026-03-09  

---

## 1. Law Statement (EGATL — Edge-Gated Adaptive Topological Law)

On a square lattice with edge conductances $g_e \in \mathbb{C}$ and lifted phases $\theta_e$, the dynamics are:

$$
\frac{dg_e}{dt}
  = \underbrace{\alpha_G \, j_e \, e^{i\theta_e}}_{\text{current-driven growth}}
  - \underbrace{\mu_G \, g_e}_{\text{decay}}
  - \underbrace{\lambda_s \, g_e \sin^2\!\!\left(\frac{\theta_e}{2\pi_a}\right)}_{\text{winding suppression}}
  + \underbrace{\chi \, c_{\text{loc},e} \, g_e}_{\text{topological healing}}
$$

where:

| Symbol | Definition | Default |
|--------|-----------|---------|
| $\alpha_G$ | Adaptive gain: $\alpha_0 (1 + \alpha_{\text{gain}} \tanh(1.4(s-1)))$ | $\alpha_0 = 0.66$ |
| $\mu_G$ | Adaptive decay: $\mu_0 (1 - \mu_{\text{relief}} \tanh(1.1(s-1)))$ | $\mu_0 = 0.22$ |
| $\pi_a$ | Adaptive ruler: $\pi_{a,c} (1 + \sigma_\pi \tanh(s-1))$, clipped to $[\pi_{a,\min}, \pi_{a,\max}]$ | $\pi_{a,c} = \pi$ |
| $\theta_e$ | Phase-lifted winding: $\theta_{\text{prev}} + \text{clip}(\text{wrap}(\theta_{\text{raw}} - \theta_{\text{prev}}), -\pi_a, \pi_a)$ | — |
| $j_e$ | Edge current: $\text{drive} \cdot \text{channel\_bias} \cdot \text{coherence\_bias} \cdot \|g_e\|$ | — |
| $c_{\text{loc},e}$ | Local Chern proxy: $\frac{1}{2}(\text{phase\_alignment} + \text{transport\_gap})$ on boundary | — |
| $s$ | State scalar: $\overline{\|g\|} / g_0$ | $g_0 = 0.34$ |

**Phase-lift** clips per-step phase jumps to the adaptive ruler $\pi_a$, preventing $2\pi$ ambiguity from corrupting winding-dependent feedback.

**Local Chern proxy** combines boundary phase alignment $\frac{1}{2}(1 + \cos(\theta_e - \theta_{\partial,\text{ref}}))$ with transport gap $\tanh((\bar{j}_\partial - \bar{j}_{\text{bulk}}) / |\bar{j}_{\text{bulk}}|)$.

---

## 2. Benchmark Results

### 2.1 Recovery Demo (4 variants, 8×8 lattice, seed=7)

Damage applied at step 170 (t = 5.1 s), scale = 0.14.

| Variant | Boundary Fraction | Edge/Bulk Ratio | Coherence | First-Hit Recovery | Rolling Recovery (0.6 s) |
|---------|:-:|:-:|:-:|:-:|:-:|
| **full_law** | 0.481 | 1.015 | **0.9999** | — | — |
| principal_branch | **0.870** | **9.334** | 0.438 | **2.46 s** | **2.46 s** |
| no_topology_feedback | 0.478 | 1.003 | 1.000 | — | — |
| fixed_ruler | 0.485 | 1.049 | 1.000 | — | — |

### 2.2 Matched-Present Ablation (shared damaged snapshot)

| Variant | Boundary Fraction | Edge/Bulk Ratio | First-Hit Recovery | Rolling Recovery (0.6 s) |
|---------|:-:|:-:|:-:|:-:|
| **full_law** | 0.481 | 1.015 | — | — |
| principal_branch | 0.755 | 3.213 | **9.0 s** | — |
| no_topology_feedback | 0.478 | 1.006 | — | — |
| fixed_ruler | 0.485 | 1.048 | — | — |

The rolling-window metric is stricter than the old first-hit metric: it requires the boundary fraction to stay above target for a sustained 0.6 s window. Under this criterion, the matched-present principal-branch run does not count as recovered.

### 2.3 Solver Verification — 5/5 PASS

| Test | Result | Key Metric |
|------|:------:|------------|
| Determinism | PASS | Exact bitwise reproduction |
| Current conservation | PASS | max ratio error = 0.0 |
| Phase-lift clipping | PASS | coherence ∈ [0.9999, 1.0], $\pi_a$ bounded |
| Solver cross-validation | PASS | GMRES: 0 failures, $Y_{\text{eff}}$ recovery ratio = 44.2 |
| Lattice symmetry | PASS | row symmetry error = 0.005 |

### 2.4 Onset-Map Parameter Sweeps (8×8 grid of params)

| Sweep | Recovery Fraction | BF Range |
|-------|:-:|:-:|
| $\alpha_0 \times \lambda_s$ | 43.8% | [0.412, 0.942] |
| $\chi \times \text{damage\_scale}$ | 59.4% | [0.423, 0.991] |

### 2.5 Units/Theory Verification

| Check | Result | Key Metric |
|-------|:------:|------------|
| Units | OK | normalized, finite, bounded $dt\,dg$ increment |
| Theory | PASS | $\chi_\text{high} > \chi_\text{low}$, $\alpha_{0,\text{high}} > \alpha_{0,\text{low}}$, damage and suppression act in the expected direction |

---

## 3. Reproducing Results

### Install

```bash
pip install -e .
```

### Run individual benchmarks

```bash
python benchmarks/run_recovery_demo.py        # → outputs/recovery_demo/
python benchmarks/run_matched_present.py       # → outputs/matched_present/
python benchmarks/run_solver_verification.py   # → outputs/solver_verification/
python benchmarks/run_onset_map.py             # → outputs/onset_map/
python benchmarks/run_units_and_theory_check.py  # → outputs/units_theory/
```

### Run full pipeline (all 5 stages + manifest)

```bash
python benchmarks/run_full_pipeline.py         # → outputs/manifest.json
```

All benchmarks are deterministic (seed=7). The full pipeline produces `outputs/manifest.json` with timestamps, git hash, and SHA256 checksums for every artifact.

---

## 4. Output Artifacts (17 files)

| Directory | Files | Description |
|-----------|-------|-------------|
| `outputs/recovery_demo/` | `recovery_traces.csv`, `summary.json`, `recovery_traces.png`, 3× snapshot PNGs | 4-variant time series + lattice snapshots |
| `outputs/matched_present/` | `matched_present_summary.csv`, `.json`, `matched_present_traces.png` | Ablation from shared damaged state |
| `outputs/onset_map/` | 2× CSV, 2× PNG, `summary.json` | $\alpha_0 \times \lambda_s$ and $\chi \times \text{damage\_scale}$ sweeps |
| `outputs/solver_verification/` | `verdict.json` | 5-test verification suite |
| `outputs/units_theory/` | `verdict.json` | Units: OK, Theory: PASS artifact |
| `outputs/` | `manifest.json` | Pipeline manifest with SHA256 hashes |

---

## 5. Architecture

```
src/arp_topology/          ← installable package
    __init__.py
    laws.py                ← EdgeLattice, ModelParams, EGATL dynamics, simulate()

solver/                    ← 6 GMRES-based benchmark modules
    egatl.py               ← block QWZ (original)
    scalar_flux.py         ← 2D Harper-Hofstadter flux lattice
    ssh.py                 ← 1D SSH chain
    kitaev.py              ← Kitaev p-wave chain (Nambu)
    haldane.py             ← Haldane honeycomb lattice
    rice_mele.py           ← Rice-Mele chain (Zak phase)

benchmarks/                ← deterministic benchmark scripts
    run_recovery_demo.py
    run_matched_present.py
    run_solver_verification.py
    run_onset_map.py
    run_units_and_theory_check.py
    run_full_pipeline.py

outputs/                   ← reproducible artifacts (CSV, JSON, PNG)
```

---

## 6. Key Parameters

```python
ModelParams(
    steps=480, dt=0.03,
    damage_step=170, damage_scale=0.14,
    drive=1.0, g0=0.34, g_max=2.40,
    alpha0=0.66, alpha_gain=0.26,
    mu0=0.22, mu_relief=0.08,
    lambda_s=0.92, chi=0.48,
    pi_a_center=π, pi_a_min=0.62π, pi_a_max=1.25π, pi_a_sensitivity=0.22,
    phase_advance=0.34, damage_phase_kick=1.10,
    boundary_drive_bias=0.72, coherence_gain=0.60,
    recovery_target_fraction=0.90, seed=7,
)
```

---

## 7. Verification Checksums

All artifact SHA256 hashes are recorded in `outputs/manifest.json`. To verify:

```bash
python -c "
import json, hashlib, pathlib
m = json.loads(pathlib.Path('outputs/manifest.json').read_text())
for a in m['artifacts']:
    h = hashlib.sha256(pathlib.Path(a['path']).read_bytes()).hexdigest()
    ok = '✓' if h == a['sha256'] else '✗'
    print(f'{ok} {a[\"path\"]}')"
```
