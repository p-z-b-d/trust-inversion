# Trust Inversion: Composition Attacks on Hardware Malware Detection

Artifact for the paper *"Trust Inversion: Composition Attacks on Hardware Malware
Detection"* (USENIX Security '27 submission).

This repository contains the detector implementations, trust-aggregation schemes,
composition and attack simulation code, the T_meanvar_z defense, and the result
tables needed to reproduce the paper's core findings.

## Start here (for reviewers)

The paper's three headline results are precomputed and inspectable without running anything:

- **Composition grid** (detectors × attacks × trust schemes) — `results/phase4_grid.csv`
- **Trust inversion** (the stochastic detector D5 driven below the honest baseline under active collusion) — `results/phase4_D5_inversion.csv`
- **Mimicry threshold** (detection vs. per-edge noise, validating the predicted σ*) — `results/phase4_sigma_sweep.csv`

## Layout

| Directory        | Contents |
|------------------|----------|
| `data/`          | Trust-graph data. (Raw HPC traces: see **Dataset** below.) |
| `detectors/`     | The six hardware malware detectors D1-D6: RF (baseline, in `trust_schemes/`), `hmd_cnn.py` (D2), `hmd_rhmd.py` (D3), `hmd_mtd.py` (D4), `hmd_stochastic.py` (D5), `hmd_drl.py` (D6, with `d6_ucb.py` / `d6_adversarial.py` variants). |
| `measurement/`   | Raspberry Pi HPC collection (`pi_measure_full.py`), mimicry-signal measurement (`measure_sm.py`), consolidation, and runtime numbers (`runtime.txt`). |
| `trust_schemes/` | First-moment trust aggregators (mean, Subjective Logic, Beta, EigenTrust, PeerTrust) in `trust_baselines.py`; IoT trust evaluation in `trust_eval_iot.py`. The RF baseline detector (D1) lives here. |
| `composition/`   | Composition simulator (`phase4_sim.py`), grid driver, attack profiles AP1-AP6 (`attack_profiles.py`), optimal adversaries, and the adaptive-adversary experiments (`adaptive/`). |
| `defense/`       | The T_meanvar_z variance-aware defense (`phase4_temporal_defense.py`), per-defense sigma calibration, and O(k) cost analysis (`scheme_cost.py`). |
| `results/`       | Precomputed result tables (CSV) backing the paper's tables and figures. |
| `figures/`       | Scripts that regenerate the figures and result workbook. |

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- The composition experiments and figures run on any machine from the provided
  `results/` CSVs. The raw HPC traces were collected on a Raspberry Pi 4B
  (quad-core Cortex-A72); reproducing the measurement step requires that hardware.

## Reproducing the paper

```bash
pip install -r requirements.txt

# Composition grid (detectors x attacks x trust schemes)
python composition/phase4_sim.py              # -> results/phase4_grid.csv

# Headline: stochastic detector inverts under active collusion
python composition/grid_driver.py             # -> results/phase4_D5_inversion.csv

# Defense cost comparison
python defense/scheme_cost.py

# Regenerate figures from the result tables
python figures/generate_figures_v2.py
```

### Result-to-paper map

| Paper item                         | File / script |
|------------------------------------|---------------|
| Operating points & R_D             | `results/` + `figures/generate_figures_v2.py` |
| Composition grid table             | `results/phase4_grid.csv` |
| Collusion-tolerance (R_D) figure   | `figures/generate_figures_v2.py` |
| D5 inversion                       | `results/phase4_D5_inversion.csv` |
| Mimicry threshold sweep            | `results/phase4_sigma_sweep.csv` |
| Defense comparison                 | `figures/generate_defense_comparison_figure.py` |

*(Command/entry-point names follow the scripts as provided; adjust if your local
entry points differ. The mapping is what matters for evaluation.)*

## Dataset

The hardware performance-counter dataset consists of 238 traces across five
microarchitectural attack workloads (Spectre, cryptojacking, flush+reload, ROP,
encryption) and three benign workloads (compute, memory, webserver), collected on
a Raspberry Pi 4B (Cortex-A72). Each trace records per-window counter values
(instructions-per-cycle, cache-miss rate, branch-miss rate). Place the trace files
under `data/traces/`; the detector and measurement scripts read from that path.

`data/trust_graph.csv` contains the derived trust-graph edges used by the
composition experiments.

## License

Code released under the MIT License (see `LICENSE`). Dataset provided for research
use.
