# UAV Smoke Defense Mathematical Modeling Project

Clean GitHub-ready package containing the final codebase, experiment results, sensitivity analysis, and the paper PDF/public source set.

## Directory Structure

```text
uav-smoke-defense-model/
|-- code_and_results/
|   |-- M1_NaiveGA/            # M1: baseline round-robin allocation
|   |-- M2_Clustering/         # M2: spatiotemporal clustering
|   |-- M3_AmmoPenalty/        # M3: OT-style allocation with ammo penalty
|   |-- shared/                # shared physics, simulation, and optimizer modules
|   |-- data/                  # scenario data and missile pools
|   |-- all result/            # aggregated final experiment JSON results
|   |-- sensitivity_analysis/  # sensitivity-analysis scripts, figures, and databases
|   |-- tools/                 # result visualization scripts and exported figures
|   `-- docs/                  # project notes and technical documents
`-- paper/
    |-- manuscript.pdf         # latest paper PDF
    `-- source/                # public paper source without publisher template assets
```

## Environment

- Python 3.10+
- Main dependencies: `numpy`, `scipy`, `tqdm`

Install dependencies:

```bash
cd code_and_results
pip install -r requirements.txt
```

## Quick Start

Common entry files are kept in [`code_and_results`](./code_and_results):

- `M1_NaiveGA/main.py`
- `M2_Clustering/main.py`
- `M3_AmmoPenalty/main.py`
- `tools/visualize_results.py`
- `sensitivity_analysis/plot_sensitivity.py`

Shell helpers are also preserved:

- `run_M1.sh`
- `run_M2.sh`
- `run_M3.sh`
- `run_cloud.sh`

## Notes

- This package was cleaned for GitHub upload.
- Cache files, compiled files, LaTeX intermediate files, repeated zip packages, and other temporary artifacts were removed.
- The folder name [`all result`](./code_and_results/all%20result) is intentionally preserved because existing scripts depend on that path.
- Original project notes with Chinese filenames were moved into [`code_and_results/docs`](./code_and_results/docs) and renamed to clearer English filenames.
- The public paper source excludes publisher-specific template assets in `paper/source/Definitions`.


