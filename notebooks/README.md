# Notebooks Directory

Jupyter notebooks for exploration, experimentation, and analysis.

## Purpose

Notebooks are for:
- Data exploration and visualization
- Prototyping models before implementing in `src/`
- Analyzing training results
- Creating visualizations for reports

## NOT for:

- Production code (goes in `src/`)
- Training final models (use `scripts/train.py`)

## Naming Convention

Use numbered prefixes to show sequence:
- `01_data_exploration.ipynb`
- `02_model_prototyping.ipynb`
- `03_results_analysis.ipynb`

## To use notebooks:

```bash
# Activate venv first
source venv/bin/activate

# Install jupyter (if not already)
pip install jupyter

# Launch
jupyter notebook
```
