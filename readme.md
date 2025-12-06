
## What each folder contains

### `src/`
Primary Python package for this project.

Contains:
- model loading and initialization
- LoRA/QLoRA helpers
- dataset loading and preprocessing
- training orchestration
- inference and scoring
- shared utilities

All logic used by notebooks should live here.

---

### `notebooks/`
Experiment and execution notebooks.

Contains:
- `train.ipynb`: calls training functions from `src/`
- `score.ipynb`: calls scoring/evaluation functions from `src/`

Notebooks should focus on:
- wiring inputs/outputs
- quick debugging
- visualization

---

### `configs/`
Configuration files for reproducible experiments.

Contains:
- training hyperparameters
- model names and checkpoints
- dataset paths and splits
- LoRA settings
- inference/scoring settings

Recommended formats:
- YAML or JSON

---

### `scripts/`
Optional command-line entry points.

Contains:
- `train.py`: runs training using `configs/`
- `score.py`: runs evaluation using `configs/`

Purpose:
- allow running without notebooks
- simplify Colab → VM workflows
- support automation and CI

---

### `data/`
Local data storage.

Contains:
- `raw/`: original datasets
- `processed/`: tokenized/cleaned outputs

Typically gitignored. Optionally include small sample data or a schema note.

---

### `artifacts/`
Generated outputs from runs.

Contains:
- `checkpoints/`
- `logs/`
- `predictions/`
- `metrics/`

Should be gitignored.

---

### `tests/`
Unit and integration tests for core logic.

Targets:
- config loading
- dataset formatting
- tokenization
- model setup
- scoring functions

---

### `docs/`
Extended documentation.

Contains:
- dataset schema
- training recipes
- troubleshooting notes
- design/architecture details

## Conventions

- Training and scoring logic belongs in `src/`.
- Notebooks should only import and call functions from `src/`.
- Hyperparameters and paths should live in `configs/`.
- Large data and outputs should not be committed to Git.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .