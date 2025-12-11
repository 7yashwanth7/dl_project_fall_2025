
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
- `evaluate.ipynb`: 

Notebooks should focus on:
- wiring inputs/outputs
- quick debugging
- visualization


### Steps to configure Git with a Personal Access Token (PAT):

1.  **Generate a GitHub Personal Access Token (PAT):**
    *   Go to your GitHub settings: `Settings > Developer settings > Personal access tokens > Tokens (classic)`. You might need to create a new token. Ensure it has the `repo` scope selected.
    *   **Copy the generated token immediately! You will not be able to see it again.**

2.  **Store your PAT securely in Colab Secrets:**
    *   In Colab, click the "🔑 Secrets" icon in the left-hand panel.
    *   Add a new secret, name it `GH_TOKEN`, and paste your GitHub PAT as the value.
    *   Make sure "Notebook access" is toggled on for this notebook.

3.  **Configure Git in Colab using the stored PAT:**
    *   Run the following Python cell. It will retrieve your PAT from Colab secrets and configure Git to use it for authentication with GitHub.