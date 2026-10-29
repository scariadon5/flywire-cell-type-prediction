"""Generates flyct_celltype.ipynb, the Kaggle notebook for runs that don't fit a laptop: every cell-type
run, and the GraphSAGE grid (set METHODS and TASKS in the first cell).

On Kaggle: New Notebook -> File -> Import Notebook -> this .ipynb. Settings: Internet on (needs a
phone-verified account), accelerator GPU T4 x2 or none. If the GitHub repo is private, add a Kaggle
secret GITHUB_TOKEN holding a fine-grained token with read access to it.

The notebook rebuilds the data with `make data` (checked against the committed build report), rebuilds
the splits with `make splits` (checked against the committed split sizes), runs the models, and saves
the new result lines to /kaggle/working/runs_kaggle.jsonl. Download that file and append it locally:

    cat runs_kaggle.jsonl >> results/runs.jsonl && python -m flyct.report
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# flyct · runs on Kaggle

Rebuilds the FlyWire graph and the saved splits from scratch, then runs the models that need more than a
laptop: cell-type models, and GraphSAGE on a GPU. Results go to `/kaggle/working/runs_kaggle.jsonl`."""),
    code("""REPO = "scariadon5/flywire-cell-type-prediction"
BRANCH = "main"
METHODS = "sage_conn sage_region sage"            # cell-type partner profile: "profile_linear profile_linear_iter"
TASKS = "super_class cell_class cell_type"        # ... with TASKS = "cell_type"
SPLITS = "all"                                    # or e.g. "frac20 hemi"

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    token = None  # public repo
url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
%cd /kaggle/working
!rm -rf /kaggle/working/flyct && git clone -q --depth 1 -b {BRANCH} {url} /kaggle/working/flyct
%cd /kaggle/working/flyct
!git log -1 --format='commit %h  %s'"""),
    code("""# Kaggle images already ship pandas, numpy, scipy, scikit-learn, torch and matplotlib.
!pip install -q "duckdb>=1.1" "pyarrow>=17" "numba>=0.60" "lightgbm>=4.5" tqdm
!pip install -q --no-deps -e ."""),
    md("## Data and splits (about 10 minutes; both steps fail loudly if anything differs from the repo)"),
    code("""!make data PY=python
!make splits PY=python"""),
    md("## Runs"),
    code("""import torch, os
print("device:", "cuda" if torch.cuda.is_available() else "cpu", "| threads:", torch.get_num_threads())
before = sum(1 for _ in open("results/runs.jsonl"))
!python -m flyct.run --method {METHODS} --task {TASKS} --split {SPLITS}"""),
    code("""lines = open("results/runs.jsonl").read().splitlines()[before:]
open("/kaggle/working/runs_kaggle.jsonl", "w").write("\\n".join(lines) + "\\n")
print(f"{len(lines)} new runs saved to /kaggle/working/runs_kaggle.jsonl")
!python -m flyct.report"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
})
nbf.write(nb, Path(__file__).with_name("flyct_celltype.ipynb"))
print("wrote flyct_celltype.ipynb")
