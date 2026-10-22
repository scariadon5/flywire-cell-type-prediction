"""Generates flyct_celltype.ipynb, the Kaggle notebook for every cell-type run that doesn't fit a laptop.

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
    md("""# flyct · cell-type runs on Kaggle

Rebuilds the FlyWire graph and the saved splits from scratch, then runs the cell-type models that need
more than a laptop's memory. Results go to `/kaggle/working/runs_kaggle.jsonl`."""),
    code("""REPO = "scariadon5/flywire-cell-type-prediction"
BRANCH = "main"
METHODS = "profile_linear profile_linear_iter"
SPLITS = "all"   # or e.g. "frac20 hemi"

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
!python -m flyct.run --method {METHODS} --task cell_type --split {SPLITS}"""),
    code("""lines = open("results/runs.jsonl").read().splitlines()[before:]
open("/kaggle/working/runs_kaggle.jsonl", "w").write("\\n".join(lines) + "\\n")
print(f"{len(lines)} new runs saved to /kaggle/working/runs_kaggle.jsonl")
!python -m flyct.report | sed -n '/Cell type/,$p'"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
})
nbf.write(nb, Path(__file__).with_name("flyct_celltype.ipynb"))
print("wrote flyct_celltype.ipynb")
