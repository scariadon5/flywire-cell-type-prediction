# On Windows run with `gmake` (ships with Strawberry Perl) from Git Bash; on Linux/Kaggle plain `make`.
ifeq ($(OS),Windows_NT)
	PY ?= .venv/Scripts/python
else
	PY ?= .venv/bin/python
endif

.PHONY: setup data download convert build check explore clean-data splits baselines graph-models gnn report test

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# One command: raw files -> verified -> Parquet -> clean graph + feature/label tables -> sanity checks.
data: download convert build check

download:
	$(PY) -m flyct.download

convert:
	$(PY) -m flyct.convert

build:
	$(PY) -m flyct.build

check:
	$(PY) -m flyct.checks

explore:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb

# Keeps the raw downloads; removes everything derived from them.
clean-data:
	rm -rf data/interim data/processed

# Week 2: saved seed/val/test splits, then the feature baselines (super class and class run on a laptop).
splits:
	$(PY) -m flyct.splits

baselines:
	$(PY) -m flyct.run --method majority lightgbm --task super_class cell_class --split all

# Connectivity models. NTAC runs every task on a laptop; partner-profile LR does super class and class here
# (cell type needs the Kaggle notebook).
graph-models:
	$(PY) -m flyct.run --method ntac --task cell_type --split all
	$(PY) -m flyct.run --method profile_lr profile_lr_iter --task super_class cell_class --split all

# Week 3: GraphSAGE and its feature ablations. 3-5 s per epoch on a 4-core laptop CPU (~10-15 min a run), so the full grid
# goes to the Kaggle notebook (GPU); this target is the one-split check that the model works locally.
gnn:
	$(PY) -m flyct.run --method sage_conn sage_region sage --task super_class cell_class --split frac20_r0

# results/runs.jsonl -> results/results.md
report:
	$(PY) -m flyct.report

test:
	$(PY) -m pytest -q tests
