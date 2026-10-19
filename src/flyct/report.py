"""Build the results tables from results/runs.jsonl: mean (± sd over reps) per method and split setting.

    python -m flyct.report        # writes results/results.md and prints it

If a (method, task, split) was run more than once, only the latest run counts.
"""

import sys

import pandas as pd

from flyct import config
from flyct.run import RESULTS

SETTINGS = [("sweep", 0.01, "1%"), ("sweep", 0.05, "5%"), ("sweep", 0.20, "20%"), ("sweep", 0.80, "80%"),
            ("hemi_l2r", None, "L→R"), ("hemi_r2l", None, "R→L")]
METHODS = ["majority", "lightgbm", "profile_lr", "profile_lr_iter", "ntac"]
TASK_NAMES = {"super_class": "Super class (10)", "cell_class": "Class (49)", "cell_type": "Cell type (8,840)"}


def latest(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.sort_values("time").drop_duplicates(["method", "task", "split"], keep="last")


def cell(group: pd.Series) -> str:
    if group.empty or group.isna().all():
        return ""
    mean = 100 * group.mean()
    return f"{mean:.1f} ± {100 * group.std():.1f}" if len(group) > 1 else f"{mean:.1f}"


def table(runs: pd.DataFrame, task: str, metric: str) -> str:
    rows = []
    for method in [m for m in METHODS if m in set(runs.method)]:
        r = runs[(runs.task == task) & (runs.method == method)]
        if r.empty:
            continue
        cells = []
        for setting, frac, _ in SETTINGS:
            sel = r[r.setting == setting]
            if frac is not None:
                sel = sel[(sel.fraction - frac).abs() < 1e-9]
            cells.append(cell(sel[metric]) if metric in sel else "")
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    head = "| Method | " + " | ".join(name for *_, name in SETTINGS) + " |"
    return "\n".join([head, "|" + " --- |" * (len(SETTINGS) + 1), *rows])


def region_table(runs: pd.DataFrame, task: str, split_setting=("sweep", 0.20)) -> str:
    setting, frac = split_setting
    r = runs[(runs.task == task) & (runs.setting == setting) & ((runs.fraction - frac).abs() < 1e-9)]
    regions = ["optic", "central", "visual_projection", "sensory"]
    sizes = ["2", "3-10", "11-100", "gt100"]
    rows = []
    for method in [m for m in METHODS if m in set(r.method)]:
        g = r[r.method == method]
        cells = [cell(g.get(f"test_acc_region_{x}", pd.Series(dtype=float))) for x in regions]
        cells += [cell(g.get(f"test_acc_size_{x}", pd.Series(dtype=float))) for x in sizes]
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    head = ("| Method | Optic | Central | Visual proj. | Sensory | Type size 2 | 3–10 | 11–100 | >100 |")
    return "\n".join([head, "|" + " --- |" * 9, *rows]) if rows else ""


def main() -> int:
    runs = latest(pd.read_json(RESULTS, lines=True))
    out = [f"# Results\n\nTest accuracy (%), mean ± sd over 5 random splits; hemisphere transfer is one split. "
           f"Annotations {config.ANNOTATIONS_TAG}, edges with ≥ {config.SYNAPSE_THRESHOLD} synapses. "
           "Built by `python -m flyct.report` from `results/runs.jsonl`.\n"]
    for task, name in TASK_NAMES.items():
        t = table(runs, task, "test_acc")
        if t:
            out.append(f"## {name}\n\n**Accuracy**\n\n{t}\n\n**Macro F1** (every class counts equally)\n\n"
                       f"{table(runs, task, 'test_f1_macro')}\n")
            if task == "cell_type":
                out.append(f"**Top-5 accuracy**\n\n{table(runs, task, 'test_top5_acc')}\n")
    rt = region_table(runs, "cell_type")
    if rt:
        out.append(f"## Cell type at 20% seeds, by region and type size (accuracy %)\n\n{rt}\n")
    text = "\n".join(out)
    (config.ROOT / "results" / "results.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
