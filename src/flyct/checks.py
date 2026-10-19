"""Sanity checks on the built tables, against published numbers and against the committed build report.

Fails loudly on anything that would make downstream numbers meaningless: wrong neuron count,
duplicate IDs, edges to unknown neurons, labels leaking into the feature table.
"""

import json
import sys

import duckdb

from flyct import config

# Annotation columns that must never appear in the feature table.
FORBIDDEN_FEATURES = {
    "super_class", "cell_class", "cell_sub_class", "supertype", "cell_type", "hemibrain_type",
    "ito_lee_hemilineage", "hartenstein_hemilineage", "top_nt", "top_nt_conf", "known_nt",
    "known_nt_source", "side", "nerve", "vfb_id", "fbbt_id", "status", "dimorphism",
    "matching_notes", "fru_dsx", "synonyms", "flow", "pos_x", "pos_y", "pos_z",
    "soma_x", "soma_y", "soma_z", "nucleus_id", "supervoxel_id",
}


def main() -> int:
    p = config.PROCESSED
    con = duckdb.connect()
    q = lambda sql: con.sql(sql).fetchone()[0]  # noqa: E731
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    feats, labels, edges = (f"'{p / n}'" for n in ("features.parquet", "labels.parquet", "edges.parquet"))

    n_feat = q(f"SELECT count(*) FROM {feats}")
    check("neuron count matches published 139,255", n_feat == config.EXPECTED_NEURONS, f"{n_feat:,}")
    check("feature root_ids unique", q(f"SELECT count(DISTINCT root_id) FROM {feats}") == n_feat)
    n_lab = q(f"SELECT count(*) FROM {labels}")
    check("one label row per neuron", n_lab == n_feat, f"{n_lab:,}")

    n_edges = q(f"SELECT count(*) FROM {edges}")
    rel = abs(n_edges - config.EXPECTED_EDGES_APPROX) / config.EXPECTED_EDGES_APPROX
    check("thresholded edges within 5% of ~2.7M", rel < 0.05, f"{n_edges:,}")
    check("every edge has >= threshold synapses",
          q(f"SELECT min(syn_count) FROM {edges}") >= config.SYNAPSE_THRESHOLD)
    check("edge pairs unique",
          q(f"SELECT count(*) FROM (SELECT DISTINCT pre_root_id, post_root_id FROM {edges})") == n_edges)
    orphans = q(f"""SELECT count(*) FROM {edges} e
                    WHERE e.pre_root_id NOT IN (SELECT root_id FROM {feats})
                       OR e.post_root_id NOT IN (SELECT root_id FROM {feats})""")
    check("every edge endpoint is a proofread neuron", orphans == 0, f"{orphans:,} orphan edges")

    cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {feats}").fetchall()}
    leaked = cols & FORBIDDEN_FEATURES
    check("no annotation columns in feature table", not leaked, ", ".join(sorted(leaked)))

    annotated = q(f"SELECT count(*) FROM {labels} WHERE super_class IS NOT NULL")
    check("almost every neuron has a super class", annotated / n_lab > 0.99, f"{annotated:,} of {n_lab:,}")

    # Reproducibility: a fresh clone must rebuild exactly the counts in the committed report.
    counts = {
        "neurons": n_feat,
        "edges": n_edges,
        "synapses_in_edges": int(q(f"SELECT sum(syn_count) FROM {edges}")),
        "feature_columns": len(cols),
        "with_super_class": annotated,
        "with_cell_class": q(f"SELECT count(*) FROM {labels} WHERE cell_class IS NOT NULL"),
        "with_cell_type": q(f"SELECT count(*) FROM {labels} WHERE cell_type IS NOT NULL"),
        "distinct_cell_types": q(f"SELECT count(DISTINCT cell_type) FROM {labels}"),
    }
    report_path = config.DATA / "build_report.json"
    if report_path.exists():
        expected = json.loads(report_path.read_text())["counts"]
        diff = {k: (expected.get(k), v) for k, v in counts.items() if expected.get(k) != v}
        check("row counts match committed build report", not diff, f"expected vs got: {diff}" if diff else "")
    else:
        report = {
            "counts": counts,
            "annotations_tag": config.ANNOTATIONS_TAG,
            "zenodo_record": config.ZENODO_RECORD,
            "synapse_threshold": config.SYNAPSE_THRESHOLD,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {report_path.name} (commit it; future builds are checked against it)")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
