"""Build the clean graph and the per-neuron tables with DuckDB.

Outputs (data/processed/):
  edges.parquet     one row per directed neuron pair: synapses summed across neuropils, kept if >= threshold,
                    plus synapse-weighted mean neurotransmitter probabilities
  features.parquet  one row per proofread neuron, connectivity-derived features only
  labels.parquet    one row per proofread neuron: targets (super_class, cell_class, cell_type) and side

Features and labels are written to separate files on purpose: no model code ever needs to open
labels.parquet to get features, so annotation columns cannot leak into a feature matrix by accident.
"""

import sys

import duckdb
import numpy as np
import pandas as pd

from flyct import config

NTS = ["gaba", "ach", "glut", "oct", "ser", "da"]
EDGE_COLUMNS = {"pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *(f"{nt}_avg" for nt in NTS)}


def connect() -> duckdb.DuckDBPyConnection:
    tmp = config.INTERIM / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    # Cap DuckDB well under the laptop's free RAM; it spills to disk beyond this.
    return duckdb.connect(config={"memory_limit": "2GB", "threads": 4, "temp_directory": str(tmp)})


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    con = connect()
    i = config.INTERIM
    conn_pq = f"'{i / 'proofread_connections_783.parquet'}'"
    pre_pq = f"'{i / 'per_neuron_neuropil_count_pre_783.parquet'}'"
    post_pq = f"'{i / 'per_neuron_neuropil_count_post_783.parquet'}'"
    out = lambda name: f"'{config.PROCESSED / name}'"  # noqa: E731

    cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {conn_pq}").fetchall()}
    missing = EDGE_COLUMNS - cols
    if missing:
        print(f"edge file is missing expected columns: {sorted(missing)}", file=sys.stderr)
        return 1

    ids = np.load(config.RAW / "proofread_root_ids_783.npy").astype("int64")
    con.register("ids_df", pd.DataFrame({"root_id": ids}))
    con.sql("CREATE TABLE ids AS SELECT root_id FROM ids_df ORDER BY root_id")

    # 1. Edges: sum across neuropils first, then threshold. Thresholding per neuropil would drop
    #    pairs whose synapses are spread over several regions.
    nt_weighted = ", ".join(f"CAST(sum({nt}_avg * syn_count) / sum(syn_count) AS FLOAT) AS {nt}" for nt in NTS)
    print("aggregating edges ...")
    con.sql(f"""
        CREATE TABLE pairs AS
        SELECT pre_pt_root_id AS pre_root_id, post_pt_root_id AS post_root_id,
               CAST(sum(syn_count) AS INTEGER) AS syn_count, {nt_weighted}
        FROM {conn_pq}
        WHERE pre_pt_root_id IN (SELECT root_id FROM ids) AND post_pt_root_id IN (SELECT root_id FROM ids)
        GROUP BY 1, 2
    """)
    con.sql(f"""
        COPY (SELECT * FROM pairs WHERE syn_count >= {config.SYNAPSE_THRESHOLD} ORDER BY pre_root_id, post_root_id)
        TO {out('edges.parquet')} (FORMAT parquet, COMPRESSION zstd)
    """)
    con.sql(f"CREATE VIEW edges AS SELECT * FROM {out('edges.parquet')}")

    # 2. Degree and synapse totals on the thresholded graph.
    degree = con.sql("""
        SELECT id.root_id,
               coalesce(o.n, 0) AS out_degree, coalesce(o.syn, 0) AS out_syn,
               coalesce(n.n, 0) AS in_degree,  coalesce(n.syn, 0) AS in_syn
        FROM ids id
        LEFT JOIN (SELECT pre_root_id AS root_id, count(*) n, sum(syn_count) syn FROM edges GROUP BY 1) o USING (root_id)
        LEFT JOIN (SELECT post_root_id AS root_id, count(*) n, sum(syn_count) syn FROM edges GROUP BY 1) n USING (root_id)
    """)
    con.sql("CREATE TABLE degree AS SELECT * FROM degree")

    # 3. Neurotransmitter mix of each neuron's outgoing and incoming synapses, over all proofread
    #    partners (before thresholding), weighted by synapse count.
    def nt_mix(side_col: str, prefix: str) -> str:
        sel = ", ".join(f"CAST(sum({nt} * syn_count) / sum(syn_count) AS FLOAT) AS {prefix}_{nt}" for nt in NTS)
        return f"SELECT {side_col} AS root_id, {sel} FROM pairs GROUP BY 1"

    con.sql(f"CREATE TABLE nt_out AS {nt_mix('pre_root_id', 'out_nt')}")
    con.sql(f"CREATE TABLE nt_in AS {nt_mix('post_root_id', 'in_nt')}")

    # 4. Synapses per neuropil (all synapses, including onto unproofread fragments), pivoted wide.
    neuropils = [r[0] for r in con.sql(
        f"SELECT DISTINCT neuropil FROM {pre_pq} UNION SELECT DISTINCT neuropil FROM {post_pq} ORDER BY 1"
    ).fetchall()]

    def pivot(src: str, id_col: str, prefix: str) -> str:
        cols = ", ".join(
            f"coalesce(sum(count) FILTER (WHERE neuropil = '{n}'), 0)::INTEGER AS {prefix}_{n.lower()}" for n in neuropils
        )
        return f"""SELECT {id_col} AS root_id, sum(count)::INTEGER AS {prefix}_total, {cols}
                   FROM {src} WHERE {id_col} IN (SELECT root_id FROM ids) GROUP BY 1"""

    print("pivoting neuropil counts ...")
    con.sql(f"CREATE TABLE np_pre AS {pivot(pre_pq, 'pre_pt_root_id', 'pre')}")
    con.sql(f"CREATE TABLE np_post AS {pivot(post_pq, 'post_pt_root_id', 'post')}")

    pre_cols = [r[0] for r in con.sql("DESCRIBE np_pre").fetchall() if r[0] != "root_id"]
    post_cols = [r[0] for r in con.sql("DESCRIBE np_post").fetchall() if r[0] != "root_id"]
    nt_out_cols = [f"out_nt_{nt}" for nt in NTS]
    nt_in_cols = [f"in_nt_{nt}" for nt in NTS]
    zero_fill = ", ".join(f"coalesce({c}, 0) AS {c}" for c in pre_cols + post_cols)
    # A neuron with no partners in a direction has no NT mix there; leave it NULL rather than invent zeros.
    nt_sel = ", ".join(nt_out_cols + nt_in_cols)

    con.sql(f"""
        COPY (
            SELECT d.*, {nt_sel}, {zero_fill}
            FROM degree d
            LEFT JOIN nt_out USING (root_id)
            LEFT JOIN nt_in USING (root_id)
            LEFT JOIN np_pre USING (root_id)
            LEFT JOIN np_post USING (root_id)
            ORDER BY root_id
        ) TO {out('features.parquet')} (FORMAT parquet, COMPRESSION zstd)
    """)

    # 5. Labels: targets plus side (for the hemisphere split), nothing else from the annotation file.
    ann = config.RAW / config.ANNOTATIONS_FILE
    con.sql(f"""
        COPY (
            SELECT id.root_id,
                   nullif(a.super_class, '') AS super_class,
                   nullif(a.cell_class, '')  AS cell_class,
                   nullif(a.cell_type, '')   AS cell_type,
                   nullif(a.side, '')        AS side
            FROM ids id
            LEFT JOIN read_csv('{ann}', delim='\t', header=true, all_varchar=true) a
                   ON CAST(a.root_id AS BIGINT) = id.root_id
            ORDER BY id.root_id
        ) TO {out('labels.parquet')} (FORMAT parquet, COMPRESSION zstd)
    """)

    for name in ("edges", "features", "labels"):
        n = con.sql(f"SELECT count(*) FROM {out(name + '.parquet')}").fetchone()[0]
        print(f"{name}.parquet: {n:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
