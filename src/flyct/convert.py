"""Convert the raw Feather files to Parquet one record batch at a time.

The 852 MB edge file decompresses to well over 1 GB; loading it whole does not fit
in a 6 GB laptop alongside Windows. Streaming batches keeps peak memory near one batch.
"""

import sys

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from flyct import config


def feather_to_parquet(src, dest) -> int:
    rows = 0
    with pa.memory_map(str(src)) as source:
        reader = ipc.open_file(source)
        tmp = dest.with_suffix(".parquet.part")
        with pq.ParquetWriter(tmp, reader.schema, compression="zstd") as writer:
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                writer.write_batch(batch)
                rows += batch.num_rows
        print(f"{src.name}: {reader.num_record_batches} batches, {rows:,} rows")
    tmp.replace(dest)
    return rows


def main() -> int:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    for name in config.ZENODO_FILES:
        if not name.endswith(".feather"):
            continue
        dest = config.INTERIM / name.replace(".feather", ".parquet")
        if dest.exists():
            print(f"skip (exists) {dest.name}")
            continue
        feather_to_parquet(config.RAW / name, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
