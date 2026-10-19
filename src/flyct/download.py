"""Download the raw files and verify them against pinned md5 checksums.

Zenodo throttles each connection (about 30 KB/s from some networks), so large files are fetched
as parallel byte ranges. Finished ranges are recorded in a sidecar file, so an interrupted
download resumes where it stopped. Files that already exist with the right checksum are skipped.
Writes data/checksums.md5 (committed to git) with the md5 of every raw file.
"""

import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from flyct import config

CHUNK = 4 << 20  # bytes per range request; small so a dropped connection costs little to redo
LEGACY_CHUNK = 16 << 20  # chunk size of sidecar files written as a plain list of starts
WORKERS = 16
HEADERS = {"User-Agent": "flyct-downloader"}


def md5sum(path, block=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(block):
            h.update(data)
    return h.hexdigest()


def content_length(url: str) -> int:
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Range"].split("/")[1])


def fetch_range(url, path, start, end, bar, retries=30):
    pos = start  # Zenodo often drops connections mid-range; each retry continues from here
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "r+b") as f:
                if resp.status != 206:
                    raise OSError(f"server ignored range request (HTTP {resp.status})")
                f.seek(pos)
                while data := resp.read(1 << 16):
                    f.write(data)
                    pos += len(data)
                    bar.update(len(data))
            if pos == end + 1:
                return start, end
            error = f"connection closed at byte {pos}"
        except OSError as e:
            error = repr(e)
        tqdm.write(f"retry {attempt + 1}/{retries} bytes {start}-{end}: {error}")
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def load_done(done_file, part, size) -> list[list[int]]:
    """Finished byte intervals [start, end]. Reads the legacy format (a list of 16 MB chunk starts) too."""
    if not (done_file.exists() and part.exists()):
        return []
    saved = json.loads(done_file.read_text())
    if saved and isinstance(saved[0], int):
        return [[s, min(s + LEGACY_CHUNK, size) - 1] for s in saved]
    return saved


def fetch(url: str, dest) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    done_file = dest.with_suffix(dest.suffix + ".ranges.json")
    size = content_length(url)
    done = load_done(done_file, part, size)
    if not part.exists():
        with open(part, "wb") as f:
            f.truncate(size)

    covered = lambda s, e: any(a <= s and e <= b for a, b in done)  # noqa: E731
    todo = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    todo = [(s, e) for s, e in todo if not covered(s, e)]
    already = size - sum(e - s + 1 for s, e in todo)
    with tqdm(total=size, initial=already, unit="B", unit_scale=True, desc=dest.name) as bar:
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(fetch_range, url, part, s, e, bar) for s, e in todo]
            for fut in as_completed(futures):
                done.append(list(fut.result()))
                done_file.write_text(json.dumps(sorted(done)))

    part.replace(dest)
    done_file.unlink(missing_ok=True)


def main() -> int:
    config.RAW.mkdir(parents=True, exist_ok=True)
    sums = {}

    for name, expected in config.ZENODO_FILES.items():
        dest = config.RAW / name
        if dest.exists() and md5sum(dest) == expected:
            print(f"ok (cached)  {name}")
        else:
            fetch(config.zenodo_url(name), dest)
            got = md5sum(dest)
            if got != expected:
                print(f"CHECKSUM MISMATCH {name}: expected {expected}, got {got}", file=sys.stderr)
                return 1
            print(f"ok           {name}")
        sums[name] = expected

    # The annotation file has no published md5; it is pinned by commit, and we record what we got.
    ann = config.RAW / config.ANNOTATIONS_FILE
    if not ann.exists():
        fetch(config.ANNOTATIONS_URL, ann)
    sums[config.ANNOTATIONS_FILE] = md5sum(ann)
    print(f"ok           {config.ANNOTATIONS_FILE} @ {config.ANNOTATIONS_TAG}")

    lines = [f"{md5}  raw/{name}" for name, md5 in sums.items()]
    (config.DATA / "checksums.md5").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
