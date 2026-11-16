"""Tahap 1a - Download 4 file zip MIMII DUE dari Zenodo (resume-safe: file lengkap dilewati)."""
import requests
from tqdm import tqdm
from common import RAW, ZIPS, ensure_dirs


def download(url, dest):
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("content-length", 0))
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"[skip] {dest.name} sudah lengkap")
        return
    with requests.get(url, stream=True, timeout=60) as r, open(dest, "wb") as f, \
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


if __name__ == "__main__":
    ensure_dirs()
    for name, url in ZIPS.items():
        download(url, RAW / name)
