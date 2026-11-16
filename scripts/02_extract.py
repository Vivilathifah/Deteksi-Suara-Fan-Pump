"""Tahap 1b - Ekstraksi zip ke data/extracted/{dev,eval_train}/{fan,pump}."""
import zipfile
from common import RAW, EXTRACTED, ZIPS, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    for name in ZIPS:
        subset = "dev" if name.startswith("dev") else "eval_train"
        machine = "fan" if "fan" in name else "pump"
        zp = RAW / name
        if not zp.exists():
            print(f"[!] {zp} tidak ada, lewati")
            continue
        out = EXTRACTED / subset / machine
        with zipfile.ZipFile(zp) as z:
            if zp.stat().st_size and any(out.rglob("*.wav")):
                print(f"[skip] {name} sudah diekstrak")
                continue
            z.extractall(out)
            print(f"[ok] {name} -> {out}")
