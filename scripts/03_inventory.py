"""Tahap 1c - Inventarisasi seluruh .wav -> data/metadata/audio_manifest.csv."""
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from common import EXTRACTED, MANIFEST, ROOT, parse_filename, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    rows = []
    files = sorted(EXTRACTED.rglob("*.wav"))
    for p in tqdm(files, desc="inventory"):
        rel = p.relative_to(EXTRACTED).parts  # subset/machine/...
        subset, machine = rel[0], rel[1]
        try:
            info = sf.info(p)
            sr, ch, dur, fr = info.samplerate, info.channels, info.duration, info.frames
        except Exception:
            sr = ch = dur = fr = None
        rows.append(dict(file_id=f"{subset}__{machine}__{p.stem}", path=str(p.relative_to(ROOT)), subset=subset, machine=machine,
                         **parse_filename(p.name), sr=sr, channels=ch, duration=dur, frames=fr))
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    print(f"{len(df)} file -> {MANIFEST}")
    if len(df):
        print(df.groupby(["subset", "machine", "section", "domain", "split", "condition"]).size().to_string())
