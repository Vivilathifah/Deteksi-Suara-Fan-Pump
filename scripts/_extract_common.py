import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
from common import FEATURES, PROC_MANIFEST, PROCESSED, SR, N_FFT, HOP, N_MELS, N_MFCC, ensure_dirs


def logmel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def mfcc(y):
    return librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)), n_mfcc=N_MFCC).astype(np.float32)


def run(name, fn, n_jobs=-1):
    ensure_dirs()
    pm = pd.read_csv(PROC_MANIFEST)

    def one(r):
        out = FEATURES / name / r.machine / f"{r.file_id}.npy"
        if out.exists():
            return
        y, _ = sf.read(PROCESSED / r.machine / f"{r.file_id}.wav")
        np.save(out, fn(y))

    Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in tqdm(list(pm.itertuples()), desc=name))
    print(f"{name}: selesai -> {FEATURES / name}")
