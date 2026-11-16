"""Pustaka prediksi k-NN dengan jarak DTW / FastDTW / Euclid.
Dipakai oleh scripts/13_predict.py (evaluasi) dan dashboard (halaman Prediksi)."""
import time
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import cdist
from common import ROOT, RESULTS, SR, N_FFT, HOP, N_MELS, N_MFCC
from _dtw_lib import dtw_pair, fastdtw_pair

PRED = ROOT / "prediction"            # referensi (template) + contoh uji untuk dashboard
PRED_RES = RESULTS / "prediction"     # metrik & prediksi
METHODS = ["DTW", "FastDTW", "Euclid"]
EUCLID_T = 60                         # baseline Euclid: deret diresampling ke 60 frame (seperti materi kuliah)


# ---------- jarak ----------
@njit(parallel=True, cache=True)
def dtw_cross(Q, R):
    """DTW exact antara tiap baris Q (query) dan tiap baris R (referensi) -> matriks (nq, nr)."""
    nq, nr = Q.shape[0], R.shape[0]
    M = np.zeros((nq, nr))
    for i in prange(nq):
        for j in range(nr):
            M[i, j] = dtw_pair(Q[i], R[j])
    return M


def _frow(q, R, radius):
    return [fastdtw_pair(q, r, radius) for r in R]


def fastdtw_cross(Q, R, radius=1, n_jobs=1):
    if n_jobs == 1:
        return np.array([_frow(q, R, radius) for q in Q])
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=n_jobs, max_nbytes="1M", mmap_mode="r")(delayed(_frow)(q, R, radius) for q in Q)
    return np.array(rows)


def resample_T(x, T=EUCLID_T):
    t0, t1 = np.linspace(0, 1, len(x)), np.linspace(0, 1, T)
    return np.stack([np.interp(t1, t0, x[:, k]) for k in range(x.shape[1])], axis=1)


def euclid_cross(Q, R, T=EUCLID_T):
    q = np.stack([resample_T(x, T).ravel() for x in Q])
    r = np.stack([resample_T(x, T).ravel() for x in R])
    return cdist(q, r)


@njit(cache=True)
def _dtw_acc(a, b):
    n, m, d = a.shape[0], b.shape[0], a.shape[1]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0.0
            for k in range(d):
                t = a[i - 1, k] - b[j - 1, k]
                c += t * t
            D[i, j] = np.sqrt(c) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def dtw_path(a, b):
    """Jalur warping optimal DTW (backtracking dari (n, m) ke (1, 1)). Return (path[(i, j)...], jarak ternormalisasi)."""
    D = _dtw_acc(a, b)
    i, j = len(a), len(b)
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1], D[len(a), len(b)] / (len(a) + len(b))


def warmup():
    """Kompilasi numba lebih dulu supaya waktu hitung pertama tidak menyesatkan."""
    z = np.zeros((1, 4, 2))
    dtw_cross(z, z)
    _dtw_acc(z[0], z[0])


# ---------- k-NN ----------
def knn_predict(Dqr, ref_labels, k=1):
    """Dqr: (nq, nr). Voting mayoritas k tetangga terdekat; seri -> label tetangga yang paling dekat."""
    ref_labels = np.asarray(ref_labels)
    idx = np.argsort(Dqr, axis=1)[:, :k]
    out = []
    for row in ref_labels[idx]:
        vals, cnt = np.unique(row, return_counts=True)
        best = set(vals[cnt == cnt.max()])
        out.append(next(l for l in row if l in best))
    return np.array(out)


# ---------- fitur ----------
def fit_scaler(X):
    """z-score per koefisien, statistik dari data referensi saja. X: (N, T, D)."""
    mu = X.mean(axis=(0, 1), keepdims=True)[0]
    sd = X.std(axis=(0, 1), keepdims=True)[0] + 1e-8
    return mu, sd


def fit_length(A, T):
    """Samakan panjang deret (T frame): potong bagian tengah bila lebih panjang, ulang tepi bila lebih pendek."""
    if len(A) >= T:
        s = (len(A) - T) // 2
        return A[s:s + T]
    pad = T - len(A)
    return np.pad(A, ((0, pad), (0, 0)), mode="edge")


def load_audio(fileobj):
    import librosa
    y, _ = librosa.load(fileobj, sr=SR, mono=True)
    return y


def preprocess(y):
    """Sama dengan 05_preprocessing.py: hapus DC offset, normalisasi puncak 0,99."""
    y = y - y.mean()
    peak = np.abs(y).max()
    return y / peak * 0.99 if peak > 0 else y


def extract(y, feature):
    """Sama dengan _extract_common.py. Return (D, T)."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    if feature == "LogMel":
        return librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC).astype(np.float32)


# ---------- paket referensi ----------
def save_refs(path, Xr, yr, idr, Xq, yq, idq, k, radius, feature):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Xr=Xr.astype(np.float32), yr=np.asarray(yr, dtype=str), idr=np.asarray(idr, dtype=str),
                        Xq=Xq.astype(np.float32), yq=np.asarray(yq, dtype=str), idq=np.asarray(idq, dtype=str),
                        k=k, radius=radius, feature=feature)


def load_refs(path):
    z = np.load(path, allow_pickle=False)
    Xr, Xq = z["Xr"].astype(np.float64), z["Xq"].astype(np.float64)
    mu, sd = fit_scaler(Xr)
    return dict(Xr=(Xr - mu) / sd, yr=z["yr"], idr=z["idr"], Xq=(Xq - mu) / sd, yq=z["yq"], idq=z["idq"],
                mu=mu, sd=sd, k=int(z["k"]), radius=int(z["radius"]), feature=str(z["feature"]), T=Xr.shape[1])


def predict_query(xq, refs, k=1, radius=1):
    """xq: (T, D) sudah distandardisasi dengan scaler referensi. Return dict per metode."""
    Q = np.ascontiguousarray(xq[None], dtype=np.float64)
    R = refs["Xr"]
    fns = {"DTW": lambda: dtw_cross(Q, R), "FastDTW": lambda: fastdtw_cross(Q, R, radius), "Euclid": lambda: euclid_cross(Q, R)}
    out = {}
    for name, fn in fns.items():
        t0 = time.perf_counter()
        d = fn()[0]
        dt = time.perf_counter() - t0
        out[name] = dict(pred=str(knn_predict(d[None], refs["yr"], k)[0]), dist=d, nn=int(np.argmin(d)), seconds=dt)
    return out
