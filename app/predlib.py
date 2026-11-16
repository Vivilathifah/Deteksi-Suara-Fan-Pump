"""Pemuat modul prediksi (numba) di latar belakang.

numba butuh 10-30 detik untuk kompilasi pertama. Dengan start() kompilasi dimulai di thread terpisah
saat dashboard pertama dibuka, sehingga halaman lain tidak ikut menunggu."""
import threading

_state = {"mod": None, "err": None, "thread": None}
_lock = threading.Lock()


def _load():
    with _lock:
        if _state["mod"] is None and _state["err"] is None:
            try:
                import _predict_lib as P
                P.warmup()
                _state["mod"] = P
            except Exception as e:  # noqa: BLE001
                _state["err"] = e


def start():
    """Mulai kompilasi di latar belakang (aman dipanggil berkali-kali)."""
    if _state["thread"] is None:
        _state["thread"] = threading.Thread(target=_load, daemon=True)
        _state["thread"].start()


def ready():
    return _state["mod"] is not None


def error():
    return _state["err"]


def get():
    """Tunggu sampai siap lalu kembalikan modul _predict_lib."""
    _load()
    if _state["err"] is not None:
        raise _state["err"]
    return _state["mod"]
