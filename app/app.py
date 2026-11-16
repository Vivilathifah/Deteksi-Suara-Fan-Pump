"""Dashboard MIMII DUE · DTW vs FastDTW (Streamlit).

Jalankan dari root MIMII_Research:   streamlit run app/app.py
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import predlib  # noqa: E402
import ui  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="MIMII DUE · DTW vs FastDTW", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))


@st.cache_resource
def _prewarm():
    # kompilasi numba untuk halaman Prediksi dimulai sekali per server, di latar belakang
    if os.environ.get("MIMII_PREWARM", "1") != "0" and D.pred_available():
        predlib.start()
    return True


_prewarm()

pages = {
    "Penelitian": [
        st.Page("views/home.py", title="Ringkasan", icon="🏠", default=True),
        st.Page("views/guide.py", title="Panduan & Istilah", icon="📘"),
        st.Page("views/dataset.py", title="Dataset", icon="🗂️"),
        st.Page("views/audio.py", title="Audio Processing", icon="🔊"),
        st.Page("views/features.py", title="Ekstraksi Fitur", icon="🎛️"),
    ],
    "Analisis": [
        st.Page("views/distance.py", title="DTW vs FastDTW", icon="📏"),
        st.Page("views/cluster.py", title="Fuzzy Clustering", icon="🧩"),
        st.Page("views/predict.py", title="Prediksi Suara", icon="🎯"),
        st.Page("views/evaluation.py", title="Evaluasi", icon="📊"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    ui.sidebar()
nav.run()
