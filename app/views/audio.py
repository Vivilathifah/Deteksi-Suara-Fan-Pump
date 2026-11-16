import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui

ui.setup("Audio Processing", f"Standarisasi: {D.SR} Hz · mono · hapus DC offset · normalisasi puncak 0,99", "🔊")
ui.flow("Preprocessing")
ui.intro("Sebelum dianalisis, setiap rekaman <b>diseragamkan</b> supaya perbandingannya adil: frekuensi sampel disamakan "
         f"({D.SR} Hz), stereo dijadikan mono, pergeseran nol (DC offset) dihapus, dan volume dinormalkan sehingga puncaknya 0,99. "
         "Bandingkan bentuk gelombang <b>Raw</b> (asli) dan <b>Processed</b> (setelah diseragamkan).")
ui.need(D.csv(D.PROC_MANIFEST), "Jalankan scripts/05_preprocessing.py dulu.")


def wave(env, sr, dur, name, color):
    """Waveform dari envelope min/max (1.200 titik) — tampak sama dengan aslinya, tapi ringan."""
    lo, hi = env
    t = np.linspace(0, dur, lo.shape[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=.6, color=color), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=.6, color=color), fill="tonexty",
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title="detik", yaxis_title="amplitudo")
    return fig


@st.fragment
def halaman():
    machine = ui.seg("Mesin", D.MACHINES, "am")
    ch = D.file_choices(machine, "audio")
    if not ch:
        ui.note("Belum ada audio. Jalankan <code>05_preprocessing.py</code> lalu <code>14_pack_for_dashboard.py</code>.")
        return
    f = st.selectbox("File (★ = contoh yang dipaketkan)", list(ch), format_func=ch.get, key=f"af{machine}")
    v = D.audio_view(machine, f)
    if not v:
        ui.note("File audio tidak tersedia di versi ini. Pilih contoh bertanda ★.")
        return
    cols = st.columns(2)
    for col, (kind, name, color) in zip(cols, [("raw", "Raw", ui.PALETTE[1]), ("proc", "Processed", ui.PALETTE[0])]):
        with col:
            if kind not in v:
                st.info(f"{name}: tidak tersedia di versi ini.", icon="ℹ️")
                continue
            env, sr, dur, peak, url = v[kind]
            ui.show(wave(env, sr, dur, name, color), 300, f"{name} ({sr} Hz)",
                    "Bentuk gelombang: sumbu datar = waktu, tinggi = kuatnya suara. "
                    + ("Ini rekaman asli." if kind == "raw" else "Setelah diseragamkan, puncaknya ±0,99."))
            ui.audio(url)
            st.caption(f"{name}: {dur:.1f} s · {sr} Hz · puncak {peak:.3f}")
    ui.how("""
- **Amplitudo** = kekuatan suara pada tiap saat. Area berwarna menunjukkan rentang naik-turun gelombang.
- **Raw** bisa berbeda volume antar rekaman; **Processed** disamakan sehingga jarak antar rekaman tidak dipengaruhi volume.
- Rekaman bisa dipotong menjadi beberapa **segmen** (id berakhiran `_segXX`) agar panjangnya seragam.
""")


halaman()
