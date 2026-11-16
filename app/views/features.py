import numpy as np
import plotly.express as px
import streamlit as st

import data as D
import ui

ui.setup("Ekstraksi Fitur", f"MFCC ({D.N_MFCC} koef.) dan Log-Mel ({D.N_MELS} band) · FFT {D.N_FFT}, hop {D.HOP}", "🎛️")
ui.flow("Fitur")
ui.intro("Gelombang suara diubah menjadi <b>deret fitur</b> per potongan waktu pendek (frame). Ada dua jenis fitur yang dibandingkan: "
         f"<b>MFCC</b> ({D.N_MFCC} angka per frame, ringkasan bentuk spektrum) dan <b>Log-Mel</b> ({D.N_MELS} angka per frame, energi "
         "per pita frekuensi). Deret inilah yang dibandingkan oleh DTW dan FastDTW.")
ui.need(D.csv(D.PROC_MANIFEST), "Jalankan scripts/05–07 dulu.")


@st.fragment
def halaman():
    machine = ui.seg("Mesin", D.MACHINES, "fm")
    ch = D.file_choices(machine, "features")
    if not ch:
        ui.note("Belum ada fitur. Jalankan scripts/06–07 lalu 14_pack_for_dashboard.py.")
        return
    f = st.selectbox("File (★ = contoh yang dipaketkan)", list(ch), format_func=ch.get, key=f"ff{machine}")
    cols = st.columns(2)
    for col, ft in zip(cols, D.FEATURE_TYPES):
        with col:
            A, shape = D.feature_preview(machine, f, ft)
            if A is None:
                st.info(f"{ft}: file fitur tidak tersedia di versi ini.", icon="ℹ️")
                continue
            fig = px.imshow(np.round(A, 2), aspect="auto", origin="lower", color_continuous_scale=ui.SEQ,
                            labels=dict(x="frame (pratinjau)", y="koefisien" if ft == "MFCC" else "pita Mel", color="nilai"))
            desc = ("Tiap baris = satu koefisien MFCC; baris bawah (koefisien 0) mewakili energi total."
                    if ft == "MFCC" else "Tiap baris = satu pita frekuensi (bawah = nada rendah, atas = nada tinggi).")
            ui.show(fig, 380, f"{ft} · ukuran asli {shape[0]}×{shape[1]}", desc + " Warna terang = nilai tinggi.")
    st.caption("Pratinjau: frame dirata-rata per blok (maks. 160 kolom) supaya ringan; analisis tetap memakai nilai asli.")
    ui.how("""
- **Sumbu datar** = waktu (frame), **sumbu tegak** = koefisien/pita, **warna** = nilai.
- Pola garis **mendatar** yang stabil = suara mesin yang konstan (khas kipas/pompa normal).
- Garis **tegak** terang yang muncul tiba-tiba = bunyi singkat (ketukan, benturan), sering muncul pada anomali.
- Ukuran asli ditulis *baris × frame*; DTW membandingkan dua deret ini frame demi frame sambil boleh menggeser waktu.
""")


halaman()
