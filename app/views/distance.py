import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui

ui.setup("DTW vs FastDTW", "Seberapa dekat FastDTW (aproksimasi) dengan DTW exact pada kelompok data yang sama", "📏")
ui.flow("Jarak DTW/FastDTW")
ui.intro("Untuk setiap <b>grup</b> (satu mesin · satu section · satu jenis fitur), jarak antara <b>setiap pasangan rekaman</b> "
         "dihitung dua kali: dengan DTW (exact) dan FastDTW (aproksimasi). Halaman ini menjawab: <b>seberapa mirip jarak dari "
         "FastDTW dibanding DTW?</b> Bila hampir sama, FastDTW aman dipakai. Pilih grup dengan tombol di bawah.")
if not D.groups_available():
    st.warning("Belum ada matriks jarak. Jalankan scripts/08 & 09 (lalu 14).", icon="⚠️")
    st.stop()


@st.fragment
def halaman():
    machine, sec, feat = ui.group_filters("d")
    g = D.group(machine, sec, feat)
    if g is None or "DTW" not in g or "FastDTW" not in g:
        st.warning("Matriks DTW/FastDTW untuk grup ini belum lengkap (scripts/08 & 09).", icon="⚠️")
        return
    A, B = g["DTW"], g["FastDTW"]
    iu = np.triu_indices_from(A, 1)
    a, b = A[iu].astype(np.float64), B[iu].astype(np.float64)
    rel = (b - a) / np.maximum(a, 1e-12) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah rekaman (N)", len(A), border=True)
    k2.metric("Pearson", f"{np.corrcoef(a, b)[0, 1]:.4f}", border=True)
    k3.metric("Spearman", f"{ui.spearman(a, b):.4f}", border=True)
    k4.metric("MAPE", f"{np.abs(rel).mean():.2f}%", delta=f"maks {np.abs(rel).max():.1f}%", delta_color="off", border=True)
    r = float(np.corrcoef(a, b)[0, 1])
    mape = float(np.abs(rel).mean())
    same_nn = float(np.mean(np.argmin(A + np.eye(len(A)) * 1e18, 1) == np.argmin(B + np.eye(len(B)) * 1e18, 1)))
    ui.insight(f"Pada grup <b>{machine} section {sec:02d} · {feat}</b> ({len(A)} rekaman, {len(a):,} pasangan), jarak FastDTW "
               f"<b>{ui.says_corr(r)}</b> dengan DTW (Pearson {r:.4f}) dan rata-rata hanya berbeda {mape:.2f}% "
               f"(<b>{ui.says_err(mape)}</b>). Untuk {same_nn * 100:.0f}% rekaman, \"tetangga paling mirip\" menurut FastDTW "
               "sama dengan menurut DTW.")
    ui.how("""
- **N** = jumlah rekaman di grup ini. Jumlah pasangan = N×(N−1)/2.
- **Pearson** (−1…1) = apakah jarak FastDTW naik-turun searah dengan DTW. ≥0,95 berarti sangat mirip.
- **Spearman** = sama, tapi hanya melihat **urutan** (rekaman mana lebih mirip dari yang lain). Ini yang paling penting untuk
  clustering & k-NN, karena keduanya hanya peduli urutan kedekatan.
- **MAPE** = rata-rata selisih jarak dalam persen. "maks" = selisih terbesar pada satu pasangan.
""", "📖 Arti angka-angka di atas")

    view = ui.seg("Tampilan", ["Matriks jarak", "Sebar DTW vs FastDTW", "Distribusi galat"], "dview")
    if view == "Matriks jarak":
        c1, c2 = st.columns(2)
        for col, M, n in [(c1, A, "DTW"), (c2, B, "FastDTW")]:
            with col:
                ui.show(px.imshow(np.round(M, 4), color_continuous_scale=ui.SEQ, labels=dict(color="jarak")), 440,
                        f"Matriks jarak {n}", "Baris & kolom = rekaman; warna = jarak (gelap = mirip, terang = berbeda). "
                        "Diagonal selalu 0 (rekaman dengan dirinya sendiri).")
        st.caption("Bila kedua matriks terlihat hampir sama polanya, FastDTW menghasilkan jarak yang setara dengan DTW. "
                   "Kotak-kotak gelap = kelompok rekaman yang saling mirip (calon cluster).")
    elif view == "Sebar DTW vs FastDTW":
        idx = np.random.default_rng(0).choice(len(a), min(len(a), 5000), replace=False)
        fig = go.Figure(go.Scattergl(x=a[idx], y=b[idx], mode="markers", name="pasangan",
                                     marker=dict(size=4, opacity=.4, color=ui.PALETTE[0])))
        lim = [float(a.min()), float(a.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="y = x", line=dict(dash="dash", color=ui.PALETTE[3])))
        fig.update_layout(xaxis_title="jarak DTW (exact)", yaxis_title="jarak FastDTW (aproksimasi)")
        ui.show(fig, 480, "Jarak per pasangan: DTW vs FastDTW",
                "Tiap titik = satu pasangan rekaman. Titik tepat di garis putus-putus (y = x) berarti jaraknya sama persis.",
                "- Titik **di atas** garis: FastDTW memberi jarak lebih besar (aproksimasi kurang optimal).\n"
                "- Titik **rapat sepanjang garis** = FastDTW sangat akurat.\n"
                "- Titik **menyebar jauh** dari garis = FastDTW kurang akurat untuk pasangan itu.")
        st.caption(f"Ditampilkan {len(idx):,} dari {len(a):,} pasangan (sampel acak tetap).")
    else:
        ui.show(px.histogram(x=rel, nbins=40, color_discrete_sequence=ui.PALETTE,
                             labels=dict(x="galat relatif FastDTW terhadap DTW (%)", y="jumlah pasangan")), 420,
                "Sebaran galat relatif", "Berapa persen jarak FastDTW lebih besar dari DTW, untuk semua pasangan. "
                "Makin banyak batang yang menumpuk di dekat 0%, makin akurat FastDTW.")
        st.caption("Galat positif = FastDTW memberi jarak lebih besar dari DTW exact (FastDTW tidak pernah lebih kecil secara teori).")

    am = D.csv(D.RESULTS / "metrics" / "approximation_metrics.csv")
    if am is not None:
        row = am[(am.machine == machine) & (am.section == sec) & (am.feature == feat)]
        if len(row):
            with st.expander("Semua metrik grup ini (dari evaluasi)"):
                st.dataframe(row.T.set_axis(["nilai"], axis=1).astype(str), width="stretch")


halaman()
