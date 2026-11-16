import plotly.express as px
import streamlit as st

import data as D
import ui

ui.setup("Evaluasi & Perbandingan", "Kualitas cluster · ketepatan aproksimasi · kecepatan · konsistensi DTW vs FastDTW", "📊")
ui.flow("Evaluasi")
ui.intro("Perbandingan lengkap DTW vs FastDTW dari empat sisi: <b>kualitas cluster</b>, <b>ketepatan aproksimasi</b> jarak, "
         "<b>kecepatan</b>, dan <b>konsistensi</b> hasil clustering. Angka-angka di sini adalah bahan utama untuk bab hasil "
         "dan pembahasan laporan.")
METRIC_INFO = {
    "silhouette": "Silhouette (−1…1): seberapa terpisah cluster satu dengan lainnya. Makin tinggi makin baik.",
    "PC": "Partition Coefficient (1/c…1): seberapa tegas keanggotaan fuzzy. Makin tinggi makin tegas.",
    "MPC": "Modified PC (0…1): PC yang dinormalkan terhadap c, jadi adil untuk membandingkan c berbeda. Makin tinggi makin baik.",
    "PE": "Partition Entropy (0…log c): tingkat keraguan keanggotaan. Makin RENDAH makin baik.",
    "ARI_domain": "ARI terhadap label domain (≈0…1): seberapa cocok cluster dengan pembagian source/target. 0 = acak, 1 = sama persis.",
    "NMI_domain": "NMI terhadap label domain (0…1): informasi bersama antara cluster dan label source/target.",
    "ARI_condition": "ARI terhadap label kondisi (≈0…1): seberapa cocok cluster dengan pembagian normal/anomali.",
    "NMI_condition": "NMI terhadap label kondisi (0…1): informasi bersama antara cluster dan label normal/anomali.",
    "pearson": "Pearson (−1…1): kesearahan jarak FastDTW dengan DTW. ≥0,95 = sangat mirip.",
    "spearman": "Spearman (−1…1): kesamaan URUTAN kedekatan. Paling relevan untuk clustering & k-NN.",
    "MAPE_pct": "MAPE (%): rata-rata selisih relatif jarak FastDTW terhadap DTW. Makin kecil makin baik.",
    "MAE": "MAE: rata-rata selisih mutlak jarak (satuan jarak). Makin kecil makin baik.",
    "max_rel_err_pct": "Galat relatif maksimum (%): selisih terbesar pada satu pasangan rekaman.",
}
cm = ui.need(D.csv(D.RESULTS / "metrics" / "cluster_metrics.csv"), "Jalankan scripts/11_evaluation.py dulu.")
am = D.csv(D.RESULTS / "metrics" / "approximation_metrics.csv")
ag = D.csv(D.RESULTS / "metrics" / "cluster_agreement.csv")
if am is not None and len(am):
    am = am.assign(grup=ui.grp_label(am))
else:
    am = None


@st.fragment
def kualitas():
    metric = ui.seg("Metrik", ["silhouette", "PC", "MPC", "PE", "ARI_domain", "NMI_domain", "ARI_condition", "NMI_condition"], "em")
    g = cm.groupby(["machine", "feature", "distance", "c"])[metric].mean().reset_index()
    ui.show(px.line(g, x="c", y=metric, color="feature", line_dash="distance", facet_col="machine", markers=True,
                    color_discrete_sequence=ui.PALETTE), 440, f"{metric} menurut jumlah cluster (rata-rata seluruh section)",
            METRIC_INFO.get(metric, "") + " Garis penuh = DTW, garis putus-putus = FastDTW; warna = jenis fitur.")
    piv = cm.groupby("distance")[metric].mean()
    if {"DTW", "FastDTW"} <= set(piv.index):
        d, f = piv["DTW"], piv["FastDTW"]
        better = "lebih rendah" if metric == "PE" else "lebih tinggi"
        fbest = cm.groupby("feature")[metric].mean()
        fb = fbest.idxmin() if metric == "PE" else fbest.idxmax()
        ui.insight(f"Rata-rata {metric}: DTW {d:.3f} vs FastDTW {f:.3f} (selisih {abs(d - f):.3f}). "
                   + ("Keduanya praktis sama. " if abs(d - f) < .02 else f"{'DTW' if (d > f) != (metric == 'PE') else 'FastDTW'} sedikit lebih baik. ")
                   + f"Fitur dengan {metric} {better}: <b>{fb}</b>.")


@st.fragment
def aproksimasi():
    m2 = ui.seg("Metrik", ["pearson", "spearman", "MAPE_pct", "MAE", "max_rel_err_pct"], "ea")
    ui.show(px.bar(am, x="grup", y=m2, color="feature", barmode="group", color_discrete_sequence=ui.PALETTE), 440,
            f"{m2} per grup (mesin · section · fitur)", METRIC_INFO.get(m2, ""))
    worst = am.loc[am.pearson.idxmin()]
    ui.insight(f"Rata-rata Pearson {am.pearson.mean():.4f} dan MAPE {am.MAPE_pct.mean():.2f}% → FastDTW "
               f"<b>{ui.says_corr(am.pearson.mean())}</b> dengan DTW. Grup paling lemah: <b>{worst.grup}</b> "
               f"(Pearson {worst.pearson:.4f}, MAPE {worst.MAPE_pct:.2f}%).")


tabs = st.tabs(["Kualitas cluster", "Aproksimasi", "Kecepatan", "Konsistensi", "Tabel & unduh"])
with tabs[0]:
    kualitas()
with tabs[1]:
    if am is None:
        st.info("Belum ada metrik aproksimasi.")
    else:
        aproksimasi()
with tabs[2]:
    if am is None:
        st.info("Belum ada metrik kecepatan.")
    else:
        long = am.melt(id_vars="grup", value_vars=["pair_ms_dtw", "pair_ms_fastdtw", "pair_ms_dtw_python"])
        long["variable"] = long.variable.map({"pair_ms_dtw": "DTW (numba)", "pair_ms_fastdtw": "FastDTW (Python)",
                                              "pair_ms_dtw_python": "DTW (Python murni)"})
        ui.show(px.bar(long, x="grup", y="value", color="variable", barmode="group", log_y=True,
                       color_discrete_sequence=ui.PALETTE, labels=dict(value="ms per pasangan", variable="")), 440,
                "Waktu hitung satu pasangan rekaman (milidetik, skala log)",
                "Makin pendek batang makin cepat. Skala log: tiap garis bantu = 10× lipat.")
        t_d, t_f, t_p = am.pair_ms_dtw.mean(), am.pair_ms_fastdtw.mean(), am.pair_ms_dtw_python.mean()
        ui.insight(f"Rata-rata per pasangan: DTW numba {t_d:.2f} ms, FastDTW {t_f:.2f} ms, DTW Python murni {t_p:.1f} ms. "
                   f"Jadi FastDTW <b>{t_p / t_f:.1f}× lebih cepat</b> dari DTW Python murni (pembanding yang adil, sama-sama Python), "
                   + (f"tetapi <b>{t_f / t_d:.1f}× lebih lambat</b> dari DTW yang dioptimasi numba." if t_f > t_d else
                      f"dan {t_d / t_f:.1f}× lebih cepat dari DTW numba.")
                   + " Keduanya perlu dilaporkan supaya perbandingannya jujur.")
with tabs[3]:
    if ag is None or not len(ag):
        st.info("Belum ada data konsistensi (butuh DTW dan FastDTW).")
    else:
        pv = ag.assign(grup=ui.grp_label(ag)).pivot_table(index="grup", columns="c", values="ARI_DTW_vs_FastDTW")
        ui.show(px.imshow(pv, text_auto=".2f", aspect="auto", color_continuous_scale=ui.SEQ, zmin=0, zmax=1,
                          labels=dict(x="jumlah cluster c", y="grup", color="ARI")), max(320, 28 * len(pv) + 120),
                "Kesamaan hasil clustering DTW vs FastDTW (ARI)",
                "Tiap kotak = satu grup pada satu nilai c. 1 = pembagian cluster identik, 0 = tidak ada kesamaan (seperti acak).")
        ui.insight(f"Rata-rata ARI {ag.ARI_DTW_vs_FastDTW.mean():.2f}: hasil clustering memakai FastDTW "
                   f"<b>{ui.says_ari(ag.ARI_DTW_vs_FastDTW.mean())}</b> dengan memakai DTW. "
                   f"{(ag.ARI_DTW_vs_FastDTW >= .9).mean() * 100:.0f}% konfigurasi memiliki ARI ≥ 0,9.")
with tabs[4]:
    st.caption("Tabel lengkap semua metrik (bisa diurutkan dengan klik judul kolom) dan tombol unduh CSV untuk laporan.")
    for name, d in [("cluster_metrics", cm), ("approximation_metrics", am), ("cluster_agreement", ag)]:
        if d is not None:
            st.markdown(f"**{name}**")
            d = d.drop(columns=["grup"], errors="ignore")
            st.dataframe(d.round(4), width="stretch", hide_index=True)
            st.download_button(f"Unduh {name}.csv", d.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"dl_{name}")
