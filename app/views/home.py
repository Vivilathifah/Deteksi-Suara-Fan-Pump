"""Ringkasan — tata letak big-screen: kiri (komposisi, radar, sampel), tengah (kurva, peringkat, gauge), kanan (status, ranking)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui

ui.setup("MIMII DUE · DTW vs FastDTW", "Fuzzy clustering suara mesin fan & pump · MFCC & Log-Mel · Fuzzy C-Medoids", "🎧")
ui.intro("Ringkasan seluruh hasil penelitian dalam satu layar: apakah <b>FastDTW</b> (cepat, aproksimasi) memberi hasil yang "
         "sama baiknya dengan <b>DTW</b> (exact) untuk mengelompokkan dan mengenali suara mesin. "
         "Istilah yang belum jelas dijelaskan di menu <b>📘 Panduan &amp; Istilah</b>.")

man, pm = D.csv(D.MANIFEST), D.csv(D.PROC_MANIFEST)
am = D.csv(D.RESULTS / "metrics" / "approximation_metrics.csv")
cm = D.csv(D.RESULTS / "metrics" / "cluster_metrics.csv")
ag = D.csv(D.RESULTS / "metrics" / "cluster_agreement.csv")
pmet = D.csv(D.PRED_RES / "metrics.csv")

if am is None or not len(am) or cm is None or not len(cm):
    ui.kpi_row([(f"{0 if man is None else len(man):,}", "Audio (raw)"),
                (f"{0 if pm is None else len(pm):,}", "Audio (processed)", ui.GREEN)])
    ui.note("Hasil belum tersedia. Jalankan pipeline (scripts 02–14 atau notebook) lalu muat ulang halaman ini.")
    st.stop()

# ------------------------------------------------------------------ KPI atas
sp = am["speedup_vs_python_dtw"].dropna()
ui.kpi_row([
    (f"{len(man):,}" if man is not None else "–", "Rekaman audio (raw)", ui.BLUE),
    (str(len(am)), "Grup jarak (mesin × section × fitur)", ui.CYAN),
    (f"{am.pearson.mean():.3f}", "Korelasi DTW ↔ FastDTW", ui.GREEN, "Pearson"),
    (f"{am.MAPE_pct.mean():.2f}%", "Galat relatif FastDTW (MAPE)", ui.ORANGE, "↓ makin baik"),
    ("–" if sp.empty else f"{sp.mean():.1f}×", "FastDTW vs DTW Python murni", ui.YELLOW, "lebih cepat"),
])

# ------------------------------------------------------------------ kesimpulan keseluruhan
r_mean, mape = am.pearson.mean(), am.MAPE_pct.mean()
pts = [f"Jarak FastDTW <b>{ui.says_corr(r_mean)}</b> dengan DTW (korelasi rata-rata {r_mean:.3f}) dan selisihnya "
       f"<b>{ui.says_err(mape)}</b> (MAPE {mape:.2f}%)."]
ari = float(ag.ARI_DTW_vs_FastDTW.mean()) if ag is not None and len(ag) else None
if ari is not None:
    pts.append(f"Hasil clustering memakai DTW dan memakai FastDTW <b>{ui.says_ari(ari)}</b> (ARI rata-rata {ari:.2f}).")
best_sil = cm.silhouette.max()
pts.append(f"Struktur cluster terbaik: silhouette {best_sil:.3f}, artinya kelompok suaranya <b>{ui.says_sil(best_sil)}</b>.")
acc_d = acc_f = None
if pmet is not None and len(pmet):
    acc = pmet.groupby("method").accuracy.mean()
    acc_d, acc_f = acc.get("DTW"), acc.get("FastDTW")
    if acc_d is not None and acc_f is not None:
        pts.append(f"Akurasi prediksi: DTW {acc_d * 100:.1f}% vs FastDTW {acc_f * 100:.1f}% "
                   f"(selisih {abs(acc_d - acc_f) * 100:.1f} poin).")
if not sp.empty:
    num = am["speedup_pair_(DTW/FastDTW)"].mean()
    pts.append(f"Kecepatan: FastDTW <b>{sp.mean():.1f}× lebih cepat</b> dari DTW versi Python biasa, tetapi "
               + (f"<b>{1 / num:.1f}× lebih lambat</b>" if num < 1 else f"{num:.1f}× lebih cepat")
               + " dari DTW yang sudah dioptimasi numba.")
ok = r_mean >= .95 and (ari is None or ari >= .7) and (acc_d is None or acc_f is None or abs(acc_d - acc_f) <= .05)
verdict = ("➡️ Secara keseluruhan, FastDTW <b>layak dipakai sebagai pengganti DTW</b> pada data ini: hasilnya hampir sama."
           if ok else "➡️ Secara keseluruhan, FastDTW <b>belum bisa sepenuhnya menggantikan DTW</b> pada data ini; "
           "periksa grup dengan korelasi rendah atau ARI rendah di halaman 📏 dan 📊.")
ui.insight("<br>".join("• " + p for p in pts) + "<br>" + verdict, "Kesimpulan otomatis dari hasil saat ini")

left, mid, right = st.columns([1, 2.15, 1], gap="small")

# ------------------------------------------------------------------ KIRI
with left:
    with ui.panel("Komposisi dataset", "Proporsi rekaman per domain (source/target) dan kondisi (normal/anomali)."):
        if man is not None:
            comp = man.groupby(["domain", "condition"]).size().rename("n").reset_index()
            comp["lab"] = comp.domain + " · " + comp.condition
            fig = go.Figure(go.Pie(labels=comp.lab, values=comp.n, hole=.62, sort=False,
                                   marker=dict(colors=ui.PALETTE, line=dict(width=0)),
                                   textinfo="percent", textfont=dict(color="#fff", size=11)))
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05, x=.5, xanchor="center", font=dict(size=10)))
            ui.chart(fig, 270)

    with ui.panel("Kualitas cluster · DTW vs FastDTW", "Rata-rata 5 metrik kualitas (0–1, makin luas makin baik). Bila kedua bentuk berimpit, FastDTW setara DTW."):
        mets = ["PC", "MPC", "silhouette", "NMI_domain", "NMI_condition"]
        names = ["PC", "MPC", "Silhouette", "NMI dom.", "NMI kond."]
        g = cm.groupby("distance")[mets].mean()
        fig = go.Figure()
        for dist, color in [("DTW", ui.BLUE), ("FastDTW", ui.GREEN)]:
            if dist in g.index:
                r = g.loc[dist].clip(lower=0).tolist()
                fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=names + names[:1], fill="toself", name=dist,
                                              line=dict(color=color, width=2), opacity=.75))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False), domain=dict(x=[.14, .86], y=[.08, .96]), angularaxis=dict(tickfont=dict(size=10))),
                          legend=dict(orientation="h", y=-0.02, x=.5, xanchor="center"))
        ui.chart(fig, 280)

    with ui.panel("Sampel per section", "Jumlah rekaman (setelah preprocessing) tiap section untuk masing-masing mesin."):
        if pm is not None:
            cnt = pm.groupby(["machine", "section"]).size().rename("n").reset_index()
            fig = go.Figure()
            for i, (mc, s) in enumerate(cnt.groupby("machine")):
                fig.add_trace(go.Bar(x=[f"s{int(x):02d}" for x in s.section], y=s.n, name=mc,
                                     marker=dict(color=ui.PALETTE[i], opacity=.85, line=dict(width=0))))
            fig.update_layout(barmode="group", bargap=.35, legend=dict(orientation="h", y=1.15, x=0))
            ui.chart(fig, 200)

# ------------------------------------------------------------------ TENGAH
with mid:
    with ui.panel("Silhouette vs jumlah cluster (rata-rata seluruh grup)", "Silhouette = seberapa terpisah cluster (makin tinggi makin baik). Puncak kurva = jumlah cluster (c) paling cocok. Kurva DTW & FastDTW yang berimpit berarti hasilnya sama."):
        s = cm.groupby(["distance", "c"]).silhouette.mean().unstack(0)
        a, b = s.get("DTW"), s.get("FastDTW")
        k1, k2 = st.columns(2)
        if a is not None:
            k1.markdown(ui.kpi(f"{a.max():.3f}", f"DTW · terbaik di c = {int(a.idxmax())}", ui.BLUE, "exact", ui.GREEN),
                        unsafe_allow_html=True)
        if b is not None:
            diff = (b.max() - a.max()) / abs(a.max()) * 100 if a is not None and a.max() else 0
            k2.markdown(ui.kpi(f"{b.max():.3f}", f"FastDTW · terbaik di c = {int(b.idxmax())}", ui.ORANGE,
                               f"{diff:+.1f}%", ui.ORANGE), unsafe_allow_html=True)
        fig = go.Figure()
        for name, ser, color, fill in [("DTW", a, ui.BLUE, "rgba(59,164,255,0.35)"),
                                       ("FastDTW", b, ui.ORANGE, "rgba(255,122,89,0.30)")]:
            if ser is not None:
                fig.add_trace(go.Scatter(x=ser.index, y=ser.values, name=name, mode="lines+markers",
                                         line=dict(color=color, width=3, shape="spline"), marker=dict(size=7),
                                         fill="tozeroy", fillcolor=fill))
        fig.update_layout(xaxis=dict(title="jumlah cluster c", dtick=1), yaxis=dict(title="silhouette"),
                          legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), margin=dict(t=30))
        ui.chart(fig, 260)

    with ui.panel("Peringkat konfigurasi terbaik (silhouette)", "5 kombinasi mesin · section · fitur · jarak · c dengan cluster paling terpisah."):
        top = cm.sort_values("silhouette", ascending=False).head(5)
        ui.rank_tiles([(f"{r.silhouette:.3f}", f"{r.machine} s{int(r.section):02d} {r.feature} · {r.distance} · c={int(r.c)}")
                       for r in top.itertuples()])

    with ui.panel("Akurasi prediksi k-NN (rata-rata seluruh tugas)", "Persentase rekaman uji yang labelnya ditebak benar memakai tiap jenis jarak, beserta waktu hitung per rekaman."):
        if pmet is None or not len(pmet):
            ui.note("Belum ada hasil prediksi. Jalankan <code>13_predict.py</code>.")
        else:
            acc = pmet.groupby("method").accuracy.mean()
            agree = pmet[pmet.method == "FastDTW"].agree_with_DTW.mean()
            cols = st.columns(3)
            for col, (m, color) in zip(cols, [("DTW", ui.BLUE), ("FastDTW", ui.ORANGE), ("Euclid", ui.YELLOW)]):
                with col:
                    if m in acc:
                        ui.chart(ui.gauge(acc[m], m, color), 190)
                        ms = pmet[pmet.method == m].ms_per_query.mean()
                        st.markdown(f'<div class="legend" style="text-align:center;color:{color}">'
                                    f'{ms:,.1f} ms per rekaman</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="legend" style="text-align:center;margin-top:.4rem">Kesamaan prediksi FastDTW dengan DTW: '
                        f'<b>{agree * 100:.1f}%</b></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ KANAN
with right:
    with ui.panel("Status penelitian", "Cakupan data yang sudah dianalisis."):
        n_feat = am.feature.nunique()
        n_sec = am.groupby("machine").section.nunique().sum()
        rad = int(am.radius.dropna().iloc[0]) if "radius" in am and am.radius.notna().any() else 1
        st.markdown(
            f'<div class="legend">Mesin<br><span class="hl">{" · ".join(sorted(am.machine.unique()))}</span>'
            f'<br>Fitur<br><span class="hl">{" · ".join(sorted(am.feature.unique()))}</span>'
            f'<br>{n_sec} section · {n_feat} fitur · radius FastDTW = {rad}</div>', unsafe_allow_html=True)

    with ui.panel("Konsistensi cluster DTW ↔ FastDTW", "Seberapa sama pembagian cluster memakai DTW dan FastDTW (ARI)."):
        if ag is not None and len(ag):
            ari = float(ag.ARI_DTW_vs_FastDTW.mean())
            ui.chart(ui.gauge(max(ari, 0), "ARI rata-rata", ui.GREEN), 200)
            st.markdown('<div class="legend" style="text-align:center">1 = cluster identik</div>', unsafe_allow_html=True)
        else:
            ui.note("Belum ada data konsistensi.")

    with ui.panel("Peringkat grup · korelasi tertinggi", "Grup di mana jarak FastDTW paling mirip dengan DTW (1 = identik)."):
        r = am.assign(grup=ui.grp_label(am)).sort_values("pearson", ascending=False).head(6)
        ui.rank_bars([(g, v) for g, v in zip(r.grup, r.pearson)], "{:.4f}")
