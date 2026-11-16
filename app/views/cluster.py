import pandas as pd
import plotly.express as px
import streamlit as st

import data as D
import ui

ui.setup("Fuzzy Clustering", "Fuzzy C-Medoids pada matriks jarak · proyeksi MDS · derajat keanggotaan", "🧩")
ui.flow("Fuzzy clustering")
ui.intro("Rekaman dikelompokkan otomatis berdasarkan jaraknya dengan <b>Fuzzy C-Medoids</b>, <b>tanpa melihat label</b>. "
         "\"Fuzzy\" berarti tiap rekaman punya <b>derajat keanggotaan</b> (0–1) di setiap cluster, bukan hanya satu. "
         "Bandingkan hasil memakai jarak DTW dan FastDTW: bila gambarnya mirip, FastDTW tidak mengubah kesimpulan.")
pm = ui.need(D.csv(D.PROC_MANIFEST), "Belum ada processed manifest.").set_index("file_id")
if not D.groups_available():
    st.warning("Belum ada hasil clustering. Jalankan scripts/08–10 (lalu 14).", icon="⚠️")
    st.stop()


@st.fragment
def halaman():
    machine, sec, feat = ui.group_filters("c")
    c1, c2 = st.columns([1, 2])
    with c1:
        dist = ui.seg("Jarak", D.DISTANCE_TYPES, "cd")
    t = D.tag(machine, sec, feat, dist)
    lab = D.csv(D.CLUSTERING / "labels" / f"{t}.csv")
    if lab is None:
        st.warning("Belum ada hasil clustering untuk grup ini (scripts/10).", icon="⚠️")
        return
    cs = sorted(int(c.split("c")[-1]) for c in lab.columns if c.startswith("label_c"))
    with c2:
        c = st.select_slider("Jumlah cluster (c)", cs, value=cs[0], key=f"cc{t}") if len(cs) > 1 else cs[0]
    color_by = ui.seg("Warna titik", ["cluster", "domain", "condition"], "ccol",
                      fmt=lambda s: {"cluster": "Cluster", "domain": "Domain", "condition": "Kondisi"}[s])

    mem = D.csv(D.CLUSTERING / "membership" / f"{t}_c{c}.csv")
    if not (D.PACK / "groups").exists():
        st.caption("ℹ️ Proyeksi MDS dihitung langsung (lambat). Jalankan `14_pack_for_dashboard.py` agar instan.")
    with st.spinner("Memuat proyeksi MDS…"):
        xy = D.mds_coords(machine, sec, feat, dist)
    if mem is None or xy is None or len(xy) != len(mem):
        st.warning("Data keanggotaan / matriks jarak tidak cocok. Jalankan ulang scripts/10 lalu 14.", icon="⚠️")
        return
    U = mem.filter(like="u").values
    meta = pm.reindex(mem.file_id)
    df = mem.assign(x=xy[:, 0], y=xy[:, 1], cluster=(U.argmax(1) + 1).astype(str), max_u=U.max(1),
                    domain=meta.domain.values, condition=meta.condition.values)

    k1, k2, k3 = st.columns(3)
    k1.metric("Rekaman", len(df), border=True)
    k2.metric("Keanggotaan maks. rata-rata", f"{df.max_u.mean():.3f}", border=True)
    k3.metric("Titik ambigu (u maks < 0,6)", f"{(df.max_u < .6).mean() * 100:.1f}%", border=True)
    cmx = D.csv(D.RESULTS / "metrics" / "cluster_metrics.csv")
    txt = (f"Dengan jarak <b>{dist}</b> dan <b>c = {c}</b>, rata-rata keyakinan keanggotaan {df.max_u.mean():.2f} "
           f"({'cukup tegas' if df.max_u.mean() >= .7 else 'kurang tegas'}; nilai minimum yang mungkin 1/c = {1 / c:.2f}). "
           f"{(df.max_u < .6).mean() * 100:.0f}% rekaman berada \"di perbatasan\" antar-cluster.")
    if cmx is not None:
        row = cmx[(cmx.machine == machine) & (cmx.section == sec) & (cmx.feature == feat) & (cmx.distance == dist) & (cmx.c == c)]
        if len(row):
            rr = row.iloc[0]
            txt += (f" Silhouette {rr.silhouette:.3f} → cluster <b>{ui.says_sil(rr.silhouette)}</b>. "
                    f"Kecocokan dengan label kondisi (ARI) {rr.ARI_condition:.2f}"
                    f"{' → cluster ikut memisahkan normal/anomali' if rr.ARI_condition >= .3 else ' → cluster belum memisahkan normal/anomali'}.")
    ui.insight(txt)

    ui.show(px.scatter(df, x="x", y="y", color=color_by, size="max_u", hover_data=["file_id", "domain", "condition", "max_u"],
                       color_discrete_sequence=ui.PALETTE, category_orders={color_by: sorted(df[color_by].dropna().unique())},
                       ), 520, f"Peta kemiripan (MDS) · {t} · c={c}",
            "Tiap titik = satu rekaman. Titik yang berdekatan = suaranya mirip. Ukuran titik = seberapa yakin rekaman itu masuk "
            "cluster-nya (besar = yakin). Ganti \"Warna titik\" ke Domain/Kondisi untuk melihat apakah cluster sesuai label asli.",
            "- Warna **Cluster**: hasil pengelompokan.\n- Warna **Domain**/**Kondisi**: label sebenarnya. Bila pola warnanya mirip "
            "dengan warna Cluster, berarti clustering berhasil menemukan perbedaan itu.\n- Sumbu x dan y tidak punya satuan; "
            "yang penting hanya jarak antar titik.")
    l, r = st.columns(2)
    with l:
        ui.show(px.imshow(pd.crosstab(df.cluster, df.domain), text_auto=True, aspect="auto",
                          color_continuous_scale=ui.SEQ), 320, "Cluster × domain",
                "Jumlah rekaman tiap cluster per domain. Bila tiap cluster didominasi satu domain, cluster memisahkan source/target.")
    with r:
        ui.show(px.imshow(pd.crosstab(df.cluster, df.condition), text_auto=True, aspect="auto",
                          color_continuous_scale=ui.SEQ), 320, "Cluster × kondisi",
                "Jumlah rekaman tiap cluster per kondisi. Bila satu cluster berisi sebagian besar anomali, cluster itu menangkap kerusakan.")
    ui.show(px.histogram(df, x="max_u", nbins=20, color_discrete_sequence=ui.PALETTE,
                         labels=dict(max_u="derajat keanggotaan tertinggi", count="jumlah rekaman")), 320,
            "Seberapa yakin tiap rekaman masuk cluster-nya",
            "Mendekati 1 = rekaman jelas milik satu cluster. Mendekati 1/c = rekaman ragu-ragu di antara beberapa cluster.")
    with st.expander("Derajat keanggotaan (tabel) — u1, u2, … = keanggotaan di cluster 1, 2, …; jumlah tiap baris = 1"):
        st.dataframe(mem.round(4), width="stretch", hide_index=True)
        st.download_button("Unduh CSV", mem.to_csv(index=False).encode(), f"{t}_c{c}.csv", "text/csv")


halaman()
