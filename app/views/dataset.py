import pandas as pd
import plotly.express as px
import streamlit as st

import data as D
import ui

ui.setup("Dataset MIMII DUE", "Rekaman suara mesin fan & pump: domain source/target, kondisi normal/anomali", "🗂️")
ui.flow("Dataset")
ui.intro("Halaman ini menunjukkan <b>data apa saja yang dipakai</b>: berapa rekaman, dari mesin apa, section mana, "
         "dan berapa yang normal atau anomali. Setiap rekaman berdurasi 10 detik.")
man = ui.need(D.csv(D.MANIFEST), "Jalankan scripts/02–03 dulu (audio_manifest.csv belum ada).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total file", f"{len(man):,}", border=True)
c2.metric("Mesin", man.machine.nunique(), border=True)
c3.metric("Section", man.groupby("machine").section.nunique().sum(), border=True)
c4.metric("Anomali", f"{(man.condition == 'anomaly').sum():,}", border=True)
n_an, n = int((man.condition == "anomaly").sum()), len(man)
n_tg = int((man.domain == "target").sum())
ui.insight(f"Dari {n:,} rekaman, {n_an:,} ({n_an / n * 100:.0f}%) berlabel <b>anomali</b> dan {n_tg:,} ({n_tg / n * 100:.0f}%) "
           f"berasal dari domain <b>target</b>. "
           + ("Jumlah normal jauh lebih banyak dari anomali, jadi akurasi perlu dibaca bersama F1 makro. " if n_an / n < .3 else "")
           + "Label hanya dipakai untuk <b>menilai</b> hasil clustering; clustering sendiri tidak melihat label.", "Gambaran data")


@st.fragment
def komposisi():
    machines = st.multiselect("Filter mesin", sorted(man.machine.unique()), default=sorted(man.machine.unique()))
    m = man[man.machine.isin(machines)]
    if not len(m):
        st.info("Pilih minimal satu mesin.")
        return
    cnt = m.groupby(["machine", "section", "domain", "split", "condition"]).size().rename("n").reset_index()
    left, right = st.columns(2)
    with left:
        sb = cnt.assign(section=cnt.section.astype(str).str.zfill(2))
        ui.show(px.sunburst(sb, path=["machine", "section", "domain", "condition"], values="n",
                            color_discrete_sequence=ui.PALETTE), 460, "Komposisi dataset",
                "Lingkaran dalam → luar: mesin → section → domain → kondisi. Klik satu bagian untuk memperbesar, klik tengah untuk kembali.")
    with right:
        ui.show(px.bar(cnt, x="section", y="n", color="condition", facet_col="domain", barmode="stack",
                       color_discrete_sequence=ui.PALETTE), 460, "Jumlah rekaman per section",
                "Tinggi batang = jumlah rekaman; warna = normal/anomali; kiri = domain source, kanan = domain target.")
    with st.expander("Tabel jumlah per subset / mesin / section"):
        st.dataframe(m.groupby(["subset", "machine", "section"]).size().rename("n").reset_index(),
                     width="stretch", hide_index=True)


@st.fragment
def contoh_audio():
    st.subheader("Contoh audio")
    st.caption("Dengarkan rekaman aslinya. Suara mesin normal biasanya stabil dan berulang; anomali bisa terdengar sebagai "
               "bunyi ketukan, gesekan, atau dengung yang tidak biasa.")
    packed = set(D.packed_raw_ids())
    ids = list(man.file_id.head(500)) if D.has_local_raw() else []
    ids = sorted(set(ids) | (packed & set(man.file_id)), key=lambda f: (f not in packed, f))
    if not ids:
        ui.note("Audio tidak ada di versi ini. Jalankan <code>scripts/14_pack_for_dashboard.py</code> agar beberapa klip kecil ikut terunggah.")
        return
    f = st.selectbox("File (★ = klip kecil yang ikut dipaketkan, bisa diputar online)", ids,
                     format_func=lambda f: f"★ {f}" if f in packed else f)
    ui.audio(D.raw_audio_url(f))
    r = man[man.file_id == f].iloc[0]
    with st.expander("Metadata file"):
        st.json({k: (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items() if not pd.isna(v)})


komposisi()
contoh_audio()

v = D.csv(D.RESULTS / "tables" / "audio_validation.csv")
if v is not None:
    st.subheader("Validasi audio")
    st.caption("Pengecekan otomatis tiap file: bisa dibaca, durasi wajar, tidak hening, tidak clipping (terpotong). "
               "File bermasalah tercantum di tabel bawah.")
    st.progress(float(v.valid.mean()), text=f"Valid: {int(v.valid.sum())}/{len(v)}")
    bad = v[v.issues.fillna("") != ""]
    if len(bad):
        st.dataframe(bad, width="stretch", hide_index=True)
