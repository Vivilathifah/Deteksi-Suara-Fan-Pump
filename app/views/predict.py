import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D
import predlib
import ui

ui.setup("Prediksi Suara Mesin", "Tebak label dari suara dengan k-NN: jarak DTW vs FastDTW (dan Euclid sebagai pembanding)", "🎯")
ui.flow("Prediksi 1-NN")
ui.intro("Selain clustering, jarak DTW/FastDTW juga bisa dipakai untuk <b>mengenali</b> suara: rekaman baru diberi label "
         "yang sama dengan rekaman referensi <b>paling mirip</b> (k-Nearest Neighbor). Di tab <b>Coba tebak suara</b> kamu bisa "
         "mencoba satu rekaman; di tab <b>Evaluasi model</b> terlihat akurasi pada semua rekaman uji.")

avail = D.pred_available()
if not avail:
    st.warning("Belum ada model prediksi. Jalankan `python 13_predict.py` di folder scripts/ dulu, "
               "lalu muat ulang halaman ini.", icon="⚠️")
    st.stop()
predlib.start()  # bila belum berjalan; tidak menunggu

c1, c2 = st.columns([2, 1])
with c1:
    task = ui.seg("Tugas prediksi", list(avail), "pt", fmt=lambda t: D.TASK_NAMES.get(t, t))
with c2:
    feat = ui.seg("Fitur", list(avail[task]), f"pf{task}")
refs = D.refs(avail[task][feat])
classes = sorted(set(refs["yr"]))
METHODS = D.METHODS

tab_try, tab_eval = st.tabs(["🎤 Coba tebak suara", "📊 Evaluasi model"])


LABEL_TXT = {
    "normal": ("NORMAL", "Mesin terdengar bekerja wajar (sehat).", "✅", ui.GREEN),
    "anomaly": ("ANOMALI", "Ada suara tidak wajar — kemungkinan mesin bermasalah atau rusak.", "⚠️", ui.ORANGE),
    "fan": ("KIPAS (fan)", "Suara ini dikenali sebagai kipas industri.", "🌀", ui.BLUE),
    "pump": ("POMPA (pump)", "Suara ini dikenali sebagai pompa industri.", "💧", ui.CYAN),
}
METHOD_TXT = {"DTW": "DTW (exact)", "FastDTW": "FastDTW (cepat)", "Euclid": "Euclid (tanpa warping)"}


def lbl(c):
    return LABEL_TXT.get(str(c), (str(c).upper(), "", "🔹", ui.BLUE))


def nice(c):
    return lbl(c)[0].split(" (")[0].capitalize() if str(c) not in ("fan", "pump") else lbl(c)[0]


def judge(r, k):
    """Keyakinan sederhana: seberapa jauh kelas lain dibanding kelas yang ditebak, plus suara tetangga (k>1)."""
    d, y = r["dist"], refs["yr"]
    p = r["pred"]
    d_in = float(d[y == p].min())
    others = [c for c in classes if c != p]
    c_out = min(others, key=lambda c: d[y == c].min()) if others else None
    d_out = float(d[y == c_out].min()) if c_out is not None else np.nan
    margin = (d_out - d_in) / d_out if c_out is not None and d_out > 0 else np.nan
    votes = int((y[np.argsort(d)[:k]] == p).sum())
    level = ("Tinggi", ui.GREEN) if margin >= .2 else (("Sedang", ui.YELLOW) if margin >= .07 else ("Rendah", ui.ORANGE))
    return dict(d_in=d_in, d_out=d_out, c_out=c_out, margin=margin, votes=votes, level=level)


def play(fid, title, sub):
    p = D.pred_audio(fid)
    st.markdown(f'<div class="pt">{title}</div><div class="pd">{sub}</div>', unsafe_allow_html=True)
    if p is None:
        st.caption("🔇 Audio rekaman ini tidak tersedia di versi ini (jalankan `14_pack_for_dashboard.py` atau buka secara lokal).")
    else:
        st.audio(str(p))


@st.fragment
def coba():
    counts = " dan ".join(f"<b>{int((refs['yr'] == c).sum())}</b> {nice(c).lower()}" for c in classes)
    ui.note(f"Komputer punya <b>{len(refs['yr'])}</b> rekaman contoh yang labelnya sudah diketahui ({counts}). "
            "Rekaman baru ditebak dengan mencari contoh yang <b>suaranya paling mirip</b>, lalu memakai label contoh itu.")

    with st.expander("⚙️ Pengaturan lanjutan (boleh dibiarkan)"):
        k1, k2 = st.columns(2)
        k_opts = [1, 3, 5, 7, 9]
        with k1:
            k = st.select_slider("k (jumlah tetangga)", k_opts, value=refs["k"] if refs["k"] in k_opts else 1)
        with k2:
            radius = st.slider("Radius FastDTW", 1, 10, refs["radius"])
        st.caption("**k** = berapa rekaman paling mirip yang ikut \"memilih\" label (k=1: ikut yang paling mirip saja). "
                   "**Radius** = lebar daerah pencarian FastDTW; makin besar makin mendekati DTW tetapi makin lambat.")

    src = ui.seg("Sumber suara", ["Pilih contoh", "Unggah .wav"], "psrc")
    xq, true, sig, qid = None, None, None, None
    with ui.panel("1 · Dengarkan rekaman yang akan ditebak"):
        if src == "Pilih contoh":
            hide = st.toggle("Sembunyikan label asli (tebak dulu sendiri sebelum komputer menjawab!)", value=True)
            i = st.selectbox("Pilih rekaman uji", range(len(refs["idq"])),
                             format_func=lambda i: f"Rekaman #{i + 1}" if hide else
                             f"Rekaman #{i + 1} · {refs['idq'][i]}  ·  label asli: {nice(refs['yq'][i])}")
            if hide:
                st.caption("Nama file disembunyikan karena nama file MIMII mengandung labelnya (normal/anomaly).")
            qid = str(refs["idq"][i])
            p = D.pred_audio(qid)
            if p is not None:
                st.audio(str(p))
            else:
                st.caption("🔇 Audio rekaman ini tidak tersedia di versi ini, tetapi tetap bisa ditebak.")
            xq, true, sig = refs["Xq"][i], str(refs["yq"][i]), ("ex", task, feat, int(i), k, radius)
        else:
            up = st.file_uploader("Unggah rekaman mesin (.wav, sebaiknya ±10 detik)", type=["wav"])
            if up is not None:
                data = up.getvalue()
                st.audio(data, format="audio/wav")
                try:
                    with st.spinner("Membaca & mengekstrak fitur…"):
                        P = predlib.get()
                        y = P.preprocess(P.load_audio(io.BytesIO(data)))
                        A0 = P.extract(y, refs["feature"]).T.astype(np.float64)
                        A = P.fit_length(A0, refs["T"])
                    xq, sig = np.ascontiguousarray((A - refs["mu"]) / refs["sd"]), ("up", task, feat, hash(data), k, radius)
                    qid = up.name
                    if len(A0) != refs["T"]:
                        st.caption(f"Panjang rekaman ({len(A0)} frame) disamakan dengan data contoh ({refs['T']} frame).")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Gagal membaca/mengekstrak audio: {e}")

        if not predlib.ready() and predlib.error() is None:
            st.caption("⏳ Mesin penghitung sedang disiapkan di latar belakang (sekali saja). Tombol tetap bisa ditekan.")
        elif predlib.error() is not None:
            st.error(f"Modul prediksi gagal dimuat: {predlib.error()}")
        go_ = st.button("🔍 Tebak suara ini", type="primary", disabled=xq is None, width="stretch")

    if go_:
        with st.spinner("Membandingkan suara dengan semua rekaman contoh…"):
            P = predlib.get()
            st.session_state["pred"] = dict(sig=sig, out=P.predict_query(xq, refs, k, radius), true=true, xq=xq, qid=qid)

    res = st.session_state.get("pred")
    if not (res and res["sig"] == sig and xq is not None):
        return
    P = predlib.get()
    out, true = res["out"], res["true"]
    main = out["DTW"]
    J = judge(main, k)
    name, meaning, icon, color = lbl(main["pred"])
    nn = main["nn"]

    # ---------- 2. jawaban utama
    correct = None if true is None else (main["pred"] == true)
    verdict_true = ("" if true is None else
                    f'<div class="vd-true">Label sebenarnya: <b>{nice(true)}</b> → tebakan '
                    f'{"<b style=color:#14B88E>BENAR ✅</b>" if correct else "<b style=color:#EF4444>SALAH ❌</b>"}</div>')
    conf_pct = 0 if np.isnan(J["margin"]) else max(0, min(1, J["margin"] / .4))
    st.markdown(f'''
<div class="vd" style="--vc:{color}">
  <div class="vd-l">2 · Jawaban komputer</div>
  <div class="vd-main"><span class="vd-ic">{icon}</span><div>
     <div class="vd-k">Suara ini diperkirakan</div><div class="vd-v">{name}</div>
     <div class="vd-m">{meaning}</div></div></div>
  <div class="vd-conf"><span>Tingkat keyakinan: <b style="color:{J['level'][1]}">{J['level'][0]}</b></span>
     <div class="vd-bar"><div style="width:{conf_pct * 100:.0f}%;background:{J['level'][1]}"></div></div></div>
  {verdict_true}
</div>''', unsafe_allow_html=True)

    # ---------- 3. kenapa
    why = (f"Dari {len(refs['yr'])} rekaman contoh, yang suaranya <b>paling mirip</b> adalah <b>{refs['idr'][nn]}</b> "
           f"yang berlabel <b>{nice(refs['yr'][nn])}</b> (jarak {J['d_in']:.2f}).")
    if J["c_out"] is not None:
        why += (f" Rekaman <b>{nice(J['c_out'])}</b> yang paling mirip jaraknya {J['d_out']:.2f}, yaitu "
                f"<b>{J['margin'] * 100:.0f}% lebih jauh</b>. Karena suara ini "
                + {"Tinggi": "jelas lebih dekat", "Sedang": "lebih dekat", "Rendah": "hanya sedikit lebih dekat"}[J["level"][0]]
                + f" ke kelompok {nice(main['pred'])}, komputer menebak <b>{nice(main['pred'])}</b>."
                if J["margin"] > 0 else
                f" Rekaman <b>{nice(J['c_out'])}</b> terdekat juga hampir sama dekatnya ({J['d_out']:.2f}), "
                "jadi tebakan ini kurang meyakinkan.")
    if k > 1:
        why += f" Dari {k} rekaman paling mirip, <b>{J['votes']}</b> berlabel {nice(main['pred'])}."
    lvl_txt = {"Tinggi": "Suara ini jelas lebih mirip ke satu kelompok.",
               "Sedang": "Suara ini lebih mirip ke satu kelompok, tetapi tidak terlalu jauh berbeda dengan kelompok lain.",
               "Rendah": "Suara ini berada di antara dua kelompok; hasilnya sebaiknya diperiksa ulang oleh teknisi."}[J["level"][0]]
    ui.insight(why + "<br><br><b>Arti tingkat keyakinan:</b> " + lvl_txt, "3 · Kenapa jawabannya begitu?")

    # ---------- 4. dengarkan & bandingkan
    with ui.panel("4 · Dengarkan & bandingkan", "Putar ketiganya. Rekaman yang ditebak seharusnya terdengar lebih mirip dengan "
                                                 "contoh di tengah daripada contoh di kanan."):
        c1, c2, c3 = st.columns(3)
        with c1:
            if res.get("qid") and src == "Pilih contoh":
                play(res["qid"], "🎤 Rekaman yang ditebak", f"{res['qid']}")
            else:
                st.markdown('<div class="pt">🎤 Rekaman yang ditebak</div><div class="pd">File yang kamu unggah (diputar di atas)</div>',
                            unsafe_allow_html=True)
        with c2:
            play(str(refs["idr"][nn]), f"{icon} Contoh paling mirip · {nice(refs['yr'][nn])}",
                 f"{refs['idr'][nn]} · jarak {J['d_in']:.2f}")
        with c3:
            if J["c_out"] is not None:
                y = refs["yr"]
                cols = np.where(y == J["c_out"])[0]
                j = cols[np.argmin(main["dist"][cols])]
                play(str(refs["idr"][j]), f"{lbl(J['c_out'])[2]} Contoh {nice(J['c_out'])} terdekat",
                     f"{refs['idr'][j]} · jarak {J['d_out']:.2f} (lebih jauh)")

    # ---------- 5. tiga cara mengukur kemiripan
    with ui.panel("5 · Apakah cara cepat (FastDTW) memberi jawaban yang sama?",
                  "Kemiripan suara dihitung dengan tiga cara. DTW adalah cara paling teliti; FastDTW versi cepatnya; Euclid "
                  "cara paling sederhana tanpa menyesuaikan perbedaan tempo."):
        cols = st.columns(3)
        for col, m in zip(cols, METHODS):
            r = out[m]
            n_, _, ic, cl = lbl(r["pred"])
            ok = "" if true is None else ("✅" if r["pred"] == true else "❌")
            same = "" if m == "DTW" else ("sama dengan DTW" if r["pred"] == main["pred"] else "BERBEDA dari DTW")
            col.markdown(ui.kpi(f"{ic} {nice(r['pred'])} {ok}", f"{METHOD_TXT[m]} · {r['seconds'] * 1e3:,.0f} ms", cl,
                                same or "acuan", ui.GREEN if same in ("", "sama dengan DTW") else ui.ORANGE)
                         .replace('class="kpi"', 'class="kpi sm"'), unsafe_allow_html=True)
        agree = out["FastDTW"]["pred"] == main["pred"]
        sp = out["FastDTW"]["seconds"] / max(main["seconds"], 1e-9)
        st.write("")
        (st.success if agree else st.warning)(
            ("FastDTW memberi jawaban yang **sama** dengan DTW. " if agree else "FastDTW memberi jawaban yang **berbeda** dari DTW. ")
            + f"Waktu hitung: DTW {main['seconds'] * 1e3:,.0f} ms, FastDTW {out['FastDTW']['seconds'] * 1e3:,.0f} ms "
            + f"({'lebih lambat' if sp > 1 else 'lebih cepat'} {max(sp, 1 / sp):.1f}×, karena DTW di sini sudah dipercepat numba).")

    # ---------- 6. peringkat rekaman paling mirip
    with ui.panel("6 · 5 rekaman contoh paling mirip (menurut DTW)",
                  "Kemiripan 100% = paling mirip di antara semua contoh. Bila kelima teratas berlabel sama, tebakannya kuat."):
        d = main["dist"]
        order = np.argsort(d)[:5]
        tb = pd.DataFrame({"Peringkat": np.arange(1, len(order) + 1), "Rekaman contoh": refs["idr"][order].astype(str),
                           "Label": [f"{lbl(refs['yr'][j])[2]} {nice(refs['yr'][j])}" for j in order],
                           "Jarak": d[order].round(3), "Kemiripan": (d[order].min() / np.maximum(d[order], 1e-12))})
        st.dataframe(tb, hide_index=True, width="stretch",
                     column_config={"Kemiripan": st.column_config.ProgressColumn("Kemiripan", format="percent", min_value=0,
                                                                                max_value=1)})

    # ---------- 7. grafik jarak
    rows = [dict(metode=METHOD_TXT[m], kelas=nice(c), jarak_min=float(out[m]["dist"][refs["yr"] == c].min()))
            for m in METHODS for c in classes]
    ui.show(px.bar(pd.DataFrame(rows), x="kelas", y="jarak_min", color="metode", barmode="group",
                   color_discrete_sequence=ui.PALETTE, labels=dict(jarak_min="jarak ke contoh terdekat", kelas="")), 360,
            "7 · Seberapa dekat suara ini ke tiap kelompok",
            "Batang lebih pendek = lebih mirip. Kelompok dengan batang terpendek adalah jawabannya. Bandingkan antar kelompok "
            "dalam warna yang sama (skala Euclid berbeda dengan DTW/FastDTW).")

    with st.expander("🔬 Untuk yang penasaran: jalur warping DTW (rekaman ini vs contoh paling mirip)"):
        ref_x = refs["Xr"][nn]
        path, dd = P.dtw_path(res["xq"], ref_x)
        cost = np.sqrt(((res["xq"][:, None, :] - ref_x[None, :, :]) ** 2).sum(-1))
        fig = px.imshow(np.round(cost, 3), origin="lower", aspect="auto", color_continuous_scale=ui.SEQ[::-1],
                        labels=dict(x=f"waktu pada contoh ({refs['idr'][nn]})", y="waktu pada rekaman", color="beda"))
        fig.add_trace(go.Scatter(x=[p[1] for p in path], y=[p[0] for p in path], mode="lines",
                                 line=dict(color="#F59E0B", width=3), name="jalur pencocokan"))
        ui.show(fig, 460, f"Jalur pencocokan DTW · jarak = {dd:.3f}",
                "DTW mencocokkan bagian-bagian suara yang mirip walau temponya sedikit berbeda. Garis emas menunjukkan "
                "bagian mana dicocokkan dengan bagian mana. Garis lurus diagonal = tempo sama; berbelok = DTW menyesuaikan tempo.")


def evaluasi():
    st.caption("Semua rekaman uji ditebak dengan tiap jenis jarak, lalu dibandingkan dengan label aslinya.")
    mt = D.csv(D.PRED_RES / "metrics.csv")
    pr = D.csv(D.PRED_RES / f"predictions_{task}_{feat}.csv")
    if mt is None or pr is None:
        st.info("Belum ada hasil evaluasi untuk tugas ini. Jalankan `python 13_predict.py`.")
        return
    sub = mt[(mt.task == task) & (mt.feature == feat)].set_index("method").reindex(METHODS)
    if sub.n_test.isna().all():
        st.info("Belum ada hasil evaluasi untuk tugas ini. Jalankan `python 13_predict.py`.")
        return
    st.caption(f"{int(sub.n_test.iloc[0])} rekaman uji · {int(sub.n_ref.iloc[0])} rekaman referensi · k = {int(sub.k.iloc[0])} · "
               f"radius FastDTW = {int(sub.radius.iloc[0])}")
    cols = st.columns(3)
    for col, m in zip(cols, METHODS):
        with col:
            st.metric(f"Akurasi {m}", f"{sub.loc[m, 'accuracy'] * 100:.1f}%", border=True,
                      delta=f"F1 {sub.loc[m, 'f1_macro']:.3f}", delta_color="off")
    l, r = st.columns(2)
    with l:
        ui.show(px.bar(sub.reset_index(), x="method", y="accuracy", color="method", color_discrete_sequence=ui.PALETTE,
                       range_y=[0, 1], title="Akurasi"), 340)
    with r:
        ui.show(px.bar(sub.reset_index(), x="method", y="ms_per_query", color="method", color_discrete_sequence=ui.PALETTE,
                       log_y=True, title="Waktu per rekaman uji (ms, skala log)"), 340)
    st.caption(f"Kesamaan prediksi FastDTW dengan DTW: **{sub.loc['FastDTW', 'agree_with_DTW'] * 100:.1f}%**")
    ad, af, ae = (sub.loc[m, "accuracy"] for m in METHODS)
    ui.insight(f"Akurasi DTW {ad * 100:.1f}%, FastDTW {af * 100:.1f}%, Euclid {ae * 100:.1f}%. "
               + ("FastDTW <b>setara</b> DTW" if abs(ad - af) <= .02 else
                  f"FastDTW {'lebih rendah' if af < ad else 'lebih tinggi'} {abs(ad - af) * 100:.1f} poin dari DTW")
               + f", dan menebak label yang sama dengan DTW pada {sub.loc['FastDTW', 'agree_with_DTW'] * 100:.0f}% rekaman. "
               + ("Warping (DTW/FastDTW) membantu dibanding Euclid." if max(ad, af) > ae + .02 else
                  "Pada tugas ini warping tidak memberi keuntungan berarti dibanding Euclid."))
    ui.how("""
- **Akurasi** = persentase rekaman uji yang ditebak benar. **F1** = rata-rata ketepatan per kelas (adil bila kelas tidak seimbang).
- **Matriks konfusi**: baris = label sebenarnya, kolom = tebakan. Angka di **diagonal** = tebakan benar; di luar diagonal = salah.
- **Waktu per rekaman** = lama menghitung jarak satu rekaman uji ke semua referensi.
""")

    labels = sorted(set(pr.y_true.astype(str)) | {str(v) for m in METHODS for v in pr[f"pred_{m}"]})
    cols = st.columns(3)
    for col, m in zip(cols, METHODS):
        ct = pd.crosstab(pr.y_true.astype(str), pr[f"pred_{m}"].astype(str)).reindex(index=labels, columns=labels, fill_value=0)
        with col:
            ui.show(px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=ui.SEQ,
                              labels=dict(x="prediksi", y="sebenarnya"), title=f"Matriks konfusi · {m}"), 340)
    with st.expander("Detail prediksi per rekaman uji"):
        show = pr.copy()
        for m in METHODS:
            show[f"benar_{m}"] = show.y_true.astype(str) == show[f"pred_{m}"].astype(str)
        st.dataframe(show.round(4), width="stretch", hide_index=True)
        st.download_button("Unduh CSV", show.to_csv(index=False).encode(), f"predictions_{task}_{feat}.csv", "text/csv")
    ui.note("Catatan metodologi: standarisasi z-score dihitung dari data referensi saja; data uji dipisah per rekaman induk "
            "sehingga potongan (segmen) dari rekaman yang sama tidak bocor antara referensi dan uji.")


with tab_try:
    coba()
with tab_eval:
    evaluasi()
