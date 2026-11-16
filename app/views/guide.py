import streamlit as st

import data as D
import ui

ui.setup("Panduan & Istilah", "Penjelasan singkat penelitian, alur kerja, dan arti setiap istilah/metrik di dashboard", "📘")
ui.flow()

ui.intro("Halaman ini menjelaskan <b>apa yang diteliti</b>, <b>bagaimana alurnya</b>, dan <b>arti setiap istilah</b> yang muncul di "
         "dashboard. Baca halaman ini dulu bila baru pertama kali membuka dashboard.")

with ui.panel("Tentang penelitian"):
    st.markdown(f"""
**Pertanyaan utama:** apakah **FastDTW** (versi cepat/aproksimasi) bisa menggantikan **DTW** (versi exact) sebagai ukuran jarak
untuk mengelompokkan (clustering) dan mengenali (klasifikasi) suara mesin, tanpa banyak mengubah hasilnya?

**Data:** MIMII DUE, rekaman suara mesin industri **fan** (kipas) dan **pump** (pompa), 10 detik per rekaman, dari beberapa
*section* (kondisi operasi berbeda). Tiap rekaman punya label:
- **domain**: *source* (kondisi asal) atau *target* (kondisi berbeda, mis. kecepatan/beban lain);
- **kondisi**: *normal* atau *anomaly* (mesin rusak/tidak wajar).

**Ide dasarnya:** suara diubah menjadi deret waktu fitur (MFCC atau Log-Mel). Jarak antar rekaman dihitung dengan DTW dan FastDTW,
lalu dipakai untuk (1) **fuzzy clustering** dan (2) **prediksi k-NN**. Hasil kedua jarak dibandingkan: seberapa mirip, seberapa cepat,
dan apakah kesimpulannya sama.
""")

with ui.panel("Alur kerja (sesuai urutan menu)"):
    st.markdown(f"""
| No | Tahap | Yang dilakukan | Halaman |
|---|---|---|---|
| 1 | Dataset | Mendata semua rekaman (mesin, section, domain, kondisi) | 🗂️ Dataset |
| 2 | Preprocessing | Semua audio diseragamkan: {D.SR} Hz, mono, DC offset dihapus, volume dinormalkan | 🔊 Audio Processing |
| 3 | Ekstraksi fitur | Audio → MFCC ({D.N_MFCC} koefisien) dan Log-Mel ({D.N_MELS} band) per frame | 🎛️ Ekstraksi Fitur |
| 4 | Jarak | Jarak tiap pasangan rekaman dihitung dengan DTW dan FastDTW | 📏 DTW vs FastDTW |
| 5 | Clustering | Fuzzy C-Medoids mengelompokkan rekaman berdasarkan matriks jarak | 🧩 Fuzzy Clustering |
| 6 | Prediksi | k-NN menebak label rekaman baru dari rekaman paling mirip | 🎯 Prediksi Suara |
| 7 | Evaluasi | Membandingkan kualitas, ketepatan, kecepatan, dan konsistensi | 📊 Evaluasi |
""")

c1, c2 = st.columns(2)
with c1:
    with ui.panel("Istilah: metode"):
        st.markdown("""
- **DTW (Dynamic Time Warping)**: mengukur kemiripan dua deret waktu dengan cara "meregangkan/menyusutkan" sumbu waktu agar
  bagian yang mirip saling bertemu. Hasilnya **exact** (pasti optimal) tetapi lambat: waktunya ∝ panjang² deret.
- **FastDTW**: versi cepat DTW. Mencari jalur pada resolusi kasar dulu, lalu memperhalusnya hanya di sekitar jalur itu
  (**radius** = lebar daerah pencarian). Lebih cepat, tetapi **aproksimasi**: jaraknya bisa sedikit lebih besar dari DTW,
  tidak pernah lebih kecil.
- **Euclid**: jarak biasa frame-ke-frame tanpa warping, dipakai sebagai pembanding paling sederhana.
- **MFCC**: ringkasan bentuk spektrum suara (mirip cara telinga manusia), sedikit koefisien → ringkas.
- **Log-Mel**: energi suara per pita frekuensi skala Mel (dalam dB), lebih rinci daripada MFCC.
- **Fuzzy C-Medoids (FCMdd)**: clustering di mana tiap rekaman boleh menjadi anggota beberapa cluster sekaligus dengan
  **derajat keanggotaan** 0–1 (jumlahnya 1). Pusat cluster (*medoid*) adalah rekaman nyata. **c** = jumlah cluster,
  **m** = tingkat "kekaburan" (m=2 umum dipakai).
- **MDS**: cara menggambar data berjarak ke bidang 2D. Titik yang berdekatan = rekaman yang mirip.
- **k-NN**: rekaman baru diberi label mayoritas dari **k** rekaman referensi paling mirip (k=1 → ikut yang paling mirip).
""")
with c2:
    with ui.panel("Istilah: metrik"):
        st.markdown("""
**Kemiripan FastDTW terhadap DTW**
- **Pearson / Spearman** (−1…1): seberapa searah jarak FastDTW dengan DTW. ≥0,95 = sangat mirip.
- **MAPE (%)**: rata-rata selisih relatif jarak FastDTW terhadap DTW. Makin kecil makin baik.
- **Speedup (×)**: berapa kali lebih cepat. Ada dua pembanding: DTW **numba** (sudah dioptimasi, sangat cepat) dan
  DTW **Python murni** (pembanding yang adil untuk FastDTW Python).

**Kualitas cluster**
- **PC** (Partition Coefficient, 1/c…1): makin tinggi makin tegas keanggotaannya.
- **MPC** (Modified PC, 0…1): PC yang sudah dinormalkan terhadap c. Makin tinggi makin baik.
- **PE** (Partition Entropy, 0…log c): makin **rendah** makin tegas.
- **Silhouette** (−1…1): seberapa terpisah antar-cluster. >0,5 jelas; 0,25–0,5 cukup; <0,25 lemah.
- **ARI / NMI** (≈0…1): kecocokan cluster dengan label sebenarnya (domain atau kondisi). 0 = tidak cocok/acak, 1 = sama persis.
- **ARI DTW vs FastDTW**: seberapa sama hasil clustering memakai DTW dan memakai FastDTW. 1 = identik.

**Prediksi**
- **Akurasi**: persentase rekaman uji yang ditebak benar.
- **F1 makro**: rata-rata F1 tiap kelas, adil bila jumlah kelas tidak seimbang.
- **Kesamaan prediksi**: persentase rekaman yang ditebak sama oleh FastDTW dan DTW.
""")

with ui.panel("Cara memakai dashboard"):
    st.markdown("""
- Pindah halaman lewat menu kiri. Tombol pilihan (mesin, fitur, section, dll.) ada di bagian atas tiap halaman.
- Setiap panel punya **keterangan singkat** di bawah judulnya; buka **📖 Cara membaca** untuk penjelasan lebih rinci.
- Kotak **💡 Kesimpulan** berisi interpretasi otomatis dari angka di halaman itu.
- Arahkan kursor ke grafik untuk melihat nilainya; seret untuk zoom, klik dua kali untuk kembali.
- Tema terang/gelap: menu **⋮** (kanan atas) → Light / Dark / System.
""")
