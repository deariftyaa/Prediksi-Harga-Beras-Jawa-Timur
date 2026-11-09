## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | 1. Riza Hanafi / 244311026 |
|            | 2. Dhea Riftia / 244311005 |
| Judul ide | [ Prediksi Harga Beras Menjelang Idul Fitri di Jawa Timur Menggunakan Pendekatan Machine Learning dan Deep Learning ] |
| SDG & target yang dipilih | SDG 2: Tanpa Kelaparan (Zero Hunger)


Target 2.c: Mengadopsi langkah-langkah untuk menjamin berfungsinya pasar komoditas pangan serta memfasilitasi akses tepat waktu terhadap informasi pasar, termasuk cadangan pangan, guna membantu membatasi gejolak harga pangan yang ekstrem. |
| Masalah nyata yang diangkat | Lonjakan dan lonjakan fluktuasi harga beras menjelang Hari Raya Idul Fitri di Jawa Timur sering memicu inflasi daerah dan menyulitkan daya beli masyarakat. Pemerintah daerah dan instansi terkait (seperti Bulog/TPID) membutuhkan alat prediksi yang akurat dan antisipatif untuk intervensi pasar dini serta menjaga stabilitas ketahanan pangan. |
| Tugas ML (Target & jenis tugas) | • Jenis Tugas: Time-Series Forecasting / Regression (Prediksi Deret Waktu).


• Fitur Target: Harga beras rerata tingkat konsumen (Harian/Mingguan) dalam Rupiah (IDR/kg).


• Fitur Prediktor: Harga historis beras, volume produksi/pasokan, tingkat curah hujan/iklim, indikator hari libur/keagamaan (variabel dummy Idul Fitri), dan harga komoditas substitusi/input.|
| Objective & metrik utama | • Objective: Membangun model forecasting deret waktu berakurasi tinggi untuk memprediksi tren dan lonjakan harga beras hingga 30–60 hari menjelang Idul Fitri.


• Metrik Utama: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan Mean Absolute Percentage Error (MAPE). |
| Dataset (sumber, URL, ukuran, jenis data) | • Sumber: PIHPS Nasional (Bank Indonesia), BPS Jawa Timur, dan BMKG.


• URL: https://hargapangan.id/ dan https://jatim.bps.go.id/


• Ukuran Data: Data historis harian/mingguan periode 2018–2025 (ratusan hingga ribuan entris deret waktu).


• Jenis Data: Tabular Time-Series Data (Format CSV/XLSX). |
| Rencana tiga model (baseline / advanced / deep learning) | 1. Baseline Model: ARIMA / SARIMAX (Model statistik deret waktu konvensional).


2. Advanced ML Model: XGBoost Regressor / Random Forest Regressor (dengan feature engineering lag waktu dan variabel kalender).


3. Deep Learning Model: Long Short-Term Memory (LSTM) atau Gated Recurrent Unit (GRU) untuk menangkap pola temporal jangka panjang dan non-linearitas. |
| Risiko & pertimbangan etika (bias, privasi) | • Privasi: Menggunakan data publik (open data) agregat tingkat provinsi/kabupaten, tidak melibatkan data individu atau privasi pribadi.


• Bias & Volatilitas Ekstrem: Risiko akumulasi error akibat gejolak eksternal yang tiba-tiba (seperti kebijakan kuota impor dadakan atau bencana alam ekstrem). Model akan menyertakan analisis ketidakpastian (confidence interval) agar tidak memicu ramalan spekulatif pasar (panic buying).|
| Referensi awal (minimal 2) | 1. Kementerian Pertanian & Bank Indonesia. (2023). Laporan Panel Harga Pangan Nasional dan Analisis Pola Musiman Harga Komoditas Pokok Jelang Hari Besar Keagamaan Nasional (HBKN).


2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.


3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). |

**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu