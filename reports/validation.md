# Laporan validasi aktual

**16/16 PASS.** Eksekusi berurutan, satu kernel baru per notebook. Mode setiap run tercantum dalam tabel. Bab 8, 9, 10, 11, 12, 13, 14, 15 belum divalidasi dengan mode full.

Bukti mesin: [validation_results.json](validation_results.json). Output stream dan gambar tersimpan di masing-masing notebook. Hash source mencakup source seluruh sel kecuali ringkasan analisis yang dibangkitkan dari hasil; dependency/support hashes mengikat loader, framework, requirement, dan manifest data.

| Bab | Status | Mode | Sel kode | Detik | Mulai (Asia/Jakarta) |
|---:|---|---|---:|---:|---|
| 1 | PASS | quick | 3 | 1.697 | 2026-10-16T01:50:07.190862+07:00 |
| 2 | PASS | quick | 3 | 1.686 | 2026-10-16T01:50:08.892591+07:00 |
| 3 | PASS | quick | 7 | 1.587 | 2026-10-16T01:50:10.585588+07:00 |
| 4 | PASS | quick | 5 | 2.703 | 2026-10-16T01:50:12.177506+07:00 |
| 5 | PASS | quick | 6 | 1.877 | 2026-10-16T01:50:14.884968+07:00 |
| 6 | PASS | quick | 5 | 1.936 | 2026-10-16T01:50:16.767120+07:00 |
| 7 | PASS | quick | 4 | 1.827 | 2026-10-16T01:50:18.707446+07:00 |
| 8 | PASS | quick | 6 | 22.383 | 2026-10-16T01:50:20.543019+07:00 |
| 9 | PASS | quick | 6 | 3.603 | 2026-10-16T01:50:42.932257+07:00 |
| 10 | PASS | quick | 5 | 3.290 | 2026-10-16T01:50:46.540972+07:00 |
| 11 | PASS | quick | 7 | 15.833 | 2026-10-16T01:50:49.835891+07:00 |
| 12 | PASS | quick | 7 | 8.424 | 2026-10-16T01:51:05.675021+07:00 |
| 13 | PASS | quick | 10 | 1.826 | 2026-10-16T01:51:14.108887+07:00 |
| 14 | PASS | quick | 7 | 5.689 | 2026-10-16T01:51:15.940954+07:00 |
| 15 | PASS | quick | 7 | 4.396 | 2026-10-16T01:51:21.636955+07:00 |
| 16 | PASS | quick | 3 | 1.552 | 2026-10-16T01:51:26.037880+07:00 |

Total waktu eksekusi notebook pada run yang tersimpan: **80.309 detik**. Tidak termasuk pengembangan, unduhan, instalasi, atau run yang gagal sebelumnya; waktu ini bukan perkiraan untuk komputer lain.

## Lingkungan dan prosedur

Python `3.13.5`; `macOS-15.7.7-x86_64-i386-64bit-Mach-O`. Dependensi langsung: `numpy==2.5.4`, `matplotlib==3.11.2`, `nbformat==5.11.1`, `nbclient==0.11.0`, `ipykernel==7.4.0`, `phe==1.5.0`.

Validator menetapkan thread BLAS/OpenMP ke1 agar run kecil stabil, menolak unexpected cell errors, memeriksa urutan execution_count, dan mengekstrak metrik JSON dari output. Assertion pada notebook memeriksa dimensi, finite values, normalisasi probabilitas, contoh yang seharusnya konvergen, gradien numerik, multiuse graph, repeated indexing, pembagian data, dan dekripsi agregat. Model yang memang dimaksudkan gagal mempunyai assertion khusus; divergence finite bukan NaN/inf.

## Hasil dan interpretasi per bab

### Bab 1 — Introducing deep learning: why you should learn it

Pemeriksaan library dan operasi NumPy berhasil. Hasil ini membuktikan environment tersedia, belum menunjukkan pembelajaran model.

### Bab 2 — Fundamental concepts: how do machines learn?

Pada contoh sintetis tambahan, half squared error turun dari 2.000 menjadi 0.720. Perubahan satu bobot mengurangi error satu titik; belum ada pengujian generalisasi.

### Bab 3 — Introduction to neural prediction: forward propagation

Prediksi satu input 0.85; banyak input 0.98. Output jaringan bertingkat [0.21350000000000002, 0.14500000000000002, 0.5065] cocok antara loop dan NumPy. Kesamaan ini menguji operasi forward, bukan kualitas bobot.

### Bab 4 — Introduction to neural learning: gradient descent

Alpha besar menghasilkan squared error akhir 5.403e+16, sedangkan alpha kecil menghasilkan 1.485e-10. Divergence adalah hasil yang sengaja didemonstrasikan, bukan kegagalan validasi. Selisih gradien numerik 1.539e-11.

### Bab 5 — Learning multiple weights at a time: generalizing gradient descent

Bobot yang dibekukan tetap 0.1. Galat maksimum outer-product gradient check 6.535e-11, mendukung penurunan rumus untuk semua koefisien yang diuji.

### Bab 6 — Building your first deep neural network: introduction to backpropagation

SSE epoch ke-60 1.5055623e-05; akurasi empat titik latihan 100%. Nilai ini cocok dengan contoh kecil buku hingga toleransi numerik. Tidak ada test independen sehingga 100% tidak boleh disebut akurasi generalisasi.

### Bab 7 — How to picture neural networks: in your head and on paper

Ekspresi bertahap dan ringkas sama. Jumlah parameter 16 sesuai 3×4 + 4×1 tanpa bias. Prediksi berasal dari bobot demonstrasi acak.

### Bab 8 — Learning signal and ignoring noise: introduction to regularization and batching

Pada 500 gambar test, baseline 84.2%, dropout koreksi 85.4%, dan mini-batch koreksi 82.8%. Gap train-validation baseline 16.7%; validation memilih epoch 20 dari 30. Perbandingan ini juga melibatkan hidden size/alpha/jumlah update berbeda, sehingga bukan ablation dropout yang terkontrol.

### Bab 9 — Modeling probabilities and nonlinearities: activation functions

Setelah 30 epoch, akurasi test listing 68.4%, koreksi 75.4% pada 500 contoh. Galat gradien koreksi 1.115e-10. Hasil mendukung konsistensi gradien koreksi, bukan klaim semua perubahan selalu meningkatkan akurasi.

### Bab 10 — Neural learning about edges and corners: intro to convolutional neural networks

Pada 256 train/200 test selama 4 epoch, listing mencapai 7.0%, koreksi 19.5%. Hasil koreksi masih rendah: run ini memeriksa alur CNN, bukan model yang sudah matang. Galat gradien kernel 1.875e-10.

### Bab 11 — Neural networks that understand language: king – man + woman == ?

Sentiment: listing 77.0% versus koreksi 69.9% pada 1000 test, dengan 1200 train. Koreksi derivative justru memberi akurasi lebih rendah pada anggaran ini: faktor sigmoid mengecilkan langkah, sedangkan alpha tidak dituning ulang. Benar secara kalkulus tidak otomatis lebih baik pada hyperparameter yang sama. CBOW melakukan 31,600 update; BCE sampling akhir 0.5148 bukan perplexity vocabulary penuh.

### Bab 12 — Neural networks that write like Shakespeare: recurrent layers for variable-length data

Gradien BPTT koreksi memiliki galat 5.724e-10; indeks listing 8.199e-03. Setelah 800 update, test perplexity 81.908 dengan 5 token OOV diabaikan. Angka masih tinggi; latihan singkat belum mempelajari pola bahasa dengan baik.

### Bab 13 — Introducing automatic optimization: let’s build a deep learning framework

Autograd lolos gradient check (galat 6.331e-12) dan uji multiuse/indexing. RNN dilatih 80 langkah, loss akhir 3.3734, akurasi training 14.0%; ini belum mengukur generalisasi.

### Bab 14 — Learning to write like Shakespeare: long short-term memory

RNN 2 epoch dan LSTM 3 epoch memakai 8,000 karakter training. NLL akhir masing-masing 3.1911 dan 3.2162. Held-out perplexity RNN 21.534, LSTM 23.198. LSTM tidak lebih baik pada run ini; arsitektur lebih kompleks memerlukan anggaran/tuning yang sesuai. Galat gradient check gerbang 1.679e-10; kualitas teks harus dibaca pada sampel aktual.

### Bab 15 — Deep learning on unseen data: introducing federated learning

Dengan 600 baris train dan 199 test, akurasi centralized 94.47%, federated 94.47%, dan agregasi terenkripsi 94.47%. Sebanyak 1 duplikat test terhadap train dibuang. Galat maksimum decrypt-average 2.776e-17; kesamaan aritmetika ini tidak membuktikan protokol aman untuk produksi. Demonstrasi satu email mengungkap 5 kata.

### Bab 16 — Where to go from here: a brief guide

Terdapat 16 notebook. Pemeriksaan ini hanya memvalidasi struktur; bukti eksekusi setiap bab tersedia dalam laporan validator.

## Konfigurasi yang benar-benar diuji versus penuh

| Bab | Quick tersimpan | Full tersedia, belum dijalankan pada pembuatan awal |
|---|---|---|
| 1–7,16 | Semua konsep/contoh kecil; Bab 6: 60 epoch | Sama; tidak ada training besar baru |
| 8 | 1.000 train,200 validation,500 test; baseline30,dropout20 per cabang,batch40 per cabang epoch | Baseline350,dropout300,batch300; test10.000 |
| 9 | 1.000 train/500 test,30 epoch,dua cabang | 300 epoch/test10.000 |
| 10 | 256 train/200 test,4 epoch,kernel3×3×16,batch128,dua cabang | 1.000 train/10.000 test,300 epoch |
| 11 | Sentiment1.200 train/1.000 test,2 epoch,H100. CBOW200 review×maks80 token×2,dua cabang | Sentiment24.000 train; CBOW25.000 review penuh×2 |
| 12 | CBOW ringan mandiri; bAbI1.000 baris,800 update,H10,dua cabang | CBOW penuh;30.000 update BPTT |
| 13 | Semua primitive; RNN80 update,H16,batch100 | RNN1.000 update |
| 14 | 8.000 karakter,H32,batch8,RNN2/LSTM3 epoch; held-out2.000 akhir | Seluruh corpus,H512,RNN100 epoch B32/T16,LSTM110 epoch B16/T25; tanpa held-out |
| 15 | 600 train,199 test setelah dedup,V1.003,batch64,3 ronde | Semua train setelah filter,semua test setelah dedup,V training penuh,batch500,3 ronde |

“Full” berarti anggaran latihan penuh teradaptasi, bukan jaminan hasil identik. Lihat reproduction_notes.md untuk perubahan matematika/normalisasi/split yang tetap berlaku. Tidak ada output full yang dibuat-buat.

## Batas interpretasi dan temuan validasi

- Bab 6 dan13 mengukur training pada data kecil; tidak ada klaim generalisasi dari skor tersebut.
- Test MNIST quick adalah subset deterministik test resmi. Validation Bab 8 berasal dari train resmi di luar1.000 contoh latihan. Tidak dipakai untuk memilih parameter berdasarkan test.
- Perbandingan dropout bukan ablation terkontrol: hidden size,alpha,dan jumlah update ikut berbeda mengikuti tahap buku.
- CBOW vocabulary/distribusi negatif menggunakan corpus tak berlabel seperti buku. Jangan melaporkan skor sentiment sebagai evaluasi CBOW.
- bAbI tokenisasi literal masih menyimpan tab/tanda baca dan evaluasi mengabaikan OOV yang dihitung. Ini bukan skor QA resmi.
- Spam: satu duplikat konten lintas train/test ditemukan pada subsetquick dan dihapus. Test199 setelah koreksi. Tidak ada raw email pribadi atau key enkripsi dipublikasikan.
- Uji enkripsi membuktikan kesetaraan agregasi numerik, tidak membuktikan keamanan arsitektur satu-proses/128-bit.
- Run awal menemukan kesalahan string escape pada notebook4,5,7 dan overlap data notebook15. Semua diperbaiki; tabel di atas berasal dari run akhir, bukan run gagal.
- Diagram dan kurva dihasilkan oleh Matplotlib dari output aktual; sumbu,legend,dan kesimpulan diperiksa terhadap angka. Grafik tidak mengimpor gambar/output buku.
- Tidak ada bab berstatus belum dijalankan dalam runquick. **Trainingfull Bab 8–15 tetap belum diverifikasi** dan tidak menjadi syarat agar notebookquick selesai.

## Mengulang

```bash
python scripts/validate.py --mode quick
python scripts/render_reports.py
```

Perubahan source setelah run akan ditolak oleh renderer sampai notebook terkait divalidasi ulang. JSON disimpan per bab sehingga re-run sebagian memperbarui bab tersebut saja. Jangan menggabungkan metrik dari mode berbeda tanpa membaca kolom mode setiap bab.
