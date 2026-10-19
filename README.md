# Grokking-Deep-Learning

Tugas individu **Code Reproduction + Theoretical Deep-Dive** berdasarkan *Grokking Deep Learning* — Andrew W. Trask, Manning, ©2019. Repository berisi **16 Jupyter Notebook**, penjelasan orisinal bahasa Indonesia, implementasi NumPy from scratch, output nyata, eksperimen berlabel, dan peta sumber PDF.

| Identitas | Nilai |
|---|---|
| Nama | Tarannia Raisha Putri |
| NIM | 101032300231 |
| Kelas | TK4702 |
| Mata kuliah | Deep Learning |

**Hasil validasi: 16/16 notebook PASS** dari kernel bersih secara berurutan. Bab 1–7 dan16 memuat konsep/contoh kecil lengkap; mode setiap bab tercantum dalam tabel. **Bab 8, 9, 10, 11, 12, 13, 14, 15 belum divalidasi dengan mode full.** PASS menyatakan sel dan pemeriksaan berhasil; bukan kesamaan semua akurasi dengan buku.

[Repository publik](https://github.com/taraanrp/Grokking-Deep-Learning) · [Laporan validasi](reports/validation.md) · [Peta 215 subbab](reports/source_map.md) · [Catatan reproduksi/koreksi](reports/reproduction_notes.md)

## Referensi dan ketentuan tugas

PDF yang diberikan mahasiswa menjadi sumber utama. Identitas buku diperiksa pada PDF 4–5; daftar isi pada PDF 8–15; halaman isi menggunakan **nomor PDF = nomor cetak + 21**. Setiap tahap notebook menuliskan nomor yang relevan. [Kode pendamping resmi penulis](https://github.com/iamtrask/Grokking-Deep-Learning/tree/e665168b4aefe3256360951504103cf98827ca51) digunakan sebagai pembanding dan sumber data, bukan menggantikan pemeriksaan PDF. [Informasi penerbit Manning](https://www.manning.com/books/grokking-deep-learning).

Sesuai gambar instruksi dosen: satu notebook per bab, teori dan ringkasan, repository baru berjudul buku, README seluruh bab, serta penjelasan komprehensif untuk bagian tanpa kode.

| Cakupan | Tenggat yang tercantum |
|---|---|
| Bab 1–6 | 10 Oktober 2026 pukul 23.59 |
| Bab 7–10 | 17 Oktober 2026 pukul 23.59 |
| Bab 11–16 | Tidak tercantum; tidak diasumsikan |

Tanggal di atas mengikuti instruksi tugas. Log eksekusi memakai zona Asia/Jakarta. Repository ini tidak mengklaim bahwa pengumpulan telah dilakukan sebelum tenggat.

## Menjalankan notebook

Diuji menggunakan **Python 3.13.5**. Buat environment baru agar dependensi tidak bercampur dengan proyek lain:

```bash
git clone https://github.com/taraanrp/Grokking-Deep-Learning.git
cd Grokking-Deep-Learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name grokking-dl --display-name "Python 3 (Grokking DL)"
python scripts/download_data.py
jupyter lab
```

Pada Windows, aktivasi PowerShell: `.venv\Scripts\Activate.ps1`. Buka notebook di `notebooks/`, pilih kernel **Python 3 (Grokking DL)**, lalu **Restart Kernel → Run All**. Notebook mandiri: tidak perlu menjalankan bab sebelumnya atau mengandalkan bobot yang tersimpan. File model/data sementara tidak diperlukan untuk berpindah bab.

Untuk memvalidasi semuanya, setiap notebook mendapat kernel baru dan dijalankan sesuai urutan nomor:

```bash
python scripts/validate.py --mode quick
python scripts/render_reports.py
# Hanya rentang prioritas:
python scripts/validate.py --mode quick --chapters 7 8 9 10
python scripts/render_reports.py
```

`validate.py` menyimpan output langsung dalam notebook dan bukti JSON berisi status, metrik, durasi, versi, hash sumber, dan mode. `render_reports.py` membangun laporan dan analisis output dari bukti tersebut; ia menolak sumber notebook yang berubah setelah validasi. Jika FAIL, perbaiki penyebabnya, jalankan ulang bab terkait, lalu bangun ulang laporan.

Untuk anggaran penuh:

```bash
python scripts/validate.py --mode full --chapters 8 9 10
# Bab 11–15 dapat jauh lebih berat; jalankan terpisah:
python scripts/validate.py --mode full --chapters 11 12 13 14 15
python scripts/render_reports.py
# Untuk sesi interaktif penuh, mulai server dengan:
GDL_MODE=full jupyter lab
```

Mode full mempertahankan anggaran utama buku/pendamping tetapi tetap menerapkan penyesuaian yang dijelaskan. Bab 9–12 juga menyediakan cabang listing untuk membandingkan bug/perbedaan matematis. Bab 13 memakai penjadwalan graph yang direfaktor; ini tidak disebut transkripsi literal framework buku. Sediakan waktu dan RAM lebih untuk CNN, CBOW, LSTM512, dan enkripsi vocabulary penuh. Tidak ada klaim durasi full sebelum dijalankan.

## Isi seluruh bab

| Bab | Notebook | Gambaran | Validasi tersimpan |
|---:|---|---|---|
| 1 | [Introducing deep learning: why you should learn it](notebooks/chapter_01_introducing_deep_learning.ipynb) | Orientasi from scratch dan environment. | PASS — konsep / contoh kecil |
| 2 | [Fundamental concepts: how do machines learn?](notebooks/chapter_02_fundamental_concepts.ipynb) | Supervisi, parameter, predict–compare–learn. | PASS — konsep / contoh kecil |
| 3 | [Introduction to neural prediction: forward propagation](notebooks/chapter_03_forward_propagation.ipynb) | Weighted sum, banyak input/output, NumPy. | PASS — konsep / contoh kecil |
| 4 | [Introduction to neural learning: gradient descent](notebooks/chapter_04_gradient_descent.ipynb) | Hot/cold, turunan, alpha, divergence. | PASS — konsep / contoh kecil |
| 5 | [Learning multiple weights at a time: generalizing gradient descent](notebooks/chapter_05_learning_multiple_weights.ipynb) | Gradien vektor, pembekuan bobot, outer product. | PASS — konsep / contoh kecil |
| 6 | [Building your first deep neural network: introduction to backpropagation](notebooks/chapter_06_backpropagation.ipynb) | Streetlight, ReLU, backprop eksplisit. | PASS — konsep / contoh kecil |
| 7 | [How to picture neural networks: in your head and on paper](notebooks/chapter_07_picturing_neural_networks.ipynb) | Diagram ↔ bentuk tensor ↔ rumus ↔ kode. | PASS — konsep / contoh kecil |
| 8 | [Learning signal and ignoring noise: introduction to regularization and batching](notebooks/chapter_08_regularization_and_batching.ipynb) | MNIST, generalisasi, dropout, batch, early stopping. | PASS — quick |
| 9 | [Modeling probabilities and nonlinearities: activation functions](notebooks/chapter_09_activation_functions.ipynb) | Aktivasi, softmax, cross-entropy, MNIST. | PASS — quick |
| 10 | [Neural learning about edges and corners: intro to convolutional neural networks](notebooks/chapter_10_convolutional_neural_networks.ipynb) | Patch, shared kernel, CNN NumPy dan audit gradien. | PASS — quick |
| 11 | [Neural networks that understand language: king – man + woman == ?](notebooks/chapter_11_word_embeddings.ipynb) | Sentiment, embedding, CBOW, negative sampling, analogi. | PASS — quick |
| 12 | [Neural networks that write like Shakespeare: recurrent layers for variable-length data](notebooks/chapter_12_recurrent_layers.ipynb) | Bag-of-words ke RNN dan BPTT bAbI. | PASS — quick |
| 13 | [Introducing automatic optimization: let’s build a deep learning framework](notebooks/chapter_13_automatic_optimization.ipynb) | Tensor, autograd, optimizer, layer, embedding, RNN. | PASS — quick |
| 14 | [Learning to write like Shakespeare: long short-term memory](notebooks/chapter_14_long_short_term_memory.ipynb) | Character RNN/LSTM, TBPTT, sampling Shakespeare. | PASS — quick |
| 15 | [Deep learning on unseen data: introducing federated learning](notebooks/chapter_15_federated_learning.ipynb) | Spam, federated averaging, leakage, Paillier. | PASS — quick |
| 16 | [Where to go from here: a brief guide](notebooks/chapter_16_where_to_go_from_here.ipynb) | Sepuluh langkah belajar lanjutan dan presentasi. | PASS — konsep / contoh kecil |

## Struktur dan data

```text
Grokking-Deep-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/                 # 16 notebook, kode pembelajaran + teori + output
├── data/
│   ├── README.md              # asal, unduhan, pembagian dataset
│   └── manifest.json          # URL terpin + SHA-256 (dataset diabaikan Git)
├── src/
│   ├── datasets.py            # unduhan/cache/checksum, tanpa training tersembunyi
│   └── autograd.py            # framework Bab 13, ditampilkan lengkap di notebook13
├── scripts/                   # unduh, validasi kernel bersih, laporan
└── reports/
    ├── reproduction_notes.md
    ├── source_map.md
    ├── coverage.json
    ├── validation.md
    └── validation_results.json
```

Dataset: **MNIST** (Bab 8–10), **IMDB preprocessing pendamping** (11–12), **bAbI task1 v1.1** (12–13), **Shakespeare pendamping** (14), dan **Enron spam/ham pendamping** (15). Tautan unduhan individual, checksum, dan split ada di [data/README.md](data/README.md) serta [manifest](data/manifest.json). Sekitar 77 MB unduhan; diperlukan internet hanya ketika cache belum tersedia. Contoh olahraga/streetlight berasal dari buku dan ditulis langsung. Dataset tambahan untuk pengujian matematis selalu diberi label; tidak menggantikan dataset buku secara diam-diam.

PDF, hasil ekstraksi teks buku, kredensial, environment lokal, dan dataset besar **tidak masuk Git**. `src` dan `scripts` ditambahkan untuk menghindari duplikasi pemuatan data/framework dan memastikan validasi dapat diulang; forward/backprop inti tetap terlihat pada notebook.

## Cara mempelajari dan mempresentasikan

Mulai dari Bab 3–6 untuk menelusuri satu prediksi/update. Gunakan Bab 7 untuk menerjemahkan dimensi menjadi diagram. Pada Bab 8–10, bandingkan loss dan evaluasi, lalu tunjukkan satu gradient check. Bab 11–12 memperlihatkan akibat perubahan objective dan urutan kata; Bab 13–15 menunjukkan bagaimana abstraksi software dibangun dari operasi yang sama. Bab 16 memberikan alur presentasi dan pertanyaan pemahaman.

Baca label **reproduksi**, **koreksi/penyesuaian**, dan **tambahan**. Jangan menyebut train accuracy sebagai test accuracy, uji quick sebagai training full, atau sampel teks awal sebagai model yang telah menguasai Shakespeare. Laporan menampilkan hasil yang kurang baik juga, termasuk CNN singkat dan BPTT yang belum matang.

## Bantuan AI dan atribusi

**OpenAI Codex membantu secara substansial:** memeriksa PDF dan daftar isi, membandingkan kode pendamping, menulis implementasi dan penjelasan notebook, memperbaiki bug, menyiapkan dataset/environment, menjalankan notebook, menulis laporan, serta membuat/mengunggah repository berdasarkan instruksi pengguna. Bantuan bukan hanya penyuntingan bahasa. Mahasiswa memberikan identitas dan pilihan publik; tidak ada klaim bahwa mahasiswa telah memeriksa/menguasai seluruh hasil. Mahasiswa tetap perlu mempelajari dan mempertanggungjawabkan kode serta analisis saat presentasi.

Konsep/contoh sumber diatribusikan kepada Andrew W. Trask dan Manning. Penjelasan praktikum merupakan parafrasa dan interpretasi tambahan; dataset tetap mengikuti ketentuan pemiliknya. Tidak ada PDF atau salinan narasi buku lengkap dalam repository.
