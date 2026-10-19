# Data lokal dan asalnya

Data besar tidak masuk Git. Notebook mengambil data yang belum ada secara otomatis melalui `src/datasets.py` dan memeriksa SHA-256 setiap pemuatan. Untuk mengunduh lebih awal:

```bash
python scripts/download_data.py
# Atau satu kelompok:
python scripts/download_data.py mnist text babi shakespeare spam
```

Butuh internet saat unduhan pertama; setelah cache lengkap notebook dapat berjalan offline. Tidak ada fallback diam-diam ke data sintetis. `manifest.json` memuat URL terpin, jumlah byte, dan SHA-256. Jika checksum berbeda, eksekusi berhenti; periksa file lalu pindahkan/hapus file rusak secara sadar sebelum mengunduh ulang.

| Berkas | Byte | Sumber unduhan |
|---|---:|---|
| `reviews.txt` | 33,678,267 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/reviews.txt) |
| `labels.txt` | 225,000 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/labels.txt) |
| `shakespear.txt` | 99,993 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/shakespear.txt) |
| `spam.txt` | 10,843,548 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/spam.txt) |
| `ham.txt` | 21,180,948 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/ham.txt) |
| `qa1_single-supporting-fact_train.txt` | 94,346 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/tasksv11/en/qa1_single-supporting-fact_train.txt) |
| `qa1_single-supporting-fact_test.txt` | 94,477 | [unduh](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/e665168b4aefe3256360951504103cf98827ca51/tasksv11/en/qa1_single-supporting-fact_test.txt) |
| `mnist.npz` | 11,490,434 | [unduh](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) |

## Pemakaian

- MNIST: berkas yang disediakan untuk Keras, 60.000 train dan 10.000 test gambar 28×28. Digunakan Bab 8–10; NumPy memuat NPZ tanpa TensorFlow. Subset dan split dicetak pada notebook.
- IMDB/Udacity sentiment preprocessing: 25.000 ulasan dan label binary dalam snapshot pendamping penulis. Bab 11 memakai train24.000/test1.000 sebelum pembatasan quick. CBOW tanpa label menggunakan corpus pendamping penuh untuk vocabulary/distribusi negatif; quick mengurangi review yang di-update. Bab 12 membangun ulang embedding agar mandiri.
- bAbI task1 v1.1 bahasa Inggris: file train/test asli dari pendamping. Corpus memang sintetis sesuai buku; tidak dibuat ulang oleh AI. Tokenisasi mengikuti listing literal (termasuk tanda baca dan token berisi tab). Bab12 melakukan next-token modeling; Bab13 memprediksi token terakhir, bukan benchmark question answering penuh.
- `shakespear.txt`: teks sekitar100.000 karakter yang dikemas penulis; bukan seluruh karya Shakespeare. Bab14 quick train8.000 awal/test2.000 akhir. Modefull memakai seluruh corpus dan tidak melaporkan test held-out.
- Spam/ham: preprocessing Enron dari repository penulis, bukan pesan pribadi. Referensi asal pada buku: `www2.aueb.gr/users/ion/data/enron-spam/` (rujukan historis). Bab15 memisahkan1.000 akhir per kelas, kemudian memilih subsetquick dan membuang duplikat konten test terhadap train.

Dataset tunduk pada ketentuan sumber masing-masing. Repository ini hanya menyimpan petunjuk dan manifest, tidak melisensikan ulang datanya. Semua data yang diunduh sekitar77 MB; pemrosesan vocabulary dan array sementara memerlukan RAM lebih besar dari ukuran berkas. Data numerik kecil seperti olahraga/streetlight dicantumkan langsung pada notebook karena bagian esensial contoh buku.
