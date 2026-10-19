# Catatan reproduksi

## Sumber dan atribusi

Referensi utama: Andrew W. Trask, **Grokking Deep Learning**, Manning Publications, ©2019 (halaman copyright PDF 5). PDF pengguna: 335 halaman; SHA-256 `00e470992446b343f37020fcc5deac24f38e4e762cc7e259bfe173b65fd221a5`. PDF hanya dibaca lokal dan tidak dimasukkan ke Git.

[Halaman penerbit](https://www.manning.com/books/grokking-deep-learning) dan [repository pendamping penulis](https://github.com/iamtrask/Grokking-Deep-Learning/tree/e665168b4aefe3256360951504103cf98827ca51). Snapshot pendamping dikunci ke commit `e665168b4aefe3256360951504103cf98827ca51`. Teori diparafrasekan dalam bahasa Indonesia; contoh algoritme dan data kecil diatribusikan ke bab/halaman. Repository ini tidak mengklaim kepemilikan atas buku atau dataset, dan tidak memberikan lisensi ulang terhadap materi sumber.

Instruksi pengumpulan diperiksa dari gambar yang diberikan pengguna: tugas individu, satu notebook per bab, ringkasan teori, README seluruh bab, repository berjudul buku, dan integritas akademik. Gambar tidak menyatakan tenggat Bab 11–16. Identitas mahasiswa diberikan langsung oleh pengguna.

Lihat [source_map.md](source_map.md) untuk seluruh subbab, nomor cetak/PDF, dan tautan tahap notebook; [coverage.json](coverage.json) memuat indeks sel berbasis nol dan anchor tahap. Notebook yang tidak memiliki listing training baru (1, 2, 7, 16) berisi penjelasan konsep; kode tambahan dilabeli.

## Arti label

- **Reproduksi:** contoh, algoritme, data, dan urutan gagasan buku ditulis ulang; bukan transkripsi verbatim seluruh listing.
- **Penyesuaian/koreksi:** perbedaan API, determinisme, matematika, data, atau struktur implementasi diungkapkan.
- **Tambahan:** gradient checking, ilustrasi, pembanding, atau evaluasi yang dibuat untuk praktikum.
- **Quick:** validasi nyata dengan anggaran terbatas. **Full:** konfigurasi anggaran buku/pendamping dengan koreksi yang tetap berlaku; belum dijalankan untuk Bab 8–15. Full tidak berarti bit-for-bit sama dengan keluaran cetak.

## Kompatibilitas umum

Python 2.7 pada buku diganti Python 3.13 yang benar-benar diuji. NumPy tetap fondasi; tidak ada TensorFlow/PyTorch yang melatih model. Keras hanya digantikan loader NPZ MNIST melalui urllib + NumPy. Berkas tersebut merupakan MNIST yang sama, bukan digits scikit-learn. Softmax menggunakan pengurangan max dan CE memakai log-softmax/logaddexp; stabilisasi ini mempertahankan fungsi secara matematis. Vocabulary/set diurutkan, seed lokal dipakai, dan output loop panjang diringkas tanpa mengklaim jumlah update yang berbeda. Urutan set asli yang tidak deterministik menghalangi kecocokan numerik persis dengan buku.

Semua gradien yang memerlukan bobot forward dihitung sebelum bobot diubah. Array integer/float dibuat eksplisit dan nilai scalar diambil dengan `.item()` jika diperlukan. Relative path dihitung dari root/notebooks. Data unduhan diperiksa SHA-256. Error dimensi yang sengaja diajarkan ditangkap sebagai expected error, sementara error tak terduga menggagalkan validator.

## Perubahan per bab dan dampaknya

| Bab / halaman cetak (PDF) | Temuan dan keputusan | Dampak |
|---|---|---|
| 1–2 / 3–19 (24–40) | Pemeriksaan environment dan satu langkah sintetis adalah tambahan. Definisi nonparametrik diperjelas, tidak sekadar “tanpa parameter”. | Tidak diklaim sebagai listing buku. |
| 3 / 35–45 (56–66) | Konvensi bobot rows-per-output dibandingkan dengan transpose NumPy; loop digeneralisasi untuk dimensi tidak persegi. | Prediksi contoh tetap cocok; error shape tetap ditunjukkan. |
| 4 / 56–76 (77–97) | Update buku sesuai half squared error, sementara error yang dicetak squared error. | Faktor 2 dijelaskan; gradient check memakai objective yang konsisten. |
| 5 / 80–93 (101–114) | Pembentukan `weight_deltas` yang hilang di sel awal upstream ditulis eksplisit; semua tahap digabung tanpa menghilangkan pembekuan/output/outer product. | Memulihkan notebook mandiri. |
| 6 / 125–130 (146–151) | Init seed1, hidden4, alpha0.2, 60 epoch dipertahankan. Least squares dan finite differences tambahan. | SSE latihan dapat dibandingkan langsung dengan listing kecil. |
| 7 / 133–143 (154–164) | Gambar baru dibuat dari dimensi; bobot ilustrasi tambahan. | Tidak menyalin diagram atau mengklaim prediksi visual sebagai training. |
| 8 / 147 (168) | NPZ loader menggantikan Keras; 1.000 training tetap. Validation train[1000:1200] tambahan; test tidak untuk memilih epoch. | Menghindari kebocoran pemilihan epoch. |
| 8 / 155–159 (176–180) | Dropout forward 2M tetapi backward listing M. Kedua versi disediakan. Turunan ReLU koreksi memilih 0 tepat di nol. PDF update batch di luar loop akurasi; upstream salah indentasi. | Versi koreksi mengubah gradien; baseline tetap tanpa dropout. |
| 8 / 158–160 (179–181) | Batch listing alpha0.001, koreksi/eksperimen alpha0.1. Skor training dihitung dengan bobot akhir epoch, bukan running online accuracy. | Metrik/kurva tidak identik dengan logging buku. |
| 9 / 175 (196) | Listing memakai B², slope tanh sesudah dropout, dan masker tanpa faktor2. Cabang listing dipertahankan; koreksi memakai B, h sebelum dropout, 2M. Alpha koreksi0.02 vs listing2. | Perubahan matematis, bukan sekadar API. |
| 10 / 181–184 (202–205) | Typo sub_section diperbaiki. Cabang listing memakai 25×25 lokasi, B², post-dropout derivative, tanda update kernel berlawanan. Koreksi memakai 26×26, derivative konsisten, update minus untuk kedua bobot, batch sisa ikut dilatih. | Jumlah classifier weights berubah; hasil tidak identik. Gradient check menguji cabang koreksi. |
| 11 / 191–195 (212–216) | Berkas pendamping 25.000 binary reviews; train24.000/test1.000. Dictionary training-only, sorted, whitespace bersih. Listing melewatkan sigmoid derivative hidden; kedua cabang dilatih. | Kosa kata/gradien berbeda dari listing; OOV dihitung. |
| 11 / 201–207 (222–228) | CBOW listing window kanan hanya1; koreksi2. Mean embedding memerlukan pembagi konteks; duplicate indices scatter-add. Analogi listing mengalikan squared norms, koreksi memakai cosine dengan L2. | Ada cabang listing untuk audit; analogi tidak dipaksa menghasilkan jawaban tertentu. |
| 12 / 212,220–227 (233,241–248) | Embedding yang undefined di upstream dibangun ulang secara mandiri. Sox mengikuti PDF. Indeks BPTT listing bergeser; cabang koreksi diuji finite difference. Listing membuang token pertama lagi, koreksi mempertahankannya. | Keluaran tidak identik; test bAbI tambahan mengabaikan OOV dan melaporkan jumlahnya. |
| 13 / 233–263 (254–284) | Semua operasi dan tahap utama hadir, tetapi child-ID/count scheduling diganti topological reverse traversal. Gradien ndarray, constants diizinkan, mutable default list dihindari. | Refaktor struktural yang eksplisit, bukan salinan literal Tensor buku. |
| 13 / 251,258–260 (272,279–281) | MSE benar-benar mean; CE backward dibagi B sesuai mean forward. Indexing scatter-add, typo penetapan Tanh dibetulkan. | Learning-rate scale berubah dibanding summed/unnormalized update buku. |
| 14 / 268–278 (289–299) | RNN listing menimpa total loss lewat loop; koreksi backward jumlah semua timestep. Dua state LSTM didetach. Decoder nol dan init komponen pertama1 mengikuti PDF. Sampling categorical diperbaiki. | Log NLL/perplexity memakai mean token yang jelas, bukan variabel loss yang tertimpa. |
| 14 / 266–280 (287–301) | Quick 8.000 karakter train, 2.000 akhir test; full memakai seluruh corpus sesuai sumber, tanpa klaim held-out test. | Hasil quick bukan bukti kemampuan model512 penuh. |
| 15 / 283–285 (304–306) | Listing membuang email ≥500 token; filter dipertahankan dan dihitung. Whitespace cleaning mengganti slicing dua karakter. `bs` undefined diganti batch_size; optimizer terikat ke model lokal; batch sisa dipakai. | Dapat berjalan dari kernel bersih; padding dibekukan0. |
| 15 / 284–286 (305–307) | Vocabulary training-only; quick dibatasi1.000 kata + lima istilah demonstrasi. Test duplikat konten terhadap train dibuang, tidak di-oversample. | Skor berbeda dari pembagian buku, mengurangi kebocoran evaluasi. |
| 15 / 290 (311) | Dekripsi agregat ketiga klien menggantikan Sue saja; parameter pubkey dipakai. Paillier128-bit hanya demonstrasi. | Memperbaiki rata-rata, tidak mengklaim protokol privasi produksi. |
| 16 / 294–299 (315–320) | Sepuluh langkah diparafrasekan sesuai konteks buku; audit repository dan presentasi tambahan. | Tidak mengarang listing, tanggal, atau rekomendasi teknologi terkini. |

## Penggunaan AI dan tanggung jawab mahasiswa

OpenAI Codex membantu membaca PDF lokal/daftar isi, membandingkan kode pendamping, menyusun seluruh notebook dan penjelasan bahasa Indonesia, mengimplementasikan serta memperbaiki kode, mengunduh data, menjalankan validasi, menulis laporan, dan memublikasikan repository atas instruksi pengguna. Kontribusi AI mencakup kode dan teori secara substansial; tidak hanya perapian bahasa. Identitas dan pilihan publik diberikan oleh mahasiswa. Repository tidak menyatakan bahwa mahasiswa sudah meninjau atau menguasai seluruh hasil. Mahasiswa perlu mempelajari, mengoreksi bila perlu, dan mampu mempertanggungjawabkan tiap contoh saat presentasi.

## Batas yang masih ada

Tidak ada notebook yang sengaja dibiarkan tanpa implementasi. Semua tahapan kode substantif dipetakan; beberapa listing berulang disatukan dan beberapa bug matematis dijalankan hanya pada cabang pembanding. Pelatihan penuh Bab 8–15 belum diverifikasi. Skor quick tidak menggantikan skor buku, dan contoh kecil tanpa test tidak diklaim sebagai generalisasi. Full mungkin memerlukan waktu/memori besar, terutama CNN, CBOW, LSTM512 dan enkripsi vocabulary penuh. Semua hasil dan batas angka ada di validation.md.
