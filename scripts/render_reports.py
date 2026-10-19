"""Bangun README/laporan dan analisis notebook dari bukti hasil validator."""
from pathlib import Path
import json,hashlib,re
import nbformat
ROOT=Path(__file__).resolve().parents[1]
results=json.loads((ROOT/'reports/validation_results.json').read_text())
coverage=json.loads((ROOT/'reports/coverage.json').read_text())

def analysis(ch,m):
    if ch==1: return 'Pemeriksaan library dan operasi NumPy berhasil. Hasil ini membuktikan environment tersedia, belum menunjukkan pembelajaran model.'
    if ch==2: return f"Pada contoh sintetis tambahan, half squared error turun dari {m['loss_before']:.3f} menjadi {m['loss_after']:.3f}. Perubahan satu bobot mengurangi error satu titik; belum ada pengujian generalisasi."
    if ch==3: return f"Prediksi satu input {m['scalar_prediction']:.2f}; banyak input {m['multiple_input_prediction']:.2f}. Output jaringan bertingkat {m['stacked_output']} cocok antara loop dan NumPy. Kesamaan ini menguji operasi forward, bukan kualitas bobot."
    if ch==4: return f"Alpha besar menghasilkan squared error akhir {m['divergent_final_error']:.3e}, sedangkan alpha kecil menghasilkan {m['stable_final_error']:.3e}. Divergence adalah hasil yang sengaja didemonstrasikan, bukan kegagalan validasi. Selisih gradien numerik {m['gradient_check_abs_error']:.3e}."
    if ch==5: return f"Bobot yang dibekukan tetap {m['frozen_weight']}. Galat maksimum outer-product gradient check {m['gradient_max_error']:.3e}, mendukung penurunan rumus untuk semua koefisien yang diuji."
    if ch==6: return f"SSE epoch ke-60 {m['final_sse']:.8g}; akurasi empat titik latihan {m['training_accuracy']:.0%}. Nilai ini cocok dengan contoh kecil buku hingga toleransi numerik. Tidak ada test independen sehingga 100% tidak boleh disebut akurasi generalisasi."
    if ch==7: return f"Ekspresi bertahap dan ringkas sama. Jumlah parameter {m['parameter_count']} sesuai 3×4 + 4×1 tanpa bias. Prediksi berasal dari bobot demonstrasi acak."
    if ch==8: return f"Pada {m['test_count']} gambar test, baseline {m['baseline_test_accuracy']:.1%}, dropout koreksi {m['dropout_test_accuracy']:.1%}, dan mini-batch koreksi {m['batch_test_accuracy']:.1%}. Gap train-validation baseline {m['baseline_gap']:.1%}; validation memilih epoch {m['early_stopping_epoch']} dari {m['baseline_epochs']}. Perbandingan ini juga melibatkan hidden size/alpha/jumlah update berbeda, sehingga bukan ablation dropout yang terkontrol."
    if ch==9: return f"Setelah {m['epochs']} epoch, akurasi test listing {m['book_test_accuracy']:.1%}, koreksi {m['corrected_test_accuracy']:.1%} pada {m['test_count']} contoh. Galat gradien koreksi {m['gradient_max_error']:.3e}. Hasil mendukung konsistensi gradien koreksi, bukan klaim semua perubahan selalu meningkatkan akurasi."
    if ch==10: return f"Pada {m['train_count']} train/{m['test_count']} test selama {m['epochs']} epoch, listing mencapai {m['book_test_accuracy']:.1%}, koreksi {m['corrected_test_accuracy']:.1%}. Hasil koreksi masih rendah: run ini memeriksa alur CNN, bukan model yang sudah matang. Galat gradien kernel {m['kernel_gradient_max_error']:.3e}."
    if ch==11: return f"Sentiment: listing {m['book_sentiment_accuracy']:.1%} versus koreksi {m['corrected_sentiment_accuracy']:.1%} pada {m['sentiment_test_count']} test, dengan {m['sentiment_train_count']} train. Koreksi derivative justru memberi akurasi lebih rendah pada anggaran ini: faktor sigmoid mengecilkan langkah, sedangkan alpha tidak dituning ulang. Benar secara kalkulus tidak otomatis lebih baik pada hyperparameter yang sama. CBOW melakukan {m['cbow_updates']:,} update; BCE sampling akhir {m['cbow_final_bce']:.4f} bukan perplexity vocabulary penuh."
    if ch==12: return f"Gradien BPTT koreksi memiliki galat {m['gradient_max_error']:.3e}; indeks listing {m['book_gradient_max_error']:.3e}. Setelah {m['steps']} update, test perplexity {m['test_perplexity']:.3f} dengan {m['test_oov']} token OOV diabaikan. Angka masih tinggi; latihan singkat belum mempelajari pola bahasa dengan baik."
    if ch==13: return f"Autograd lolos gradient check (galat {m['autograd_gradient_max_error']:.3e}) dan uji multiuse/indexing. RNN dilatih {m['rnn_steps']} langkah, loss akhir {m['rnn_final_loss']:.4f}, akurasi training {m['rnn_train_accuracy']:.1%}; ini belum mengukur generalisasi."
    if ch==14:
        result=f"RNN {m['rnn_epochs']} epoch dan LSTM {m['lstm_epochs']} epoch memakai {m['train_characters']:,} karakter training. NLL akhir masing-masing {m['rnn_final_nll']:.4f} dan {m['lstm_final_nll']:.4f}. "
        if m['rnn_test_perplexity'] is not None: result+=f"Held-out perplexity RNN {m['rnn_test_perplexity']:.3f}, LSTM {m['lstm_test_perplexity']:.3f}. LSTM tidak lebih baik pada run ini; arsitektur lebih kompleks memerlukan anggaran/tuning yang sesuai. "
        else: result+='Mode ini memakai seluruh corpus untuk training, tanpa test held-out. '
        return result+f"Galat gradient check gerbang {m['lstm_gradient_max_error']:.3e}; kualitas teks harus dibaca pada sampel aktual."
    if ch==15: return f"Dengan {m['train_count']} baris train dan {m['test_count']} test, akurasi centralized {m['centralized_test_accuracy']:.2%}, federated {m['federated_test_accuracy']:.2%}, dan agregasi terenkripsi {m['secure_test_accuracy']:.2%}. Sebanyak {m['test_duplicates_removed']} duplikat test terhadap train dibuang. Galat maksimum decrypt-average {m['aggregate_max_error']:.3e}; kesamaan aritmetika ini tidak membuktikan protokol aman untuk produksi. Demonstrasi satu email mengungkap {m['revealed_word_count']} kata."
    return f"Terdapat {m['notebook_count']} notebook. Pemeriksaan ini hanya memvalidasi struktur; bukti eksekusi setiap bab tersedia dalam laporan validator."

for ch in coverage:
    n=ch['chapter']; result=results[str(n)]; path=ROOT/'notebooks'/ch['notebook']
    book=nbformat.read(path,as_version=4)
    book.cells=[c for c in book.cells if not c.metadata.get('generated_analysis')]
    digest=hashlib.sha256(json.dumps([(c.cell_type,c.source) for c in book.cells],ensure_ascii=False).encode()).hexdigest()
    if result['status']=='PASS':
        if digest!=result['source_sha256']: raise ValueError(f'Bab {n} berubah setelah divalidasi')
        msg=f"## Analisis output aktual — run tersimpan\n\nMode **{result['mode']}**, eksekusi {result['started_at']}. Ringkasan ini dibuat dari output validator, bukan angka contoh dalam buku.\n\n"+analysis(n,result['metrics'])
        cell=nbformat.v4.new_markdown_cell(msg,metadata={'generated_analysis':True}); cell.id=f'ch{n:02}-actual-analysis'
        index=next(i for i,c in enumerate(book.cells) if c.cell_type=='markdown' and c.source.startswith('## Kesimpulan'))
        book.cells.insert(index,cell); nbformat.write(book,path)

summary=[
'Orientasi from scratch dan environment.', 'Supervisi, parameter, predict–compare–learn.',
'Weighted sum, banyak input/output, NumPy.', 'Hot/cold, turunan, alpha, divergence.',
'Gradien vektor, pembekuan bobot, outer product.', 'Streetlight, ReLU, backprop eksplisit.',
'Diagram ↔ bentuk tensor ↔ rumus ↔ kode.', 'MNIST, generalisasi, dropout, batch, early stopping.',
'Aktivasi, softmax, cross-entropy, MNIST.', 'Patch, shared kernel, CNN NumPy dan audit gradien.',
'Sentiment, embedding, CBOW, negative sampling, analogi.', 'Bag-of-words ke RNN dan BPTT bAbI.',
'Tensor, autograd, optimizer, layer, embedding, RNN.', 'Character RNN/LSTM, TBPTT, sampling Shakespeare.',
'Spam, federated averaging, leakage, Paillier.', 'Sepuluh langkah belajar lanjutan dan presentasi.']
pass_count=sum(r['status']=='PASS' for r in results.values())
full_unverified=[n for n in range(8,16) if results[str(n)]['mode']!='full' or results[str(n)]['status']!='PASS']
full_status='Bab '+', '.join(map(str,full_unverified))+' belum divalidasi dengan mode full.' if full_unverified else 'Bab 8–15 sudah divalidasi mode full; periksa hasil dan batas di laporan.'
readme=f'''# Grokking-Deep-Learning

Tugas individu **Code Reproduction + Theoretical Deep-Dive** berdasarkan *Grokking Deep Learning* — Andrew W. Trask, Manning, ©2019. Repository berisi **16 Jupyter Notebook**, penjelasan orisinal bahasa Indonesia, implementasi NumPy from scratch, output nyata, eksperimen berlabel, dan peta sumber PDF.

| Identitas | Nilai |
|---|---|
| Nama | Tarannia Raisha Putri |
| NIM | 101032300231 |
| Kelas | TK4702 |
| Mata kuliah | Deep Learning |

**Hasil validasi: {pass_count}/16 notebook PASS** dari kernel bersih secara berurutan. Bab 1–7 dan16 memuat konsep/contoh kecil lengkap; mode setiap bab tercantum dalam tabel. **{full_status}** PASS menyatakan sel dan pemeriksaan berhasil; bukan kesamaan semua akurasi dengan buku.

[Repository publik](https://github.com/taraanrp/Grokking-Deep-Learning) · [Laporan validasi](reports/validation.md) · [Peta215 subbab](reports/source_map.md) · [Catatan reproduksi/koreksi](reports/reproduction_notes.md)

## Referensi dan ketentuan tugas

PDF yang diberikan mahasiswa menjadi sumber utama. Identitas buku diperiksa pada PDF4–5; daftar isi pada PDF8–15; halaman isi menggunakan **nomor PDF = nomor cetak +21**. Setiap tahap notebook menuliskan nomor yang relevan. [Kode pendamping resmi penulis](https://github.com/iamtrask/Grokking-Deep-Learning/tree/e665168b4aefe3256360951504103cf98827ca51) digunakan sebagai pembanding dan sumber data, bukan menggantikan pemeriksaan PDF. [Informasi penerbit Manning](https://www.manning.com/books/grokking-deep-learning).

Sesuai gambar instruksi dosen: satu notebook per bab, teori dan ringkasan, repository baru berjudul buku, README seluruh bab, serta penjelasan komprehensif untuk bagian tanpa kode.

| Cakupan | Tenggat yang tercantum |
|---|---|
| Bab1–6 | 10 Oktober2026 pukul23.59 |
| Bab7–10 | 17 Oktober2026 pukul23.59 |
| Bab11–16 | Tidak tercantum; tidak diasumsikan |

Tanggal di atas mengikuti instruksi tugas. Log eksekusi memakai zona Asia/Jakarta. Repository ini tidak mengklaim bahwa pengumpulan telah dilakukan sebelum tenggat.

## Menjalankan notebook

Diuji menggunakan **Python3.13.5**. Buat environment baru agar dependensi tidak bercampur dengan proyek lain:

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

Pada Windows, aktivasi PowerShell: `.venv\\Scripts\\Activate.ps1`. Buka notebook di `notebooks/`, pilih kernel **Python3 (Grokking DL)**, lalu **Restart Kernel → Run All**. Notebook mandiri: tidak perlu menjalankan bab sebelumnya atau mengandalkan bobot yang tersimpan. File model/data sementara tidak diperlukan untuk berpindah bab.

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
# Bab11–15 dapat jauh lebih berat; jalankan terpisah:
python scripts/validate.py --mode full --chapters 11 12 13 14 15
python scripts/render_reports.py
# Untuk sesi interaktif penuh, mulai server dengan:
GDL_MODE=full jupyter lab
```

Modefull mempertahankan anggaran utama buku/pendamping tetapi tetap menerapkan penyesuaian yang dijelaskan. Bab9–12 juga menyediakan cabang listing untuk membandingkan bug/perbedaan matematis. Bab13 memakai penjadwalan graph yang direfaktor; ini tidak disebut transkripsi literal framework buku. Sediakan waktu dan RAM lebih untuk CNN, CBOW, LSTM512, dan enkripsi vocabulary penuh. Tidak ada klaim durasi full sebelum dijalankan.

## Isi seluruh bab

| Bab | Notebook | Gambaran | Validasi tersimpan |
|---:|---|---|---|
'''
for ch,s in zip(coverage,summary):
    n=ch['chapter']; result=results[str(n)]; scope='konsep / contoh kecil' if n<=7 or n==16 else result['mode']
    readme+=f'| {n} | [{ch["title"]}](notebooks/{ch["notebook"]}) | {s} | {result["status"]} — {scope} |\n'
readme+='''
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
│   └── autograd.py            # framework Bab13, ditampilkan lengkap di notebook13
├── scripts/                   # unduh, validasi kernel bersih, laporan
└── reports/
    ├── reproduction_notes.md
    ├── source_map.md
    ├── coverage.json
    ├── validation.md
    └── validation_results.json
```

Dataset: **MNIST** (Bab8–10), **IMDB preprocessing pendamping** (11–12), **bAbI task1 v1.1** (12–13), **Shakespeare pendamping** (14), dan **Enron spam/ham pendamping** (15). Tautan unduhan individual, checksum, dan split ada di [data/README.md](data/README.md) serta [manifest](data/manifest.json). Sekitar77 MB unduhan; diperlukan internet hanya ketika cache belum tersedia. Contoh olahraga/streetlight berasal dari buku dan ditulis langsung. Dataset tambahan untuk pengujian matematis selalu diberi label; tidak menggantikan dataset buku secara diam-diam.

PDF, hasil ekstraksi teks buku, kredensial, environment lokal, dan dataset besar **tidak masuk Git**. `src` dan `scripts` ditambahkan untuk menghindari duplikasi pemuatan data/framework dan memastikan validasi dapat diulang; forward/backprop inti tetap terlihat pada notebook.

## Cara mempelajari dan mempresentasikan

Mulai dari Bab3–6 untuk menelusuri satu prediksi/update. Gunakan Bab7 untuk menerjemahkan dimensi menjadi diagram. Pada Bab8–10, bandingkan loss dan evaluasi, lalu tunjukkan satu gradient check. Bab11–12 memperlihatkan akibat perubahan objective dan urutan kata; Bab13–15 menunjukkan bagaimana abstraksi software dibangun dari operasi yang sama. Bab16 memberikan alur presentasi dan pertanyaan pemahaman.

Baca label **reproduksi**, **koreksi/penyesuaian**, dan **tambahan**. Jangan menyebut train accuracy sebagai test accuracy, ujiquick sebagai trainingfull, atau sampel teks awal sebagai model yang telah menguasai Shakespeare. Laporan menampilkan hasil yang kurang baik juga, termasuk CNN singkat dan BPTT yang belum matang.

## Bantuan AI dan atribusi

**OpenAI Codex membantu secara substansial:** memeriksa PDF dan daftar isi, membandingkan kode pendamping, menulis implementasi dan penjelasan notebook, memperbaiki bug, menyiapkan dataset/environment, menjalankan notebook, menulis laporan, serta membuat/mengunggah repository berdasarkan instruksi pengguna. Bantuan bukan hanya penyuntingan bahasa. Mahasiswa memberikan identitas dan pilihan publik; tidak ada klaim bahwa mahasiswa telah memeriksa/menguasai seluruh hasil. Mahasiswa tetap perlu mempelajari dan mempertanggungjawabkan kode serta analisis saat presentasi.

Konsep/contoh sumber diatribusikan kepada Andrew W. Trask dan Manning. Penjelasan praktikum merupakan parafrasa dan interpretasi tambahan; dataset tetap mengikuti ketentuan pemiliknya. Tidak ada PDF atau salinan narasi buku lengkap dalam repository.
'''
# Perbaikan spasi pada label manusia; jangan mengubah kode/URL.
for a,b in [('Bab1','Bab 1'),('Bab2','Bab 2'),('Bab3','Bab 3'),('Bab4','Bab 4'),('Bab5','Bab 5'),('Bab6','Bab 6'),('Bab7','Bab 7'),('Bab8','Bab 8'),('Bab9','Bab 9'),('Bab10','Bab 10'),('Bab11','Bab 11'),('Bab12','Bab 12'),('Bab13','Bab 13'),('Bab14','Bab 14'),('Bab15','Bab 15'),('Bab16','Bab 16'),('Python3','Python 3'),('Oktober2026','Oktober 2026'),('pukul23.59','pukul 23.59'),('Peta215','Peta 215'),('PDF4','PDF 4'),('PDF8','PDF 8'),('cetak +21','cetak + 21'),('Modefull','Mode full'),('Bab 8–15','Bab 8–15'),('sekitar77','sekitar 77'),('Sekitar77','Sekitar 77'),('trainingfull','training full'),('ujiquick','uji quick')]:
    readme=readme.replace(a,b)
(ROOT/'README.md').write_text(readme)

lines=['# Laporan validasi aktual','',f'**{pass_count}/16 PASS.** Eksekusi berurutan, satu kernel baru per notebook. Mode setiap run tercantum dalam tabel. {full_status}','',
'Bukti mesin: [validation_results.json](validation_results.json). Output stream dan gambar tersimpan di masing-masing notebook. Hash source mencakup source seluruh sel kecuali ringkasan analisis yang dibangkitkan dari hasil; dependency/support hashes mengikat loader, framework, requirement, dan manifest data.',
'', '| Bab | Status | Mode | Sel kode | Detik | Mulai (Asia/Jakarta) |','|---:|---|---|---:|---:|---|']
for ch in coverage:
    r=results[str(ch['chapter'])]
    lines.append(f"| {ch['chapter']} | {r['status']} | {r['mode']} | {r.get('code_cells','—')} | {r['seconds']:.3f} | {r['started_at']} |")
lines.extend(['',f"Total waktu eksekusi notebook pada run yang tersimpan: **{sum(r['seconds'] for r in results.values()):.3f} detik**. Tidak termasuk pengembangan, unduhan, instalasi, atau run yang gagal sebelumnya; waktu ini bukan perkiraan untuk komputer lain.",'',
'## Lingkungan dan prosedur','',f"Python `{results['1']['python']}`; `{results['1']['platform']}`. Dependensi langsung: "+', '.join(f"`{k}=={v}`" for k,v in results['1']['dependencies'].items())+'.',
'', 'Validator menetapkan thread BLAS/OpenMP ke1 agar run kecil stabil, menolak unexpected cell errors, memeriksa urutan execution_count, dan mengekstrak metrik JSON dari output. Assertion pada notebook memeriksa dimensi, finite values, normalisasi probabilitas, contoh yang seharusnya konvergen, gradien numerik, multiuse graph, repeated indexing, pembagian data, dan dekripsi agregat. Model yang memang dimaksudkan gagal mempunyai assertion khusus; divergence finite bukan NaN/inf.',
'', '## Hasil dan interpretasi per bab',''])
for ch in coverage:
    n=ch['chapter']; r=results[str(n)]
    lines.extend([f"### Bab {n} — {ch['title']}",'',analysis(n,r['metrics']) if r['status']=='PASS' else 'FAIL: '+r.get('error','tanpa detail'),''])
lines.extend(['## Konfigurasi yang benar-benar diuji versus penuh','',
'| Bab | Quick tersimpan | Full tersedia, belum dijalankan pada pembuatan awal |','|---|---|---|',
'| 1–7,16 | Semua konsep/contoh kecil; Bab6: 60 epoch | Sama; tidak ada training besar baru |',
'| 8 | 1.000 train,200 validation,500 test; baseline30,dropout20 per cabang,batch40 per cabang epoch | Baseline350,dropout300,batch300; test10.000 |',
'| 9 | 1.000 train/500 test,30 epoch,dua cabang | 300 epoch/test10.000 |',
'| 10 | 256 train/200 test,4 epoch,kernel3×3×16,batch128,dua cabang | 1.000 train/10.000 test,300 epoch |',
'| 11 | Sentiment1.200 train/1.000 test,2 epoch,H100. CBOW200 review×maks80 token×2,dua cabang | Sentiment24.000 train; CBOW25.000 review penuh×2 |',
'| 12 | CBOW ringan mandiri; bAbI1.000 baris,800 update,H10,dua cabang | CBOW penuh;30.000 update BPTT |',
'| 13 | Semua primitive; RNN80 update,H16,batch100 | RNN1.000 update |',
'| 14 | 8.000 karakter,H32,batch8,RNN2/LSTM3 epoch; held-out2.000 akhir | Seluruh corpus,H512,RNN100 epoch B32/T16,LSTM110 epoch B16/T25; tanpa held-out |',
'| 15 | 600 train,199 test setelah dedup,V1.003,batch64,3 ronde | Semua train setelah filter,semua test setelah dedup,V training penuh,batch500,3 ronde |','',
'“Full” berarti anggaran latihan penuh teradaptasi, bukan jaminan hasil identik. Lihat reproduction_notes.md untuk perubahan matematika/normalisasi/split yang tetap berlaku. Tidak ada output full yang dibuat-buat.',
'', '## Batas interpretasi dan temuan validasi','',
'- Bab6 dan13 mengukur training pada data kecil; tidak ada klaim generalisasi dari skor tersebut.',
'- Test MNIST quick adalah subset deterministik test resmi. Validation Bab8 berasal dari train resmi di luar1.000 contoh latihan. Tidak dipakai untuk memilih parameter berdasarkan test.',
'- Perbandingan dropout bukan ablation terkontrol: hidden size,alpha,dan jumlah update ikut berbeda mengikuti tahap buku.',
'- CBOW vocabulary/distribusi negatif menggunakan corpus tak berlabel seperti buku. Jangan melaporkan skor sentiment sebagai evaluasi CBOW.',
'- bAbI tokenisasi literal masih menyimpan tab/tanda baca dan evaluasi mengabaikan OOV yang dihitung. Ini bukan skor QA resmi.',
'- Spam: satu duplikat konten lintas train/test ditemukan pada subsetquick dan dihapus. Test199 setelah koreksi. Tidak ada raw email pribadi atau key enkripsi dipublikasikan.',
'- Uji enkripsi membuktikan kesetaraan agregasi numerik, tidak membuktikan keamanan arsitektur satu-proses/128-bit.',
'- Run awal menemukan kesalahan string escape pada notebook4,5,7 dan overlap data notebook15. Semua diperbaiki; tabel di atas berasal dari run akhir, bukan run gagal.',
'- Diagram dan kurva dihasilkan oleh Matplotlib dari output aktual; sumbu,legend,dan kesimpulan diperiksa terhadap angka. Grafik tidak mengimpor gambar/output buku.',
'- Tidak ada bab berstatus belum dijalankan dalam runquick. **Trainingfull Bab8–15 tetap belum diverifikasi** dan tidak menjadi syarat agar notebookquick selesai.',
'', '## Mengulang','', '```bash','python scripts/validate.py --mode quick','python scripts/render_reports.py','```','',
'Perubahan source setelah run akan ditolak oleh renderer sampai notebook terkait divalidasi ulang. JSON disimpan per bab sehingga re-run sebagian memperbarui bab tersebut saja. Jangan menggabungkan metrik dari mode berbeda tanpa membaca kolom mode setiap bab.'])
text='\n'.join(lines)+'\n'
text=re.sub(r'\bBab(?=\d)','Bab ',text)
(ROOT/'reports/validation.md').write_text(text)
print('Laporan dan analisis diperbarui dari',pass_count,'hasil PASS.')
