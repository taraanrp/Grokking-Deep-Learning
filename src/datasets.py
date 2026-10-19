"""Dataset asli buku: pengambilan terpin, cache lokal, pemeriksaan SHA-256."""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

def fetch(name):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    info = manifest[name]
    path = DATA / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(info['url'], timeout=120) as response:
            temporary.write_bytes(response.read())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != info['sha256']:
            temporary.unlink()
            raise ValueError(f'Checksum gagal: {name}')
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError(f'Cache {name} berubah; pindahkan file ini lalu unduh ulang.')
    return path

def mnist():
    with np.load(fetch('mnist.npz'), allow_pickle=False) as d:
        arrays = tuple(d[k].copy() for k in ('x_train','y_train','x_test','y_test'))
    assert arrays[0].shape == (60000,28,28)
    assert arrays[2].shape == (10000,28,28)
    return arrays

def reviews():
    texts = fetch('reviews.txt').read_text().splitlines()
    labels = np.array([s == 'positive' for s in fetch('labels.txt').read_text().splitlines()], dtype=int)
    assert len(texts) == len(labels) == 25000
    return texts, labels

def babi(split='train', limit=1000):
    # Tokenisasi literal listing: ID baris dibuang; tab pada pertanyaan tetap ada.
    lines = fetch(f'qa1_single-supporting-fact_{split}.txt').read_text().splitlines()
    return [line.lower().split(' ')[1:] for line in lines[:limit]]
