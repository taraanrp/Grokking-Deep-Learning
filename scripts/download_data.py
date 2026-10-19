"""Unduh semua data, atau satu kelompok: mnist, text, babi, shakespeare, spam."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import fetch
GROUPS = {'mnist':['mnist.npz'], 'text':['reviews.txt','labels.txt'],
          'babi':['qa1_single-supporting-fact_train.txt','qa1_single-supporting-fact_test.txt'],
          'shakespeare':['shakespear.txt'], 'spam':['spam.txt','ham.txt']}
for group in sys.argv[1:] or GROUPS:
    for name in GROUPS[group]:
        p=fetch(name)
        print(f'{name}: checksum OK, {p.stat().st_size:,} byte')
