"""Jalankan notebook berurutan, masing-masing dengan kernel bersih, simpan bukti aktual."""
from pathlib import Path
import argparse,json,time,platform,os,hashlib,importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--chapters', nargs='*',type=int,default=list(range(1,17)))
parser.add_argument('--mode',choices=['quick','full'],default='quick')
args=parser.parse_args()
os.environ['GDL_MODE']=args.mode
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
report=ROOT/'reports'/'validation_results.json'
results=json.loads(report.read_text()) if report.exists() else {}
failed=False
for chapter in args.chapters:
    path=next((ROOT/'notebooks').glob(f'chapter_{chapter:02}_*.ipynb'))
    notebook=nbformat.read(path,as_version=4)
    notebook.cells=[c for c in notebook.cells if not c.metadata.get('generated_analysis')]
    for cell in notebook.cells:
        if cell.cell_type=='code': cell.outputs=[]; cell.execution_count=None
    start=time.perf_counter()
    source_digest=hashlib.sha256(json.dumps([(c.cell_type,c.source) for c in notebook.cells],ensure_ascii=False).encode()).hexdigest()
    entry={'source_sha256':source_digest,'chapter':chapter,'notebook':path.name,'mode':args.mode,'started_at':datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),'python':platform.python_version()}
    try:
        NotebookClient(notebook,timeout=1200 if args.mode=='quick' else None,kernel_name='grokking-dl',resources={'metadata':{'path':str(ROOT/'notebooks')}},allow_errors=False).execute()
        code_cells=[c for c in notebook.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code_cells)
        assert [c.execution_count for c in code_cells]==list(range(1,len(code_cells)+1))
        metrics={}
        for cell in code_cells:
            for out in cell.outputs:
                if out.output_type=='stream':
                    for line in out.text.splitlines():
                        if line.startswith('VALIDATION_METRICS='): metrics=json.loads(line.split('=',1)[1])
        entry.update(status='PASS',code_cells=len(code_cells),metrics=metrics)
    except Exception as exc:
        entry.update(status='FAIL',error=str(exc)[-5000:]); failed=True
    entry['dependencies']={name:importlib.metadata.version(name) for name in ['numpy','matplotlib','nbformat','nbclient','ipykernel','phe']}
    entry['support_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))+[ROOT/'requirements.txt',ROOT/'data/manifest.json']}
    entry['platform']=platform.platform()
    entry['seconds']=round(time.perf_counter()-start,3)
    notebook.metadata['grokking'].update(status=entry['status'],executed_mode=args.mode,executed_at=entry['started_at'])
    nbformat.write(notebook,path)
    results[str(chapter)]=entry
    report.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(f"{chapter:02}: {entry['status']} ({entry['seconds']} s)",flush=True)
    if entry['status']=='FAIL': print(entry['error'],flush=True)
raise SystemExit(1 if failed else 0)
