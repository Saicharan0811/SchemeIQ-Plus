import sys, json
sys.path.insert(0, 'C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus')
from pathlib import Path

root = Path('C:/Users/CHIKITHA/OneDrive/SchemeIQ-Plus')

paths_to_check = [
    'data/documents/raw',
    'data/processed',
    'data/vector_store/chroma_db',
    'src/rag/hybrid_retriever.py',
    'src/rag/query_detector.py',
    'src/rag/keyword_retriever.py',
    'src/rag/vector_store.py',
    'src/rag/schemas.py',
]
print('=== PATH EXISTENCE CHECK ===')
all_ok = True
for p in paths_to_check:
    exists = (root / p).exists()
    status = 'OK' if exists else 'MISSING'
    if not exists:
        all_ok = False
    print(f'  [{status}] {p}')

proc_docs = sorted((root / 'data/processed/documents').glob('*.txt'))
print(f'\n=== PROCESSED CORPUS ===')
print(f'  .txt files: {len(proc_docs)}')
for f in proc_docs:
    print(f'    {f.name}')

manifest_path = root / 'data/processed/corpus_manifest.json'
with open(manifest_path) as f:
    manifest = json.load(f)
scheme_ids = sorted(set(e['scheme_id'] for e in manifest))
print(f'  Schemes covered: {len(scheme_ids)}')
print(f'  Scheme IDs: {scheme_ids}')

raw_files = sorted((root / 'data/documents/raw').glob('*.html'))
print(f'\n=== RAW CORPUS ===')
print(f'  .html files: {len(raw_files)}')
with open(root / 'data/documents/raw/metadata.json') as f:
    metadata = json.load(f)
print(f'  Metadata records: {len(metadata)}')

print(f'\n=== VECTOR STORE ===')
from src.rag.vector_store import VectorStoreManager
vs = VectorStoreManager()
count = vs.collection.count()
print(f'  Total vectors: {count}')

from src.rag.hybrid_retriever import HybridRetriever
hr = HybridRetriever()
stats = hr._keyword_retriever.get_index_stats()
print(f'  HybridRetriever: importable OK')
print(f'  BM25 index chunks: {stats["total_chunks"]}')
per_scheme = stats["per_scheme_chunks"]
print(f'  Per-scheme chunk counts: {dict(sorted(per_scheme.items()))}')

env_path = root / '.env'
print(f'\n=== ENVIRONMENT (.env) ===')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            masked = (val.strip()[:4] + '...') if len(val.strip()) > 4 else ('[empty]' if val.strip() == '' else val.strip())
            print(f'  {key.strip()} = {masked}')
else:
    print('  .env NOT FOUND')

print('\n=== OPENAI PACKAGE ===')
try:
    import openai
    print(f'  openai version: {openai.__version__}')
except ImportError:
    print('  openai: NOT installed')

print('\n=== SRC/RAG DIRECTORY ===')
for f in sorted((root / 'src/rag').iterdir()):
    if f.suffix == '.py':
        print(f'  {f.name}  ({f.stat().st_size} bytes)')

print(f'\n=== PRE-FLIGHT: {"PASS" if all_ok else "FAIL"} ===')
