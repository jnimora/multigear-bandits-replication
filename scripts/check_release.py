"""Check the portable release inventory, checksums and conservative size budget."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'provenance/FILE_SHA256.json'
EXCLUDED={'.git','__pycache__','results','.julia','release-artifacts'}
def files():
    return [p for p in ROOT.rglob('*') if p.is_file() and p!=MANIFEST
            and not any(x in EXCLUDED or x.startswith('.venv') for x in p.relative_to(ROOT).parts)
            and p.suffix not in ('.pyc','.pyo') and p.name!='.DS_Store']
def main():
    paths=files();actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    expected=json.loads(MANIFEST.read_text())
    assert actual==expected,'Release inventory changed; review and deliberately regenerate its manifest'
    total=sum(p.stat().st_size for p in paths)
    largest=max(paths,key=lambda p:p.stat().st_size)
    assert total<50*1024**2,'Release exceeds its 50 MiB curation budget'
    assert largest.stat().st_size<5*1024**2,'A file exceeds the 5 MiB curation budget'
    for p in paths:
        if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    print(f'PASS: {len(paths)} files, {total/1024**2:.2f} MiB; all SHA-256 checksums match.')
if __name__=='__main__':main()
