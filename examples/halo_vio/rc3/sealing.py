"""Write-once package integrity. Does not modify a published package."""
from pathlib import Path
import hashlib,json,os
EXCLUDED = {'MANIFEST.json', 'SEAL.json'}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()
def payload(root):
    root=Path(root).resolve();files={}
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root).as_posix()
        if p.is_symlink():raise ValueError('sealed payload cannot contain symlinks: '+rel)
        if p.is_file() and rel not in EXCLUDED:files[rel]=dict(sha256=digest(p),size=p.stat().st_size)
    return files
def seal(root,metadata):
    root=Path(root)
    if any((root/n).exists()for n in EXCLUDED):raise FileExistsError('already sealed; use a new candidate directory')
    manifest=dict(schema=1,metadata=metadata,files=payload(root))
    with(root/'MANIFEST.json').open('x')as f:json.dump(manifest,f,indent=2)
    with(root/'SEAL.json').open('x')as f:json.dump(dict(manifest_sha256=digest(root/'MANIFEST.json')),f,indent=2)
    return verify(root)
def verify(root):
    root=Path(root);seal=json.loads((root/'SEAL.json').read_text())
    if digest(root/'MANIFEST.json')!=seal['manifest_sha256']:raise ValueError('manifest was changed')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if payload(root)!=manifest['files']:raise ValueError('sealed payload differs from manifest')
    return manifest

def publish(stage,destination):
    stage=Path(stage);destination=Path(destination)
    verify(stage)
    claim=destination.with_name('.'+destination.name+'.publish-claim')
    fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.close(fd)
    try:
        if destination.exists():raise FileExistsError('destination already exists; refusing overwrite')
        stage.rename(destination)
        return verify(destination)
    finally:
        claim.unlink()
