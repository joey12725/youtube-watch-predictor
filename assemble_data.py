"""Restore large reviewed data files from small GitHub transport chunks, offline."""
import hashlib
import json
from pathlib import Path

TARGETS = {'data/rebuild_inputs.json.gz', 'data/all_available.csv.gz', 'data/evaluation_10to1.csv.gz'}


def assemble(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'data/storage_manifest.json').read_text())
    if manifest.get('version') != 1:
        raise ValueError('Unsupported storage manifest version')
    restored = []
    for entry in manifest['files']:
        name = entry['path']
        if name not in TARGETS or name in restored:
            raise ValueError('Invalid or duplicate target path')
        target = root / name
        if target.is_symlink() or target.parent.is_symlink():
            raise ValueError('Symlink target is not permitted')
        if target.exists():
            if len(target.read_bytes()) != entry['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Existing output differs; refusing to overwrite: ' + name)
            restored.append(name)
            continue
        data = bytearray()
        for index, part in enumerate(entry['parts']):
            expected_path = 'data/parts/' + Path(name).name + f'.part{index:03d}'
            if part['path'] != expected_path:
                raise ValueError('Invalid chunk path or order')
            path = root / expected_path
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError('Symlink chunk is not permitted')
            chunk = path.read_bytes()
            if len(chunk) != part['bytes'] or hashlib.sha256(chunk).hexdigest() != part['sha256']:
                raise ValueError('Corrupt chunk: ' + expected_path)
            data.extend(chunk)
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Assembled output hash/size mismatch: ' + name)
        with target.open('xb') as output:
            output.write(data)
        restored.append(name)
    return restored


if __name__ == '__main__':
    print(json.dumps({'restored': assemble(Path(__file__).resolve().parent)}))
