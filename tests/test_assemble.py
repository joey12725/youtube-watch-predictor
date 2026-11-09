import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from assemble_data import assemble


class AssemblyTests(unittest.TestCase):
    def test_verified_chunks_restore_exact_bytes_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/parts').mkdir(parents=True)
            raw = b'actual fixture compressed bytes'
            (root / 'data/parts/all_available.csv.gz.part000').write_bytes(raw)
            info = {'path': 'data/all_available.csv.gz', 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'parts': [{
                        'path': 'data/parts/all_available.csv.gz.part000', 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()}]}
            (root / 'data/storage_manifest.json').write_text(json.dumps({'version': 1, 'files': [info]}))
            result = assemble(root)
            self.assertEqual(result, ['data/all_available.csv.gz'])
            self.assertEqual((root / info['path']).read_bytes(), raw)
            self.assertEqual(assemble(root), result)
            (root / info['parts'][0]['path']).write_bytes(b'corrupt')
            (root / info['path']).unlink()
            with self.assertRaises(ValueError):
                assemble(root)
            self.assertFalse((root / info['path']).exists())


if __name__ == '__main__':
    unittest.main()
