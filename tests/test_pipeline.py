"""End to end on fabricated data: generate -> build -> benchmark -> evaluate."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from build_dataset import build_rows
import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


class SyntheticTests(unittest.TestCase):
    def test_generator_is_deterministic_and_well_formed(self):
        first, second = make_synthetic.generate(seed=3, channels=10), make_synthetic.generate(seed=3, channels=10)
        self.assertEqual(first, second)
        self.assertTrue(first['source']['synthetic'])
        for group in ('history', 'watch_later', 'feeds', 'progress'):
            for item in first[group]:
                self.assertRegex(item['video_id'], r'^syn[A-Za-z0-9_-]{8}$')
        rows = build_rows(first['history'], first['watch_later'], first['feeds'], first['progress'],
                          first['metadata'], None)
        self.assertTrue(any(r['watched'] for r in rows))
        self.assertTrue(any(not r['watched'] for r in rows))
        self.assertTrue(all(re.fullmatch(r'\d+', str(r['duration_seconds'])) for r in rows))

    def test_cli_pipeline_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def run(*args):
                done = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                return done.stdout

            run('make_synthetic.py', '--output', str(tmp / 'inputs.json.gz'), '--channels', '40')
            run('build_dataset.py', '--inputs', str(tmp / 'inputs.json.gz'), '--output-dir', str(tmp))
            dataset = str(tmp / 'evaluation_10to1.csv.gz')
            table = run('benchmark.py', '--dataset', dataset, '--protocol', 'random', '--bootstrap', '0',
                        '--models', 'prior', 'channel-rate', '--output', str(tmp / 'results.json'))
            self.assertIn('| random | channel-rate |', table)
            results = json.loads((tmp / 'results.json').read_text())
            self.assertEqual(set(results['protocols']['random']['models']),
                             {'prior', 'channel-rate', 'probe: upload age only'})
            metrics = json.loads(run('evaluate.py', '--dataset', dataset, '--baseline', 'always-negative'))
            self.assertEqual(metrics['roc_auc'], 0.5)


if __name__ == '__main__':
    unittest.main()
