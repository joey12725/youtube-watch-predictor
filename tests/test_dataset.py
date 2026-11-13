import unittest

from build_dataset import build_rows


def history(video_id, date='2026-04-22', is_short=False):
    return {'video_id': video_id, 'local_date': date, 'is_short': is_short,
            'format_evidence': ['test fixture format'], 'title': 'Fixture title'}


def feed(video_id):
    return {'video_id': video_id, 'is_short': False,
            'source_feeds': ['home'], 'title': 'Fixture feed video',
            'first_seen_at': '2026-11-09T15:00:00Z', 'format_evidence': ['test fixture format']}


class DatasetTests(unittest.TestCase):
    def test_cutoff_deduplication_history_exclusion_and_unknown_values(self):
        h = [history('p0000000001'), history('p0000000001', '2026-05-01'),
             history('o0000000001', '2023-08-08'), history('s0000000001', is_short=True),
             history('u0000000001', is_short=None)]
        feeds = [feed('n0000000001'), feed('o0000000001'), feed('p0000000001')]
        rows = build_rows(h, [{'video_id': 'p0000000001'}], feeds, [], {}, '2026-04-22')
        by_id = {r['video_id']: r for r in rows}
        self.assertEqual(set(by_id), {'p0000000001', 'n0000000001'})
        self.assertTrue(by_id['p0000000001']['watched'])
        self.assertEqual(by_id['p0000000001']['watch_event_count'], 2)
        self.assertTrue(by_id['p0000000001']['in_watch_later'])
        self.assertIsNone(by_id['p0000000001']['percentage_watched'])
        self.assertIsNone(by_id['n0000000001']['view_count'])
        self.assertFalse(by_id['n0000000001']['watched'])

    def test_ratio_subset_is_unique_deterministic_and_preserves_full_input(self):
        import build_dataset
        rows = [{'video_id': f'p{i:010d}', 'watched': True} for i in range(12)]
        rows += [{'video_id': f'n{i:010d}', 'watched': False} for i in range(105)]
        subset = build_dataset.select_ratio(rows, ratio=10, seed=42)
        self.assertEqual(sum(r['watched'] for r in subset), 10)
        self.assertEqual(sum(not r['watched'] for r in subset), 100)
        self.assertEqual(len({r['video_id'] for r in subset}), 110)
        self.assertEqual(subset, build_dataset.select_ratio(list(reversed(rows)), ratio=10, seed=42))
        self.assertEqual(len(rows), 117)
        self.assertFalse(any('split' in r for r in rows))
        for split in {'train', 'validation', 'test'}:
            part = [r for r in subset if r['split'] == split]
            self.assertEqual(sum(not r['watched'] for r in part), 10 * sum(r['watched'] for r in part))

    def test_cli_writes_reproducible_full_and_ten_to_one_csvs(self):
        import csv
        import gzip
        import json
        import pathlib
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            inputs = root / 'input.json.gz'
            data = {'history': [history('p0000000001')], 'watch_later': [], 'progress': [],
                    'feeds': [feed(f'n{i:010d}') for i in range(10)], 'metadata': {},
                    'source': {'test_fixture': True}}
            with gzip.open(inputs, 'wt') as f:
                json.dump(data, f)
            args = [sys.executable, 'build_dataset.py', '--inputs', str(inputs), '--output-dir', str(root / 'out')]
            run = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = root / 'out/evaluation_10to1.csv.gz'
            self.assertTrue(output.exists(), 'CLI must produce the actual CSV')
            original = output.read_bytes()
            subprocess.run(args, check=True, capture_output=True)
            self.assertEqual(original, output.read_bytes())
            with gzip.open(output, 'rt', newline='') as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 11)
            self.assertEqual(sum(r['watched'] == 'true' for r in records), 1)
            self.assertEqual(records[0]['percentage_watched'], '')
            report = json.loads((root / 'out/dataset_report.json').read_text())
            self.assertTrue(report['full_history_ratio_met'])
            self.assertEqual(report['negative_shortfall'], 0)

    def test_progress_evidence_and_conflicting_short_flags_exclude_negative_candidates(self):
        feeds = [feed('n0000000001'), feed('n0000000002')]
        shorts = [history('n0000000002', is_short=True)]
        progress = [{'video_id': 'n0000000001', 'progress_bar_start_percent_raw': 15}]
        self.assertEqual(build_rows(shorts, [], feeds, progress, {}, '2026-04-22'), [])

    def test_metadata_and_progress_are_snapshotted_without_inventing_exact_percentage(self):
        video_id = 'p0000000001'
        m = {video_id: {'id': video_id,
             'snippet': {'title': 'Actual title', 'channelTitle': 'Actual channel', 'channelId': 'UCfixture',
                         'description': 'Full description', 'publishedAt': '2026-04-01T00:00:00Z',
                         'thumbnails': {'high': {'url': 'https://i.ytimg.com/vi/p0000000001/hqdefault.jpg'}}},
             'contentDetails': {'duration': 'PT1H2M3S'},
             'statistics': {'viewCount': '1234', 'likeCount': '7'}, 'fetched_at': '2026-11-09T16:00:00Z'}}
        p = [{'video_id': video_id, 'observed_at': '2026-11-09T15:00:00Z',
              'resume_start_seconds_raw': 60, 'duration_seconds': 120, 'method': 'resume URL proxy'}]
        row = build_rows([history(video_id)], [], [], p, m, '2026-04-22')[0]
        self.assertEqual(row.get('channel_name'), 'Actual channel')
        self.assertEqual(row.get('view_count'), 1234)
        self.assertEqual(row.get('like_count'), 7)
        self.assertEqual(row.get('description'), 'Full description')
        self.assertEqual(row.get('duration_seconds'), 3723)
        self.assertEqual(row.get('watch_percentage_proxy'), 50)
        self.assertIsNone(row['percentage_watched'])


if __name__ == '__main__':
    unittest.main()
