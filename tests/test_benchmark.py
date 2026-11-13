import random
import unittest

import benchmark
from benchmark import ChannelRate, LogisticRegression, NaiveBayes, assign_splits, best_f1_threshold, featurize


def row(video_id, channel, title, watched, views=1000, split='train'):
    return {'video_id': video_id, 'channel_id': channel, 'video_title': title, 'description': '',
            'view_count': str(views), 'like_count': '', 'duration_seconds': '600', 'watched': watched,
            'split': split, 'published_at': '2026-01-01T00:00:00Z', 'metadata_fetched_at': '2026-01-11T00:00:00Z'}


class FeatureTests(unittest.TestCase):
    def test_groups_produce_namespaced_features_and_missing_flags(self):
        features = featurize(row('a', 'UC1', 'Sourdough Starter, Day 3!', True), ('meta', 'channel', 'text', 'age'))
        self.assertIn('c:UC1', features)
        self.assertIn('t:sourdough', features)
        self.assertIn('m:likes', features)
        self.assertNotIn('n:log_likes', features)
        self.assertAlmostEqual(features['n:log_views'], 6.9087547793)
        self.assertGreater(features['n:log_age_days'], 0)
        self.assertEqual(featurize(row('a', 'UC1', 'x', True), ('meta',)).keys() & {'c:UC1'}, set())

    def test_channel_holdout_keeps_every_channel_in_one_split(self):
        rows = [row(f'v{i}', f'UC{i % 37}', 't', i % 5 == 0) for i in range(400)]
        splits = assign_splits(rows, 'channel-holdout')
        by_channel = {}
        for r, split in zip(rows, splits):
            by_channel.setdefault(r['channel_id'], set()).add(split)
        self.assertTrue(all(len(s) == 1 for s in by_channel.values()))
        self.assertEqual(set(splits), {'train', 'validation', 'test'})
        self.assertEqual(assign_splits(rows, 'random'), ['train'] * 400)


class ModelTests(unittest.TestCase):
    def test_threshold_maximizes_validation_f1(self):
        self.assertEqual(best_f1_threshold([True, True, False, False], [.9, .8, .3, .1]), .8)
        self.assertEqual(best_f1_threshold([True, False], [.5, .5]), .5)

    def test_channel_rate_is_smoothed_toward_global_rate(self):
        rows = [row('a', 'UC1', 't', True), row('b', 'UC1', 't', True), row('c', 'UC2', 't', False),
                row('d', 'UC2', 't', False)]
        model = ChannelRate(strength=2).fit(rows, [r['watched'] for r in rows])
        seen, other, unseen = model.predict([rows[0], rows[2], row('e', 'UC9', 't', False)])
        self.assertAlmostEqual(seen, 0.75)
        self.assertAlmostEqual(other, 0.25)
        self.assertAlmostEqual(unseen, 0.5)

    def test_text_models_learn_a_separable_vocabulary(self):
        rng = random.Random(1)
        rows = [row(f'v{i}', f'UC{i}', rng.choice(['chess opening', 'chess endgame']) if i % 4 == 0
                    else rng.choice(['pasta night', 'ramen broth']), i % 4 == 0) for i in range(80)]
        labels = [r['watched'] for r in rows]
        for model in (NaiveBayes(), LogisticRegression(('text',))):
            scores = model.fit(rows, labels).predict(rows)
            self.assertEqual(benchmark.metrics(labels, scores)['roc_auc'], 1.0, type(model).__name__)

    def test_run_reports_every_model_with_confidence_intervals(self):
        rng = random.Random(3)
        rows = []
        for i in range(600):
            channel = f'UC{i % 60}'
            watched = rng.random() < (0.5 if i % 60 < 10 else 0.03)
            rows.append(row(f'v{i}', channel, 'video', watched, split=rng.choice(['train'] * 8 + ['validation', 'test'])))
        result = benchmark.run(rows, 'random', {'prior': benchmark.MODELS['prior'],
                                                'channel-rate': benchmark.MODELS['channel-rate']}, bootstrap=50)
        self.assertEqual(result['models']['prior']['roc_auc'], 0.5)
        self.assertGreater(result['models']['channel-rate']['roc_auc'], 0.8)
        low, high = result['models']['channel-rate']['ci95']['roc_auc']
        self.assertLessEqual(low, high)
        self.assertEqual(sum(s['rows'] for s in result['splits'].values()), 600)


if __name__ == '__main__':
    unittest.main()
