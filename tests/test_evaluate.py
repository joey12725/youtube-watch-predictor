import unittest
from evaluate import metrics


class EvaluationTests(unittest.TestCase):
    def test_prediction_ids_must_match_exactly_and_be_unique(self):
        import evaluate
        truth = [{'video_id': 'a', 'watched': True}, {'video_id': 'b', 'watched': False}]
        predictions = [{'video_id': 'b', 'score': '.1'}, {'video_id': 'a', 'score': '.9'}]
        self.assertEqual(evaluate.score_rows(truth, predictions).get('accuracy'), 1)
        for bad in (predictions[:1], predictions + [predictions[0]], predictions + [{'video_id': 'c', 'score': '.2'}]):
            with self.assertRaises(ValueError):
                evaluate.score_rows(truth, bad)

    def test_cli_runs_offline_baseline_on_selected_split(self):
        import csv, gzip, json, pathlib, subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dataset = pathlib.Path(tmp) / 'data.csv.gz'
            with gzip.open(dataset, 'wt', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['video_id', 'watched', 'split'])
                writer.writeheader()
                writer.writerows([{'video_id': 'a', 'watched': 'true', 'split': 'test'},
                                  {'video_id': 'b', 'watched': 'false', 'split': 'test'},
                                  {'video_id': 'c', 'watched': 'false', 'split': 'train'}])
            run = subprocess.run([sys.executable, 'evaluate.py', '--dataset', str(dataset),
                                  '--baseline', 'always-negative', '--split', 'test'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(run.stdout.strip(), 'CLI must return actual metrics')
            report = json.loads(run.stdout)
            self.assertEqual(report['rows'], 2)
            self.assertEqual(report['accuracy'], .5)
            self.assertEqual(report['roc_auc'], .5)

    def test_ties_and_invalid_scores(self):
        self.assertEqual(metrics([True, False], [.5, .5])['roc_auc'], .5)
        self.assertEqual(metrics([True, False], [.5, .5])['average_precision'], .5)
        for labels, scores in [([], []), ([True], []), ([True], [float('nan')]),
                               ([True], [float('inf')]), ([True], [1.1]), ([True], [-.1])]:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    metrics(labels, scores)

    def test_actual_binary_scores_produce_confusion_and_ranking_metrics(self):
        result = metrics([True, False, True, False], [.9, .1, .4, .3])
        self.assertEqual(result.get('accuracy'), 3 / 4)
        self.assertEqual(result.get('true_positive'), 1)
        self.assertEqual(result.get('false_negative'), 1)
        self.assertEqual(result.get('true_negative'), 2)
        self.assertAlmostEqual(result.get('f1', -1), 2 / 3)
        self.assertEqual(result.get('roc_auc'), 1)
        self.assertEqual(result.get('average_precision'), 1)


if __name__ == '__main__':
    unittest.main()
