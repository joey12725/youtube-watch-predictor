"""Offline evaluation of video_id,score CSV predictions; no network dependencies."""
from itertools import groupby
import math


def score_rows(rows, predictions, threshold=.5):
    truth = {r['video_id']: r['watched'] for r in rows}
    scores = {r['video_id']: float(r['score']) for r in predictions}
    if len(truth) != len(rows):
        raise ValueError('Duplicate video IDs in dataset')
    if len(scores) != len(predictions):
        raise ValueError('Duplicate video IDs in predictions')
    missing, extra = set(truth) - set(scores), set(scores) - set(truth)
    if missing or extra:
        raise ValueError(f'Prediction ID mismatch: {len(missing)} missing, {len(extra)} extra')
    ids = sorted(truth)
    return metrics([truth[v] for v in ids], [scores[v] for v in ids], threshold)


def metrics(labels, scores, threshold=.5):
    if not labels or len(labels) != len(scores):
        raise ValueError('Nonempty labels and one score per label are required')
    if any(type(y) is not bool for y in labels):
        raise ValueError('Labels must be booleans')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and between 0 and 1')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Scores must be finite probabilities between 0 and 1')
    ordered = sorted(zip(scores, labels), reverse=True)
    positive = sum(labels)
    negative = len(labels) - positive
    tp = sum(y and s >= threshold for y, s in zip(labels, scores))
    fp = sum(not y and s >= threshold for y, s in zip(labels, scores))
    fn, tn = positive - tp, negative - fp
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    seen_p = seen_n = 0
    auc_num = ap_num = 0.0
    for _, group in groupby(ordered, key=lambda x: x[0]):
        group = list(group)
        gp = sum(y for _, y in group)
        gn = len(group) - gp
        auc_num += gp * (negative - seen_n - gn + gn / 2)
        seen_p += gp
        seen_n += gn
        ap_num += gp * seen_p / (seen_p + seen_n)
    return {'rows': len(labels), 'positive': positive, 'negative': negative,
            'threshold': threshold, 'true_positive': tp, 'false_positive': fp,
            'true_negative': tn, 'false_negative': fn,
            'accuracy': (tp + tn) / len(labels), 'precision': precision, 'recall': recall,
            'specificity': specificity,
            'balanced_accuracy': (recall + specificity) / 2 if positive and negative else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            'roc_auc': auc_num / (positive * negative) if positive and negative else None,
            'average_precision': ap_num / positive if positive else None,
            'brier_score': sum((float(y) - s) ** 2 for y, s in zip(labels, scores)) / len(labels)}


def main():
    import argparse
    import csv
    import gzip
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/evaluation_10to1.csv.gz'))
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--threshold', type=float, default=.5)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', type=Path, help='CSV with video_id,score for exactly the selected split')
    group.add_argument('--baseline', choices=('always-negative',))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.dataset.suffix == '.gz' else open
    with opener(args.dataset, 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['video_id'] for r in rows}) != len(rows):
        parser.error('Dataset contains duplicate video IDs')
    if any(r['watched'] not in ('true', 'false') for r in rows):
        parser.error('watched must be literal true or false')
    rows = [{**r, 'watched': r['watched'] == 'true'} for r in rows
            if args.split == 'all' or r.get('split') == args.split]
    if args.baseline:
        predictions = [{'video_id': r['video_id'], 'score': 0.0} for r in rows]
    else:
        with args.predictions.open(encoding='utf-8', newline='') as f:
            predictions = list(csv.DictReader(f))
    try:
        result = score_rows(rows, predictions, args.threshold)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    result.update({'split': args.split, 'baseline': args.baseline,
                   'label_warning': 'watched=false means unrecorded in collected history, not confirmed rejection'})
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
