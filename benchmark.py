"""Train and compare baseline watch predictors under leakage-aware protocols (stdlib only).

Every model is fit on the train split, its decision threshold is tuned for F1 on
validation, and it is scored once on test with evaluate.metrics. Ranking metrics
get seeded bootstrap 95% confidence intervals.

Protocols:
  random           the split column stored in the CSV (label-stratified, video-disjoint)
  channel-holdout  re-split by a stable hash of channel_id, so test channels are never seen in training
"""
from collections import Counter, defaultdict
from datetime import datetime
import math
import random
import re
import zlib

from evaluate import metrics

WORD = re.compile(r'[^\W_]+')
DESCRIPTION_TOKENS = 80
SPLITS = ('train', 'validation', 'test')


def tokens(text, limit=None):
    words = [w for w in WORD.findall((text or '').lower()) if len(w) > 1]
    return words[:limit] if limit else words


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value):
    try:
        return datetime.fromisoformat((value or '').replace('Z', '+00:00'))
    except ValueError:
        return None


def upload_age_days(row):
    published, fetched = _timestamp(row.get('published_at')), _timestamp(row.get('metadata_fetched_at'))
    if published is None or fetched is None:
        return None
    return max(0.0, (fetched - published).total_seconds() / 86400)


def featurize(row, groups):
    """Sparse feature dict. Keys starting with 'n:' are numeric and get standardized."""
    features = {}
    if 'meta' in groups:
        views, likes, duration = (_float(row.get(k)) for k in ('view_count', 'like_count', 'duration_seconds'))
        for name, value in (('views', views), ('likes', likes), ('duration', duration)):
            if value is None or value < 0:
                features['m:' + name] = 1.0
            else:
                features['n:log_' + name] = math.log1p(value)
        if views and likes is not None:
            features['n:like_rate'] = likes / views
        if duration is not None:
            features['b:duration_' + str(min(7, int(math.log2(1 + duration / 60))))] = 1.0
    if 'channel' in groups and row.get('channel_id'):
        features['c:' + row['channel_id']] = 1.0
    if 'text' in groups:
        for word in set(tokens(row.get('video_title'))):
            features['t:' + word] = 1.0
        for word in set(tokens(row.get('description'), DESCRIPTION_TOKENS)):
            features['d:' + word] = 1.0
    if 'age' in groups:
        age = upload_age_days(row)
        if age is None:
            features['m:age'] = 1.0
        else:
            features['n:log_age_days'] = math.log1p(age)
    return features


class Prior:
    """Predicts the training positive rate for everything."""

    def fit(self, rows, labels):
        self.rate = sum(labels) / len(labels)
        return self

    def predict(self, rows):
        return [self.rate for _ in rows]


class ChannelRate:
    """Beta-smoothed per-channel watch rate; unseen channels fall back to the global rate."""

    def __init__(self, strength=4.0):
        self.strength = strength

    def fit(self, rows, labels):
        self.rate = sum(labels) / len(labels)
        self.counts = defaultdict(lambda: [0, 0])
        for row, label in zip(rows, labels):
            self.counts[row.get('channel_id')][0] += label
            self.counts[row.get('channel_id')][1] += 1
        return self

    def predict(self, rows):
        out = []
        for row in rows:
            positive, total = self.counts.get(row.get('channel_id'), (0, 0))
            out.append((positive + self.strength * self.rate) / (total + self.strength))
        return out


class NaiveBayes:
    """Multinomial naive Bayes over title and description word sets."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, rows, labels):
        counts = (Counter(), Counter())
        for row, label in zip(rows, labels):
            counts[int(label)].update(featurize(row, ('text',)))
        vocab = set(counts[0]) | set(counts[1])
        totals = [sum(c.values()) + self.alpha * len(vocab) for c in counts]
        self.log_ratio = {w: math.log((counts[1][w] + self.alpha) / totals[1])
                          - math.log((counts[0][w] + self.alpha) / totals[0]) for w in vocab}
        positive = sum(labels)
        self.bias = math.log(positive / (len(labels) - positive))
        return self

    def predict(self, rows):
        return [_sigmoid(self.bias + sum(self.log_ratio.get(w, 0.0) for w in featurize(row, ('text',))))
                for row in rows]


class LogisticRegression:
    """L2-regularized logistic regression trained with Adagrad on sparse features."""

    def __init__(self, groups, epochs=12, learning_rate=0.1, l2=1e-4, min_count=2, seed=0):
        self.groups, self.epochs, self.learning_rate = groups, epochs, learning_rate
        self.l2, self.min_count, self.seed = l2, min_count, seed

    def _encode(self, rows):
        encoded = []
        for row in rows:
            vector = []
            for key, value in featurize(row, self.groups).items():
                index = self.index.get(key)
                if index is None:
                    continue
                if key.startswith('n:'):
                    mean, std = self.scale[key]
                    value = (value - mean) / std
                vector.append((index, value))
            encoded.append(vector)
        return encoded

    def fit(self, rows, labels):
        raw = [featurize(row, self.groups) for row in rows]
        seen = Counter(key for features in raw for key in features)
        keys = sorted(k for k, n in seen.items() if k.startswith('n:') or n >= self.min_count)
        self.index = {k: i for i, k in enumerate(keys)}
        self.scale = {}
        for key in keys:
            if key.startswith('n:'):
                values = [f[key] for f in raw if key in f]
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
                self.scale[key] = (mean, std)
        data = self._encode(rows)
        self.weights = [0.0] * len(keys)
        squared = [1e-8] * len(keys)
        self.bias, bias_squared = 0.0, 1e-8
        order = list(range(len(rows)))
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(order)
            for i in order:
                vector = data[i]
                error = _sigmoid(self.bias + sum(self.weights[j] * v for j, v in vector)) - labels[i]
                for j, v in vector:
                    gradient = error * v + self.l2 * self.weights[j]
                    squared[j] += gradient * gradient
                    self.weights[j] -= self.learning_rate * gradient / math.sqrt(squared[j])
                bias_squared += error * error
                self.bias -= self.learning_rate * error / math.sqrt(bias_squared)
        return self

    def predict(self, rows):
        return [_sigmoid(self.bias + sum(self.weights[j] * v for j, v in vector)) for vector in self._encode(rows)]


def _sigmoid(x):
    if x < -35:
        return 1e-15
    if x > 35:
        return 1 - 1e-15
    return 1 / (1 + math.exp(-x))


MODELS = {
    'prior': lambda seed: Prior(),
    'channel-rate': lambda seed: ChannelRate(),
    'naive-bayes-text': lambda seed: NaiveBayes(),
    'logreg-meta': lambda seed: LogisticRegression(('meta',), seed=seed),
    'logreg-text': lambda seed: LogisticRegression(('text',), seed=seed),
    'logreg-channel-meta': lambda seed: LogisticRegression(('meta', 'channel'), seed=seed),
    'logreg-all': lambda seed: LogisticRegression(('meta', 'channel', 'text'), seed=seed),
}
# Upload age separates classes for collection reasons (watches accumulate over months,
# feed candidates are recent), not taste. It is measured on its own as a leakage probe
# and kept out of every model's feature set.
PROBES = {
    'probe: upload age only': lambda seed: LogisticRegression(('age',), seed=seed),
}


def assign_splits(rows, protocol):
    if protocol == 'random':
        return [row['split'] for row in rows]
    if protocol == 'channel-holdout':
        out = []
        for row in rows:
            bucket = zlib.crc32((row.get('channel_id') or row['video_id']).encode('utf-8')) % 10
            out.append('test' if bucket == 0 else 'validation' if bucket == 1 else 'train')
        return out
    raise ValueError('Unknown protocol: ' + protocol)


def best_f1_threshold(labels, scores):
    """Threshold on the validation scores that maximizes F1 (predict positive when score >= threshold)."""
    positive = sum(labels)
    best, best_f1 = 1.0, -1.0
    tp = fp = 0
    ranked = sorted(zip(scores, labels), reverse=True)
    for i, (score, label) in enumerate(ranked):
        tp += label
        fp += not label
        if i + 1 < len(ranked) and ranked[i + 1][0] == score:
            continue
        f1 = 2 * tp / (tp + fp + positive) if positive else 0.0
        if f1 > best_f1:
            best, best_f1 = score, f1
    return best


def bootstrap_ci(labels, scores, keys=('roc_auc', 'average_precision'), samples=500, seed=0):
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    n = len(labels)
    draws = defaultdict(list)
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [labels[i] for i in idx]
        if all(sample) or not any(sample):
            continue
        result = metrics(sample, [scores[i] for i in idx])
        for key in keys:
            draws[key].append(result[key])
    out = {}
    for key, values in draws.items():
        values.sort()
        out[key] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return out


def run(rows, protocol, models, seed=0, bootstrap=500):
    splits = assign_splits(rows, protocol)
    parts = {name: [r for r, s in zip(rows, splits) if s == name] for name in SPLITS}
    labels = {name: [r['watched'] for r in part] for name, part in parts.items()}
    for name in SPLITS:
        if not any(labels[name]) or all(labels[name]):
            raise ValueError(f'{protocol} split {name!r} needs both classes')
    summary = {'splits': {name: {'rows': len(parts[name]), 'positive': sum(labels[name])} for name in SPLITS},
               'models': {}}
    for name, factory in models.items():
        model = factory(seed).fit(parts['train'], labels['train'])
        threshold = best_f1_threshold(labels['validation'], model.predict(parts['validation']))
        test_scores = model.predict(parts['test'])
        result = metrics(labels['test'], test_scores, threshold)
        result['ci95'] = bootstrap_ci(labels['test'], test_scores, samples=bootstrap, seed=seed)
        summary['models'][name] = result
    return summary


def load_rows(path):
    import csv
    import gzip
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if any(r['watched'] not in ('true', 'false') for r in rows):
        raise ValueError('watched must be literal true or false')
    return [{**r, 'watched': r['watched'] == 'true'} for r in rows]


def markdown_table(results):
    lines = ['| Protocol | Model | ROC AUC (95% CI) | Avg precision | F1 | Balanced acc. |',
             '|---|---|---|---|---|---|']
    for protocol, summary in results['protocols'].items():
        for name, r in summary['models'].items():
            ci = r['ci95'].get('roc_auc')
            auc = f"{r['roc_auc']:.3f}" + (f' ({ci[0]:.3f}-{ci[1]:.3f})' if ci else '')
            lines.append(f"| {protocol} | {name} | {auc} | {r['average_precision']:.3f} | {r['f1']:.3f} "
                         f"| {r['balanced_accuracy']:.3f} |")
    return '\n'.join(lines)


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path('data/evaluation_10to1.csv.gz'))
    parser.add_argument('--protocol', choices=('random', 'channel-holdout', 'both'), default='both')
    parser.add_argument('--models', nargs='+', choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument('--no-probes', action='store_true', help='skip the upload-age leakage probe')
    parser.add_argument('--bootstrap', type=int, default=500, help='bootstrap resamples for CIs (0 disables)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='write the full results JSON here')
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    models = {name: MODELS[name] for name in args.models}
    if not args.no_probes:
        models.update(PROBES)
    protocols = ('random', 'channel-holdout') if args.protocol == 'both' else (args.protocol,)
    results = {'dataset_rows': len(rows), 'dataset_positive': sum(r['watched'] for r in rows),
               'seed': args.seed, 'bootstrap': args.bootstrap,
               'protocols': {p: run(rows, p, models, args.seed, args.bootstrap) for p in protocols}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(markdown_table(results))


if __name__ == '__main__':
    main()
