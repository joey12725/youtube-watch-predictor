"""Build the offline personal YouTube evaluation dataset (standard library only)."""
from collections import defaultdict
import re
import random


def select_ratio(rows, ratio=10, seed=20261109):
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError('ratio must be a positive integer')
    rng = random.Random(seed)
    positive = sorted((dict(r) for r in rows if r['watched']), key=lambda r: r['video_id'])
    negative = sorted((dict(r) for r in rows if not r['watched']), key=lambda r: r['video_id'])
    rng.shuffle(positive)
    rng.shuffle(negative)
    count = min(len(positive), len(negative) // ratio)
    positive, negative = positive[:count], negative[:count * ratio]
    counts = [int(count * .8), int(count * .1)]
    counts.append(count - sum(counts))
    start = 0
    for name, size in zip(('train', 'validation', 'test'), counts):
        for row in positive[start:start + size] + negative[start * ratio:(start + size) * ratio]:
            row['split'] = name
        start += size
    return sorted(positive + negative, key=lambda r: r['video_id'])


def duration_seconds(value):
    if not value:
        return None
    match = re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?', value)
    if not match:
        raise ValueError('Invalid ISO duration: ' + value)
    return int(sum(float(v or 0) * unit for v, unit in zip(match.groups(), (86400, 3600, 60, 1))))


def progress_proxy(observation):
    for key in ('percent_duration_watched_raw', 'progress_bar_start_percent_raw'):
        value = observation.get(key)
        if isinstance(value, (float, int)) and 0 <= value <= 100:
            return value
    seconds, duration = observation.get('resume_start_seconds_raw'), observation.get('duration_seconds')
    if isinstance(seconds, (float, int)) and isinstance(duration, (float, int)) and duration > 0:
        return round(min(100, max(0, 100 * seconds / duration)), 4)
    return None


def build_rows(history, watch_later, feeds, progress, metadata, cutoff):
    events = defaultdict(list)
    all_history = {r['video_id'] for r in history}
    shorts = {r['video_id'] for r in history + feeds if r.get('is_short') is True}
    for r in history:
        if r['local_date'] >= cutoff and r.get('is_short') is False and r['video_id'] not in shorts:
            events[r['video_id']].append(r)
    saved = {r['video_id'] for r in watch_later}
    evidence_of_watching = {p['video_id'] for p in progress
                           if (progress_proxy(p) or 0) > 0
                           or (p.get('resume_start_seconds_raw') or 0) > 0
                           or 'WATCHED' in str(p.get('playback_status_label_raw') or '').upper()}
    candidates = {r['video_id']: r for r in feeds if r.get('is_short') is False
                  and r['video_id'] not in all_history and r['video_id'] not in shorts
                  and r['video_id'] not in evidence_of_watching}
    feed_map = {r['video_id']: r for r in feeds}
    observations = defaultdict(list)
    for p in progress:
        observations[p['video_id']].append(p)
    result = []
    for video_id in sorted(set(events) | set(candidates)):
        watched = video_id in events
        source = events[video_id][0] if watched else candidates[video_id]
        dates = sorted(r['local_date'] for r in events.get(video_id, []))
        m = metadata.get(video_id, {})
        snippet, statistics = m.get('snippet', {}), m.get('statistics', {})
        thumbnails = snippet.get('thumbnails', {})
        thumbnail = next((thumbnails[k]['url'] for k in ('maxres', 'standard', 'high', 'medium', 'default') if k in thumbnails), source.get('thumbnail_url'))
        duration = duration_seconds(m.get('contentDetails', {}).get('duration'))
        observations_with_values = [p for p in observations[video_id] if progress_proxy(p) is not None]
        latest = max(observations_with_values, key=lambda p: p.get('observed_at') or '', default={})
        result.append({
            'video_id': video_id, 'url': 'https://www.youtube.com/watch?v=' + video_id,
            'channel_id': snippet.get('channelId') or source.get('channel_id'),
            'channel_name': snippet.get('channelTitle') or source.get('channel_name'),
            'view_count': int(statistics['viewCount']) if 'viewCount' in statistics else None,
            'video_title': snippet.get('title') or source.get('title'),
            'like_count': int(statistics['likeCount']) if 'likeCount' in statistics else None,
            'description': snippet.get('description'), 'thumbnail_url': thumbnail,
            'duration_seconds': duration if duration is not None else source.get('duration_seconds'),
            'watched': watched, 'percentage_watched': None, 'in_watch_later': video_id in saved,
            'watch_percentage_proxy': progress_proxy(latest),
            'watch_percentage_proxy_method': latest.get('method'),
            'progress_observed_at': latest.get('observed_at'),
            'first_watch_date': dates[0] if dates else None,
            'last_watch_date': dates[-1] if dates else None,
            'watch_event_count': len(dates),
            'source_feeds': '|'.join(sorted(feed_map.get(video_id, {}).get('source_feeds', []))),
            'first_feed_seen_at': feed_map.get(video_id, {}).get('first_seen_at'),
            'published_at': snippet.get('publishedAt'),
            'metadata_status': 'available' if snippet else 'unavailable',
            'metadata_fetched_at': m.get('fetched_at'),
            'label_basis': 'recorded_watch_in_window' if watched else 'feed_candidate_absent_from_all_collected_history',
            'format_evidence': '|'.join(source.get('format_evidence', [])),
        })
    return result


def write_csv(path, rows):
    import csv
    import gzip
    import io
    if not rows:
        raise ValueError('No eligible rows')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: str(v).lower() if type(v) is bool else v for k, v in row.items()})
    path.write_bytes(gzip.compress(buffer.getvalue().encode('utf-8'), mtime=0))


def main():
    import argparse
    import gzip
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/rebuild_inputs.json.gz'))
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--cutoff', default='2026-04-22')
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.cutoff):
        parser.error('cutoff must be YYYY-MM-DD')
    inputs = json.loads(gzip.decompress(args.inputs.read_bytes()))
    for group in ('history', 'watch_later', 'feeds', 'progress'):
        for row in inputs[group]:
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}', row['video_id']):
                raise ValueError('Invalid video ID in ' + group)
    rows = build_rows(inputs['history'], inputs['watch_later'], inputs['feeds'],
                      inputs['progress'], inputs['metadata'], args.cutoff)
    rng = random.Random(args.seed)
    for label in (True, False):
        group = [r for r in rows if r['watched'] is label]
        rng.shuffle(group)
        for i, row in enumerate(group):
            row['split'] = 'train' if i < int(len(group) * .8) else 'validation' if i < int(len(group) * .9) else 'test'
    subset = select_ratio(rows, seed=args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'all_available.csv.gz', rows)
    write_csv(output / 'evaluation_10to1.csv.gz', subset)
    positives = sum(r['watched'] for r in rows)
    negatives = len(rows) - positives
    shortfall = max(0, positives * 10 - negatives)
    report = {
        'cutoff': args.cutoff, 'cutoff_status': 'first recorded day after large history gap, not proven setting-change date',
        'seed': args.seed, 'rows': len(rows), 'watched': positives, 'negative_candidates': negatives,
        'full_history_ratio_met': shortfall == 0, 'negative_shortfall': shortfall,
        'ten_to_one_subset_rows': len(subset), 'ten_to_one_subset_watched': sum(r['watched'] for r in subset),
        'ten_to_one_subset_negative': sum(not r['watched'] for r in subset),
        'positives_not_selected_for_ratio_subset': positives - sum(r['watched'] for r in subset),
        'metadata_unavailable': sum(r['metadata_status'] == 'unavailable' for r in rows),
        'missing_like_counts': sum(r['like_count'] is None for r in rows),
        'proxy_percentage_present': sum(r['watch_percentage_proxy'] is not None for r in rows),
        'exact_percentage_present': sum(r['percentage_watched'] is not None for r in rows),
        'source': inputs.get('source', {}),
        'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'artifacts': {name: {'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest(),
                             'compressed_bytes': (output / name).stat().st_size}
                      for name in ('all_available.csv.gz', 'evaluation_10to1.csv.gz')},
        'split_counts': {variant: {split: {'watched': sum(r['watched'] for r in items if r['split'] == split),
                                         'negative': sum(not r['watched'] for r in items if r['split'] == split)}
                                   for split in ('train', 'validation', 'test')}
                         for variant, items in [('all_available', rows), ('evaluation_10to1', subset)]},
        'limitations': [
            'False watched labels mean absent from all collected history, not proven never watched or disliked.',
            'Exact personal watch percentages are unavailable and remain blank; proxies are separately named.',
            'Shorts and unknown formats are excluded; unavailable video metadata remain explicitly missing.',
            'Negative exposures were collected in November, whereas positive watches span April-November: temporal/exposure bias.',
            'Metadata and Watch Later are collection-time snapshots, not historical state at watch/exposure time.',
            'Splits are stratified by label and video-disjoint within each artifact, not channel-disjoint or temporal. Do not mix dataset variants.',
            'The ratio subset samples positives; the all-available file preserves every eligible positive.'
        ]
    }
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source', 'split_counts', 'limitations')}))


if __name__ == '__main__':
    main()
