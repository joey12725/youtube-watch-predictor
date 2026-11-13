"""Generate a fabricated YouTube activity export in the format build_dataset.py reads.

Every channel, title, description and ID here is invented. A hidden viewer profile
(topic affinity, per-channel affinity, a taste for certain lengths) decides which
videos get "watched", so the baselines have real signal to find and the
channel-holdout protocol has something to expose.
"""
from datetime import date, datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import random

ID_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
COLLECTED = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)

TOPICS = {
    'cooking': ['sourdough', 'ramen', 'braise', 'knife', 'pasta', 'spice', 'kitchen', 'grill', 'broth', 'dumpling'],
    'woodworking': ['dovetail', 'chisel', 'workbench', 'walnut', 'joinery', 'lathe', 'plane', 'finish', 'shop', 'oak'],
    'astronomy': ['telescope', 'nebula', 'eclipse', 'galaxy', 'orbit', 'comet', 'saturn', 'astrophotography', 'star', 'moon'],
    'chess': ['opening', 'gambit', 'endgame', 'sicilian', 'blitz', 'tactics', 'grandmaster', 'puzzle', 'rook', 'checkmate'],
    'synths': ['modular', 'oscillator', 'filter', 'sequencer', 'patch', 'analog', 'eurorack', 'drum', 'ambient', 'synth'],
    'printing': ['filament', 'nozzle', 'slicer', 'resin', 'calibration', 'printer', 'layer', 'enclosure', 'petg', 'bed'],
    'motorsport': ['qualifying', 'pitstop', 'overtake', 'tyre', 'circuit', 'downforce', 'race', 'lap', 'paddock', 'grid'],
    'finance': ['budget', 'index', 'portfolio', 'mortgage', 'tax', 'savings', 'dividend', 'inflation', 'retirement', 'fund'],
    'retro-games': ['cartridge', 'speedrun', 'console', 'emulator', 'pixel', 'arcade', 'platformer', 'boss', 'glitch', 'crt'],
    'urbanism': ['transit', 'zoning', 'bikeway', 'streetcar', 'housing', 'suburb', 'parking', 'downtown', 'rail', 'plaza'],
    'birding': ['warbler', 'migration', 'binoculars', 'heron', 'owl', 'feeder', 'birdsong', 'hawk', 'wetland', 'sparrow'],
    'climbing': ['bouldering', 'crimp', 'belay', 'route', 'chalk', 'overhang', 'rope', 'crag', 'grade', 'campus'],
}
TEMPLATES = [
    'I tried {a} for 30 days', 'The only {a} guide you need', 'Why {a} beats {b}', '{a} vs {b}: honest comparison',
    'Building a {a} from scratch', '{a} mistakes beginners make', 'A quiet afternoon of {a}', 'The {a} {b} deep dive',
    'Everything wrong with my {a}', '{a} explained in 12 minutes', 'Ranking every {a}', 'Fixing a broken {a}',
]
FILLER = ['In this episode', 'Today we look at', 'Thanks for watching', 'Links in the description',
          'Chapters below', 'Let me know in the comments', 'Big thanks to everyone who asked about']
ADJECTIVES = ['Copper', 'Quiet', 'Northern', 'Lucky', 'Hollow', 'Bright', 'Patient', 'Velvet', 'Rusty', 'Little',
              'Midnight', 'Second', 'Paper', 'Hidden', 'Daily', 'Slow']
NOUNS = ['Lantern', 'Workshop', 'Signal', 'Harbor', 'Notebook', 'Garage', 'Orbit', 'Kettle', 'Bench', 'Field',
         'Tape', 'Compass', 'Atlas', 'Studio', 'Corner', 'Lab']


def fake_id(rng, prefix, length, used):
    while True:
        value = prefix + ''.join(rng.choice(ID_ALPHABET) for _ in range(length - len(prefix)))
        if value not in used:
            used.add(value)
            return value


def iso_duration(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return 'PT' + (f'{hours}H' if hours else '') + (f'{minutes}M' if minutes else '') + f'{secs}S'


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def generate(seed=7, channels=120, videos_per_channel=(25, 60), intercept=-3.0):
    rng = random.Random(seed)
    used = set()
    topic_affinity = {t: rng.gauss(0, 1.6) for t in TOPICS}
    favourite_length = rng.choice([8 * 60, 15 * 60, 35 * 60])
    names = rng.sample([f'{a} {n}' for a in ADJECTIVES for n in NOUNS], channels)
    channel_rows = []
    for name in names:
        topic = rng.choice(list(TOPICS))
        channel_rows.append({'id': fake_id(rng, 'UCsyn', 24, used), 'name': name + ' ' + topic.title().replace('-', ' '),
                             'topic': topic, 'affinity': rng.gauss(0, 0.9), 'popularity': rng.lognormvariate(0, 1.2),
                             'subscribed': rng.random() < 0.5})

    history, watch_later, feeds, progress, metadata = [], [], [], [], {}
    for channel in channel_rows:
        words = TOPICS[channel['topic']]
        for _ in range(rng.randint(*videos_per_channel)):
            video_id = fake_id(rng, 'syn', 11, used)
            a, b = rng.sample(words, 2)
            title = rng.choice(TEMPLATES).format(a=a, b=b)
            title = title[0].upper() + title[1:]
            description = '. '.join([rng.choice(FILLER) + ' ' + ' '.join(rng.sample(words, 3))
                                     for _ in range(rng.randint(1, 4))]) + '.'
            is_short = rng.random() < 0.08
            duration = rng.randint(15, 59) if is_short else int(min(4 * 3600, max(61, rng.lognormvariate(6.6, 0.8))))
            published = COLLECTED - timedelta(days=rng.uniform(1, 420))
            age_days = (COLLECTED - published).days + 1
            views = int(channel['popularity'] * rng.lognormvariate(9, 1.1) * min(1, age_days / 30))
            stats = {'viewCount': str(views)}
            if rng.random() > 0.01:
                stats['likeCount'] = str(int(views * rng.uniform(0.01, 0.06)))
            metadata[video_id] = {
                'id': video_id, 'fetched_at': COLLECTED.isoformat().replace('+00:00', 'Z'),
                'snippet': {'title': title, 'description': description, 'channelId': channel['id'],
                            'channelTitle': channel['name'], 'publishedAt': published.strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'thumbnails': {}},
                'contentDetails': {'duration': iso_duration(duration)}, 'statistics': stats}
            evidence = ['synthetic:' + ('short' if is_short else 'video')]
            score = (intercept + topic_affinity[channel['topic']] + channel['affinity']
                     + 0.35 * (math.log10(views + 1) - 4) - 0.6 * abs(math.log(duration / favourite_length))
                     + rng.gauss(0, 0.4))
            watched = rng.random() < sigmoid(score)
            if watched:
                for _ in range(rng.choice([1, 1, 1, 2, 3])):
                    day = published.date() + timedelta(days=rng.randint(0, max(0, age_days - 1)))
                    history.append({'video_id': video_id, 'local_date': min(day, COLLECTED.date()).isoformat(),
                                    'is_short': None if rng.random() < 0.02 else is_short,
                                    'format_evidence': evidence, 'title': title,
                                    'channel_id': channel['id'], 'channel_name': channel['name']})
                if rng.random() < 0.3:
                    progress.append({'video_id': video_id, 'observed_at': COLLECTED.isoformat().replace('+00:00', 'Z'),
                                     'progress_bar_start_percent_raw': rng.randint(10, 100),
                                     'method': 'synthetic progress bar'})
            if rng.random() < 0.05:
                watch_later.append({'video_id': video_id})
            if rng.random() < (0.3 if watched else 0.9):
                sources = ['subscriptions'] if channel['subscribed'] else ['home']
                if channel['subscribed'] and rng.random() < 0.1:
                    sources.append('home')
                seen = COLLECTED - timedelta(hours=rng.uniform(1, 72))
                feeds.append({'video_id': video_id, 'is_short': is_short, 'source_feeds': sources,
                              'first_seen_at': seen.strftime('%Y-%m-%dT%H:%M:%SZ'), 'format_evidence': evidence,
                              'title': title, 'channel_id': channel['id'], 'channel_name': channel['name'],
                              'duration_seconds': duration})
                if not watched and rng.random() < 0.01:
                    progress.append({'video_id': video_id, 'observed_at': COLLECTED.isoformat().replace('+00:00', 'Z'),
                                     'resume_start_seconds_raw': rng.randint(5, duration), 'duration_seconds': duration,
                                     'method': 'synthetic resume offset'})
    return {'source': {'generator': 'make_synthetic.py', 'seed': seed, 'synthetic': True,
                       'collected_at': COLLECTED.isoformat().replace('+00:00', 'Z')},
            'history': history, 'watch_later': watch_later, 'feeds': feeds, 'progress': progress,
            'metadata': metadata}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('data/sample/inputs.json.gz'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--channels', type=int, default=120)
    args = parser.parse_args()
    data = generate(seed=args.seed, channels=args.channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    args.output.write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps({'output': str(args.output), 'videos': len(data['metadata']),
                      'history_events': len(data['history']), 'feed_cards': len(data['feeds']),
                      'collected_on': date.isoformat(COLLECTED.date())}))


if __name__ == '__main__':
    main()
