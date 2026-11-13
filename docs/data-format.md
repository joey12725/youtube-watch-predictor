# Data format

## Input: `inputs.json.gz`

One gzipped JSON object. `make_synthetic.py` writes a complete example; run it and open the result to see every field populated.

```json
{
  "source": {"anything": "copied verbatim into dataset_report.json"},
  "history": [...],
  "watch_later": [...],
  "feeds": [...],
  "progress": [...],
  "metadata": {"<video_id>": {...}}
}
```

Every `video_id` must be an 11-character YouTube ID (`[A-Za-z0-9_-]{11}`); the builder rejects anything else.

### `history`: one entry per watch event

| Field | Type | Notes |
|---|---|---|
| `video_id` | string | |
| `local_date` | `YYYY-MM-DD` | Date in the viewer's time zone. Compared against `--cutoff`. |
| `is_short` | `true` / `false` / `null` | `null` means unknown; unknown watches are never positives but still block negatives. |
| `format_evidence` | list of strings | Free-text provenance for the format decision. |
| `title`, `channel_id`, `channel_name`, `thumbnail_url`, `duration_seconds` | optional | Fallbacks when `metadata` lacks a value. |

Repeat watches are separate entries; the CSV reports first and last date and the event count.

### `watch_later`

`[{"video_id": "..."}]`, a snapshot of the playlist. Becomes the `in_watch_later` column.

### `feeds`: one entry per distinct card seen

| Field | Type | Notes |
|---|---|---|
| `video_id` | string | |
| `is_short` | bool or `null` | Only `false` cards can become negatives. |
| `source_feeds` | list | `home`, `subscriptions` or both. |
| `first_seen_at` | ISO timestamp | |
| `format_evidence`, `title`, `channel_id`, `channel_name`, `duration_seconds` | optional | |

### `progress`: UI progress or resume observations

| Field | Notes |
|---|---|
| `video_id`, `observed_at`, `method` | |
| `percent_duration_watched_raw` or `progress_bar_start_percent_raw` | 0-100 |
| `resume_start_seconds_raw` + `duration_seconds` | Converted to a percentage of that duration |
| `playback_status_label_raw` | A value containing `WATCHED` disqualifies the video as a negative |

### `metadata`

Keyed by video ID, each value shaped like a YouTube Data API v3 `videos.list` item with `part=snippet,contentDetails,statistics`, plus a `fetched_at` timestamp:

```json
{
  "id": "synAb3dE5fG",
  "fetched_at": "2026-01-15T18:00:00Z",
  "snippet": {"title": "...", "description": "...", "channelId": "UC...", "channelTitle": "...",
              "publishedAt": "2025-11-02T14:00:00Z", "thumbnails": {"high": {"url": "..."}}},
  "contentDetails": {"duration": "PT12M34S"},
  "statistics": {"viewCount": "48210", "likeCount": "1630"}
}
```

Missing `likeCount` (hidden likes) stays blank in the CSV rather than becoming zero.

### Where the pieces come from

- **Watch history:** Google Takeout's `watch-history.json` provides video IDs and timestamps. Takeout does not mark Shorts, so you need a format signal (for example the `/shorts/` URL form or duration from metadata) to set `is_short`.
- **Feeds:** cards on the Home and Subscriptions pages. Any browser-side collector works, as long as it records the video ID, the feed and whether the card was a Short.
- **Metadata:** `videos.list` from the YouTube Data API, batched 50 IDs per call.

## Output CSV columns

Blank cells mean missing or not applicable, never zero. Descriptions contain commas, quotes and newlines, so read with a real CSV parser.

| Column | Notes |
|---|---|
| `video_id`, `url`, `channel_id`, `channel_name`, `video_title`, `description`, `thumbnail_url`, `published_at` | From metadata, falling back to the export |
| `view_count`, `like_count`, `duration_seconds` | Integers from the metadata snapshot |
| `watched` | `true` / `false`, the label |
| `percentage_watched` | Always blank (not exposed by the sources) |
| `in_watch_later` | `true` / `false` from the playlist snapshot |
| `watch_percentage_proxy`, `watch_percentage_proxy_method`, `progress_observed_at` | Latest UI progress observation |
| `first_watch_date`, `last_watch_date`, `watch_event_count` | Watch events in the window |
| `source_feeds`, `first_feed_seen_at` | Feed provenance, pipe-delimited |
| `metadata_status`, `metadata_fetched_at` | `available` / `unavailable` |
| `label_basis`, `format_evidence` | Why the row has its label |
| `split` | `train` / `validation` / `test` |

Only the first block (identity and content metadata) belongs in a model. See [methodology.md](methodology.md#features-and-leakage-control).
