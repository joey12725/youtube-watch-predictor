# Methodology

## The task

Given the metadata a recommender would have at serving time (channel, title, description, duration, view and like counts), predict whether a viewer watches a video that was shown to them. The benchmark measures **recorded-watch discrimination**: separating videos that appear in watch history from videos that were shown in a feed and never appear in it.

## Labels

| Label | Rule |
|---|---|
| `watched=true` | At least one recorded watch event (optionally on or after `--cutoff`) whose format is explicitly a regular video, not a Short. |
| `watched=false` | A non-Short card from the Home or Subscriptions feed that is absent from **all** collected history (including watches before the cutoff and unknown-format entries), has no positive progress or resume observation, and has no "WATCHED" badge. |

A negative is a weak negative: an impression the viewer did not act on during the collection window, not proof they disliked it. Any explicit Short evidence for a video ID, from history or feed, removes it from both classes.

`percentage_watched` is always blank because the sources do not expose cumulative watch time. UI progress observations are kept in separately named proxy columns (`watch_percentage_proxy`, `watch_percentage_proxy_method`, `progress_observed_at`). The latest numeric observation wins; a resume offset is divided by the duration captured with that same observation.

## Datasets

`build_dataset.py` writes two CSVs:

- `all_available.csv.gz`: every eligible positive and negative, split 80/10/10 per label.
- `evaluation_10to1.csv.gz`: an exact 10:1 negative-to-positive sample. Each split independently holds the 10:1 ratio. When the pool cannot support 10:1 with every positive, positives are sampled and the shortfall is recorded in `dataset_report.json`.

The two CSVs are sampled independently. Train and test on the same file.

## Features and leakage control

Model inputs are limited to `channel_id`, `video_title`, `description`, `view_count`, `like_count` and `duration_seconds`. `video_id` is a join key only.

These columns are never model inputs: `watched`, `percentage_watched`, every progress proxy, watch dates and counts, `label_basis`, `split`, `source_feeds`, `first_feed_seen_at`, `in_watch_later` and collection timestamps. They are either the label, derived from it, or artifacts of how each class was collected.

Upload age is the subtle one. Positives are watches gathered over months; negatives are feed cards scraped near collection time, so negatives skew newer for reasons unrelated to preference. `benchmark.py` trains an **age-only probe** to quantify that shortcut (0.59 AUC on the private export) and keeps age out of every reported model.

Metadata is a collection-time snapshot, so view and like counts for older watches include views accumulated after the watch.

## Protocols

| Protocol | How test is chosen | What it answers |
|---|---|---|
| `random` | The `split` column written by the builder | How well can we rank unseen videos, including from familiar channels? |
| `channel-holdout` | `crc32(channel_id) % 10 == 0` is test, `== 1` is validation | How well does the model generalize to channels it has never seen? |

The gap between the two is the headline diagnostic: on the private export, the best model drops from 0.919 to 0.622 AUC when channels are held out.

## Training and scoring

1. Fit on train.
2. Pick the decision threshold that maximizes F1 on validation (ties in score are treated as one threshold).
3. Score test once with `evaluate.metrics` at that threshold.
4. Bootstrap the test set (default 500 resamples, 1,000 for the published results, seeded) for 95% intervals on ROC AUC and average precision.

ROC AUC is threshold-free and tie-aware. Average precision is the better headline under 10:1 imbalance because its chance level is the base rate (0.091), not 0.5. Accuracy is reported but not useful here: predicting "not watched" for everything already scores 90.9%.

## Models

| Name | Description |
|---|---|
| `prior` | Constant training positive rate. Floor for every metric. |
| `channel-rate` | `(positives_c + k * p) / (n_c + k)` with `k = 4`; unseen channels get the global rate `p`. |
| `naive-bayes-text` | Multinomial NB over title tokens and the first 80 description tokens (as sets), Laplace smoothing. |
| `logreg-meta` | Logistic regression on standardized log views, log likes, like rate, log duration, duration buckets and missing-value flags. |
| `logreg-text` | Logistic regression on title and description token indicators (vocabulary from train, min count 2). |
| `logreg-channel-meta` | Channel one-hot plus the metadata features. |
| `logreg-all` | Channel, metadata and text together. |

Logistic regression uses Adagrad (learning rate 0.1, L2 1e-4, 12 epochs, seeded shuffle). Without class weighting the probabilities stay calibrated, which keeps the Brier score meaningful; the validation-tuned threshold handles the imbalance.

## Extending

- **New model:** add a class with `fit(rows, labels)` and `predict(rows) -> list[float]` and register it in `MODELS`. It is picked up by both protocols, the table and the JSON report.
- **New features:** add a group to `featurize`. Prefix numeric keys with `n:` to get train-set standardization for free.
- **External models:** write `video_id,score` for one split and run `evaluate.py`. Tune on validation, report test once.
- **New protocol:** add a branch to `assign_splits`. A temporal split by `first_watch_date` / `first_feed_seen_at` is a natural next one.
