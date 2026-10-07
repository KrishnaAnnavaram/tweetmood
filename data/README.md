# data/

Git ignores everything in this folder except this file. Do not commit tweets, splits or models. The
offline demo and the tests use the synthetic generator (`tweetmood synth`) and need no download.

## Supported sources

| KIND | Corpus | Where to get it | Expected file and columns | Label mapping |
|---|---|---|---|---|
| `sentiment140` | Sentiment140 (1.6 M tweets, 2009) | Kaggle `kazanova/sentiment140` | `training.1600000.processed.noemoticon.csv`, latin-1, no header, six columns: `target, id, date, flag, user, text` | `0` negative, `4` positive. Only `target` and `text` are read |
| `tweeteval` | TweetEval sentiment | GitHub `cardiffnlp/tweeteval`, folder `datasets/sentiment` | `train_text.txt`, `train_labels.txt`, `val_*`, `test_*` | `0` negative, `2` positive. `1` (neutral) is dropped |
| `social` | Social Media Sentiments Analysis Dataset (732 rows) | Kaggle `kashishparmar02/social-media-sentiments-analysis-dataset` | `sentimentdataset.csv` with `Text, Sentiment` | Only rows with exactly `Positive` or `Negative` (about 49 rows). Other emotion names are dropped |
| `unified` | Your own CSV | - | `text, label` and optional `source` | `negative`, `positive` |

Read the license and terms of each dataset before you use it. Sentiment140 and TweetEval are for
research use. Tweets can contain personal data: do not publish raw tweets or user names. The loaders
never read the user-name columns.

## Prepare the shared split

```bash
tweetmood prepare --source sentiment140=data/raw/training.1600000.processed.noemoticon.csv --sample 30000 \
                  --source social=data/raw/sentimentdataset.csv --holdout-source social_recent
```

`--sample` takes a class-balanced sample WITHOUT replacement after exact duplicates are removed.
`--holdout-source` puts a whole source in the test split, so it measures transfer to newer tweets.

## Files that `prepare` writes

| File | Contents |
|---|---|
| `data/prepared/split.csv` | `id, text, label, source, split` (`split` is `train`, `val` or `test`) |
| `data/prepared/manifest.json` | fingerprint, rows per split, labels and sources per split, de-duplication counts, seed |
