import pandas as pd
import pytest

from tweetmood.data.loaders import (
    SchemaError,
    load_many,
    load_sentiment140,
    load_social_sentiments,
    load_tweeteval,
    validate,
)
from tweetmood.data.synthetic import make_tweets
from tweetmood.data.split import deduplicate, fingerprint, load_split, make_split, save_split


def _s140(tmp_path, n=40):
    rows = []
    for i in range(n):
        target = "0" if i % 2 else "4"
        rows.append(f'"{target}","{i}","Mon Apr 06","NO_QUERY","user{i}","tweet number {i} is here"')
    rows.append('"4","999","d","NO_QUERY","x","tweet number 0 is here"')  # exact duplicate
    rows.append('"2","1000","d","NO_QUERY","x","a neutral line"')
    path = tmp_path / "s140.csv"
    path.write_text("\n".join(rows), encoding="latin-1")
    return path


def test_sentiment140_sample_is_balanced_and_without_replacement(tmp_path):
    # reference problem 6: sampling with replacement put the same tweet in train and test
    df, report = load_sentiment140(_s140(tmp_path), sample=20, seed=1)
    assert len(df) == 20 and not df["text"].duplicated().any()
    assert df["label"].value_counts().to_dict() == {"negative": 10, "positive": 10}
    assert report.dropped == {"label_not_binary": 1, "exact_duplicate": 1}
    assert "user" not in " ".join(df.columns)


def test_social_set_keeps_only_exact_binary_labels(tmp_path):
    # reference problem 7: the "recent" set has about 190 emotion names, and few rows are binary
    path = tmp_path / "social.csv"
    pd.DataFrame({"Text": ["a", "b", "c", "d"], "Sentiment": [" Positive ", "Joy", "Negative", "Neutral"],
                  "User": ["u1", "u2", "u3", "u4"]}).to_csv(path, index=False)
    df, report = load_social_sentiments(path)
    assert df["label"].tolist() == ["positive", "negative"]
    assert set(df["source"]) == {"social_recent"} and report.dropped == {"label_not_binary": 2}


def test_tweeteval_drops_neutral(tmp_path):
    for part in ("train", "val"):
        (tmp_path / f"{part}_text.txt").write_text("good\nok\nbad\n", encoding="utf-8")
        (tmp_path / f"{part}_labels.txt").write_text("2\n1\n0\n", encoding="utf-8")
    df, _ = load_tweeteval(tmp_path)
    assert df["label"].tolist() == ["positive", "negative"] * 2


def test_load_many_rejects_bad_specs(tmp_path):
    with pytest.raises(ValueError):
        load_many(["imdb=x.csv"])
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"text": ["x"], "label": ["neutral"], "source": ["s"]}))


def test_deduplicate_drops_conflicts():
    df = pd.DataFrame({"text": ["Great :)", "great :)", "bad day", "BAD DAY!", "fine"],
                       "label": ["positive", "positive", "negative", "positive", "positive"], "source": "s"})
    out, rep = deduplicate(df)
    assert rep == {"rows_in": 5, "conflicting_rows": 2, "duplicates": 1, "rows_out": 2}
    assert sorted(out["text"]) == ["Great :)", "fine"]


def test_one_shared_split(split_df):
    # reference problem 1: the prototype used seed 42 for one model family and seed 52 for the other
    again = make_split(make_tweets(n=2500, seed=3), seed=3)
    assert fingerprint(again) == fingerprint(split_df)
    assert fingerprint(make_split(make_tweets(n=2500, seed=3), seed=4)) != fingerprint(split_df)
    assert not set(split_df[split_df.split == "train"]["id"]) & set(split_df[split_df.split == "test"]["id"])
    shares = split_df["split"].value_counts(normalize=True)
    assert 0.15 < shares["test"] < 0.25 and 0.05 < shares["val"] < 0.15


def test_holdout_source_goes_to_test_only(split_df):
    df = make_split(split_df[["text", "label", "source"]], seed=3, holdout_sources=("synthetic_genz",))
    assert set(df[df.source == "synthetic_genz"]["split"]) == {"test"}
    assert set(df[df.split == "train"]["source"]) == {"synthetic_classic"}


def test_saved_split_detects_changes(saved_split):
    df, manifest, root = saved_split
    back, m2 = load_split(root / "prepared")
    assert m2["fingerprint"] == manifest["fingerprint"] == fingerprint(back)
    path = root / "prepared" / "split.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",train\n", ",test\n", 1), encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(root / "prepared")
