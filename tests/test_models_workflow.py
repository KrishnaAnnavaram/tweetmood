import json

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import f1_score

from tweetmood.cli import main
from tweetmood.data.split import part
from tweetmood.evaluate import bootstrap_ci, macro_f1, mcnemar, scores
from tweetmood.models import build_model, load_model
from tweetmood.models.lexicon import LexiconModel
from tweetmood.workflow import ablate, evaluate_models, predict, train_models

LAB = {"negative": 0, "positive": 1}


def _xy(df):
    return df["text"].tolist(), np.array([LAB[v] for v in df["label"]])


def test_macro_f1_matches_sklearn_and_ci():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))
    lo, hi = bootstrap_ci(y, p, n_boot=200)
    assert lo <= macro_f1(y, p) <= hi
    out = scores(y, np.column_stack([1 - p, p]).astype(float), n_boot=50)
    assert out["n"] == 300 and "roc_auc" in out


def test_mcnemar_counts_disagreements():
    y = np.array([1, 1, 1, 1, 0, 0])
    res = mcnemar(y, [1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0])
    assert res["only_a_correct"] == 3 and res["only_b_correct"] == 0 and res["p_value"] == pytest.approx(0.25)


def test_tfidf_stays_sparse_and_normalizes_inside_the_pipeline(split_df):
    # reference problem 9: .toarray() made a dense 2.4 GB matrix
    tr_x, tr_y = _xy(part(split_df, "train"))
    va_x, va_y = _xy(part(split_df, "val"))
    model = build_model("tfidf-logreg").fit(tr_x, tr_y, va_x, va_y)
    steps = model.pipeline.named_steps
    assert list(steps) == ["normalize", "tfidf", "clf"]
    assert sp.issparse(steps["tfidf"].transform(steps["normalize"].transform(va_x[:5])))
    assert model.selected["strength"] in (0.3, 1.0, 3.0, 10.0)


@pytest.mark.parametrize("name", ["lexicon", "tfidf-logreg", "tfidf-svm", "tfidf-nb"])
def test_save_load_roundtrip(name, split_df, tmp_path):
    tr_x, tr_y = _xy(part(split_df, "train"))
    va_x, va_y = _xy(part(split_df, "val"))
    model = build_model(name).fit(tr_x, tr_y, va_x, va_y)
    model.save(tmp_path / name)
    again = load_model(tmp_path / name)
    assert np.allclose(model.predict_proba(va_x), again.predict_proba(va_x))
    assert macro_f1(va_y, model.predict_proba(va_x).argmax(1)) > 0.7


def test_lexicon_flips_negated_words():
    lex = LexiconModel()
    assert lex.score("this is good") > 0 > lex.score("this is not good")
    assert lex.score("this album is mid") < 0  # through the slang map
    assert lex.predict_proba(["great :)"])[0, 1] > 0.5


def test_best_model_is_selected_on_validation_among_trained_models(saved_split):
    # reference problem 4: the documented best model was not in the run
    df, manifest, root = saved_split
    reg = train_models(df, manifest, ["lexicon", "tfidf-nb"], root / "models", log=lambda m: None)
    assert set(reg["models"]) == {"lexicon", "tfidf-nb"}
    assert reg["best"] == max(reg["models"], key=lambda n: reg["models"][n]["val_macro_f1"])
    assert reg["selected_on"] == "val_macro_f1"
    reg2 = train_models(df, manifest, ["tfidf-logreg"], root / "models", log=lambda m: None)
    assert set(reg2["models"]) == {"lexicon", "tfidf-nb", "tfidf-logreg"}


def test_evaluate_uses_one_test_split_for_all_models(saved_split):
    df, manifest, root = saved_split
    train_models(df, manifest, ["lexicon", "tfidf-logreg"], root / "models", log=lambda m: None)
    report = evaluate_models(df, manifest, root / "models", n_boot=50)
    ns = {m["test"]["n"] for m in report["models"].values()}
    assert ns == {int((df["split"] == "test").sum())}
    other = [n for n in report["models"] if n != report["best"]]
    assert set(report["mcnemar_vs_best"]) == set(other)
    assert (root / "models" / "evaluation.md").exists()
    with pytest.raises(ValueError):
        evaluate_models(df, dict(manifest, fingerprint="other"), root / "models")


def test_predict_uses_the_selected_model_and_the_same_normalizer(saved_split):
    # reference problem 5: the demo used the last model of a loop and skipped a training step
    df, manifest, root = saved_split
    reg = train_models(df, manifest, ["lexicon", "tfidf-logreg"], root / "models", log=lambda m: None)
    texts = ["this update is sooo mid :(", "the concert was bussin \U0001F525"]
    rows = predict(texts, root / "models")
    assert {r["model"] for r in rows} == {reg["best"]}
    direct = load_model(root / "models" / reg["best"]).predict_proba(texts)[:, 1]
    assert [r["p_positive"] for r in rows] == [round(float(p), 4) for p in direct]
    with pytest.raises(FileNotFoundError):
        predict(texts, root / "nothing")


def test_ablation_shows_the_cost_of_stopword_removal(split_df):
    table = ablate(split_df).set_index("variant")
    assert table.loc["full", "has_negation"] > table.loc["stopword-removal", "has_negation"]
    assert set(table.index) == {"full", "no-negation-scope", "no-slang", "no-emoji-emoticons", "stopword-removal"}


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TWEETMOOD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TWEETMOOD_MODEL_DIR", str(tmp_path / "models"))
    assert main(["synth", "--n", "800"]) == 0
    raw = tmp_path / "data" / "raw" / "synthetic_tweets.csv"
    assert main(["prepare", "--source", f"unified={raw}"]) == 0
    assert main(["train", "--model", "lexicon", "--model", "tfidf-nb"]) == 0
    assert main(["evaluate", "--n-boot", "20"]) == 0
    capsys.readouterr()
    assert main(["predict", "so good :)"]) == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["label"] in ("positive", "negative")
    assert main(["normalize", "not mid", "--negation-scope"]) == 0
    assert capsys.readouterr().out.strip() == "not NEG_mediocre"
    with pytest.raises(SystemExit):
        main(["train", "--model", "bert"])
