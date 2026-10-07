"""Hybrid tests. They skip when torch is not installed (CI installs only the dev extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tweetmood.data.split import part  # noqa: E402
from tweetmood.models import build_model, load_model  # noqa: E402
from tweetmood.models.hybrid import HEADS, HashEmbedder, _batch  # noqa: E402

LAB = {"negative": 0, "positive": 1}


def _xy(df):
    return df["text"].tolist(), np.array([LAB[v] for v in df["label"]])


def test_head_learning_rate_and_early_stopping_defaults():
    # reference problem 3: a frozen encoder with a fine-tuning lr (2e-5) left the heads under-trained
    model = build_model("hybrid-bilstm")
    assert model.lr == pytest.approx(1e-3) and model.epochs >= 10 and model.patience >= 1


def test_hash_embedder_is_deterministic():
    a = HashEmbedder(dim=16).embed(["good day"])[0]
    b = HashEmbedder(dim=16).embed(["good day"])[0]
    assert a.shape == (2, 16) and torch.equal(a, b)


@pytest.mark.parametrize("kind", ["bilstm", "transformer"])
def test_heads_ignore_padding(kind):
    torch.manual_seed(0)
    states = [torch.randn(5, 16), torch.randn(2, 16)]
    head = HEADS[kind](16).eval()
    x, m = _batch(states, [0, 1])
    wide = torch.cat([x, torch.randn(2, 4, 16) * 30], dim=1)
    wide_mask = torch.cat([m, torch.zeros(2, 4, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(head(x, m), head(wide, wide_mask), atol=1e-5)


def test_hybrid_trains_keeps_best_epoch_and_roundtrips(split_df, tmp_path):
    tr_x, tr_y = _xy(part(split_df, "train"))
    va_x, va_y = _xy(part(split_df, "val"))
    model = build_model("hybrid-bilstm", epochs=4, patience=1).fit(tr_x, tr_y, va_x, va_y)
    assert model.best_epoch == int(np.argmax(model.history))
    proba = model.predict_proba(va_x)
    assert proba.shape == (len(va_x), 2) and np.allclose(proba.sum(1), 1, atol=1e-5)
    model.save(tmp_path / "h")
    again = load_model(tmp_path / "h")
    assert np.allclose(again.predict_proba(va_x), proba, atol=1e-5)
