"""Settings from environment variables. All values have offline-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models_out")
    seed: int = 42
    device: str = "auto"
    hf_model: str = "cardiffnlp/twitter-roberta-base"
    hf_offline: bool = False


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    raw_seed = env.get("TWEETMOOD_SEED", "42")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"TWEETMOOD_SEED must be an integer, got {raw_seed!r}") from exc
    device = (env.get("TWEETMOOD_DEVICE") or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("TWEETMOOD_DEVICE must be auto, cpu or cuda")
    return Settings(
        data_dir=Path(env.get("TWEETMOOD_DATA_DIR") or "data"),
        model_dir=Path(env.get("TWEETMOOD_MODEL_DIR") or "models_out"),
        seed=seed,
        device=device,
        hf_model=env.get("TWEETMOOD_HF_MODEL") or "cardiffnlp/twitter-roberta-base",
        hf_offline=(env.get("TWEETMOOD_HF_OFFLINE") or "").strip().lower() in {"1", "true", "yes"},
    )
