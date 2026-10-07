import pytest

from tweetmood.data.split import make_split, save_split
from tweetmood.data.synthetic import make_tweets


@pytest.fixture(scope="session")
def split_df():
    return make_split(make_tweets(n=2500, seed=3), seed=3)


@pytest.fixture()
def saved_split(split_df, tmp_path):
    manifest = save_split(split_df, tmp_path / "prepared")
    return split_df, manifest, tmp_path
