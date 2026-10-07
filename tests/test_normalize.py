import time

from tweetmood.text.normalize import Normalizer, emoji_name, has_negation, has_slang, mark_negation

N = Normalizer()


def test_negations_are_kept_and_expanded():
    # reference problem 2: stop-word removal turned "not good" into "good"
    assert N("This is not good") == "this is not good"
    assert N("I dont like it") == "i do not like it"
    assert N("it isn’t fun") == "it is not fun"
    assert "never" in N("never again")


def test_emoticons_are_mapped_before_lower_casing():
    # reference problem 8: lower-casing first turned ":D" into ":d", which no longer matched
    assert "emo_laugh" in N("great day :D")
    assert "emo_laugh" in N("XD that was funny")
    assert "emo_sad" in N("missed the bus :(")
    assert "emo_heart" in N("love you <3")
    assert "emo" not in N("AND: the end")  # no false emoticon inside a word


def test_emoji_names_keep_underscores_and_drop_modifiers():
    # reference problem 8: the symbol strip merged "smiling_face" into "smilingface"
    out = N("yes \U0001F44D\U0001F3FD and \U0001F602")
    assert "emoji_thumbs_up_sign" in out.split()
    assert "emoji_face_with_tears_of_joy" in out.split()
    assert emoji_name("\U0001F3FD") == "" and emoji_name("a") is None


def test_slang_hashtags_and_case_sensitive_letters():
    assert N("this album is mid ngl") == "this album is mediocre honestly"
    assert N("#NoCap that was bussin") == "honestly that was excellent"
    assert N("huge W today") == "huge win today"
    assert N("w/o sugar") == "without sugar"
    assert has_slang("lowkey cringe") and not has_slang("a calm evening")


def test_urls_mentions_and_letter_runs():
    assert N("@bob sooooo good!!!! https://t.co/x") == "@user soo good ! ! url"


def test_negation_scope_stops_at_punctuation():
    assert mark_negation("not good at all . fine") == "not NEG_good NEG_at NEG_all . fine"
    assert Normalizer(negation_scope=True)("I don't love it, ok") == "i do not NEG_love NEG_it , ok"
    assert has_negation("I dont care") and not has_negation("I care")


def test_flags_and_fingerprint():
    assert Normalizer(slang=False)("so mid") == "so mid"
    assert Normalizer(emoji=False)("\U0001F525") == "\U0001F525"
    assert Normalizer().fingerprint() != Normalizer(slang=False).fingerprint()


def test_normalizer_is_fast():
    # one compiled pattern per map, not hundreds of substitutions per tweet
    texts = ["ngl this update is sooo mid :( \U0001F644 #NoCap @user https://x.y"] * 5000
    start = time.perf_counter()
    for t in texts:
        N(t)
    assert time.perf_counter() - start < 5.0
