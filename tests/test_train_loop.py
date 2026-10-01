"""Tests for the parts of training that can be wrong without a GPU.

Everything here runs with no deep-learning framework installed, which is
deliberate: these are the failures that waste a Kaggle session, and finding
them should not require one.
"""

import math
import pathlib

import pytest

from src.train import batching, schedule
from src.train.checkpoint import CheckpointDir, RetentionPolicy
from src.train.runner import load_config
from src.train.text import TextEncoder, Vocab


# --- the learning-rate schedule --------------------------------------------

def test_schedule_is_continuous_at_the_warmup_join():
    """A jump here is a silent order-of-magnitude change mid-training."""
    peak, warm = 2e-4, 4000
    assert schedule.lr_at(warm, peak, warm) == pytest.approx(peak)
    assert schedule.lr_at(warm + 1, peak, warm) == pytest.approx(peak, rel=1e-3)


def test_schedule_rises_then_falls():
    peak, warm = 2e-4, 4000
    rising = [schedule.lr_at(s, peak, warm) for s in (1, 100, 1000, warm)]
    falling = [schedule.lr_at(s, peak, warm) for s in (warm, 10_000, 50_000, 100_000)]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)


def test_schedule_decay_matches_the_closed_form():
    assert schedule.lr_at(100_000, 2e-4, 4000) == pytest.approx(
        2e-4 * math.sqrt(4000 / 100_000))


def test_step_zero_is_rejected():
    with pytest.raises(ValueError):
        schedule.lr_at(0, 2e-4, 4000)


# --- batching ---------------------------------------------------------------

def _utts(n, seconds=3.0, sr=22050):
    return [batching.Utterance(f"u{i}", f"w/{i}.wav", seconds, "क", 
                               batching.frames_for(seconds, sr)) for i in range(n)]


def test_frames_round_up():
    """A partial final frame is still computed, so the budget must include it."""
    assert batching.frames_for(1.0, 22050) == math.ceil(22050 / 256)


def test_no_batch_exceeds_the_padded_frame_budget():
    """The budget bounds what the GPU computes, which is len(batch) * max_frames."""
    utts = [batching.Utterance(f"u{i}", "w", 1 + (i % 9), "क",
                               batching.frames_for(1 + (i % 9), 22050))
            for i in range(400)]
    for b in batching.make_batches(utts, 12_000, seed=0, epoch=0):
        assert len(b) * max(u.frames for u in b) <= 12_000 or len(b) == 1


def test_batching_is_a_pure_function_of_seed_and_epoch():
    utts = _utts(200)
    a = batching.make_batches(utts, 12_000, seed=7, epoch=3)
    b = batching.make_batches(utts, 12_000, seed=7, epoch=3)
    assert [[u.uid for u in x] for x in a] == [[u.uid for u in x] for x in b]


def test_different_epochs_give_a_different_order():
    utts = _utts(200)
    a = batching.make_batches(utts, 12_000, seed=7, epoch=0)
    b = batching.make_batches(utts, 12_000, seed=7, epoch=1)
    assert [[u.uid for u in x] for x in a] != [[u.uid for u in x] for x in b]


def test_every_utterance_appears_exactly_once_per_epoch():
    """Dropping or repeating items would change the training set silently."""
    utts = _utts(237)
    seen = [u.uid for b in batching.make_batches(utts, 12_000, 0, 0) for u in b]
    assert sorted(seen) == sorted(u.uid for u in utts)


def test_an_oversized_utterance_is_kept_and_batched_alone():
    utts = _utts(10, seconds=2.0) + [batching.Utterance(
        "huge", "w", 600.0, "क", batching.frames_for(600.0, 22050))]
    batches = batching.make_batches(utts, 12_000, 0, 0)
    alone = [b for b in batches if b[0].uid == "huge"]
    assert len(alone) == 1 and len(alone[0]) == 1
    assert [u.uid for u in batching.oversized(utts, 12_000)] == ["huge"]


def test_resume_reproduces_the_uninterrupted_batch_sequence():
    """The property the whole resume path rests on."""
    utts = _utts(120)
    full = [[u.uid for u in b] for _, b in batching.batch_stream(utts, 12_000, 0, 60)]
    tail = [[u.uid for u in b] for _, b in batching.batch_stream(utts, 12_000, 0, 60, 25)]
    assert tail == full[25:]


def test_stream_crosses_epochs_rather_than_stopping():
    utts = _utts(40)
    steps = [s for s, _ in batching.batch_stream(utts, 12_000, 0, 100)]
    assert steps == list(range(100))


# --- the vocabulary ---------------------------------------------------------

def test_pad_is_zero_in_every_vocabulary():
    for repr_ in ("phoneme", "grapheme"):
        assert Vocab.build(repr_).ids["<pad>"] == 0


def test_vocabulary_does_not_depend_on_the_corpus():
    """The 10-minute rung and the 9-hour rung must share one embedding table."""
    assert Vocab.build("phoneme").symbols == Vocab.build("phoneme").symbols


def test_an_unknown_symbol_raises_rather_than_becoming_unk():
    v = Vocab.build("phoneme")
    with pytest.raises(KeyError):
        v.encode(["☃"])


def test_vocabulary_round_trips_through_disk(tmp_path):
    v = Vocab.build("grapheme")
    v.save(tmp_path / "vocab.json")
    assert Vocab.load(tmp_path / "vocab.json").symbols == v.symbols


def test_the_two_arms_differ_only_in_representation():
    """Both arms must run through the same normaliser, or the ablation
    measures the normaliser rather than the input representation."""
    p = TextEncoder.for_config("hindi", "phoneme")
    g = TextEncoder.for_config("hindi", "grapheme")
    text = "नमकीन कमल।"
    assert p.tokens(text) != g.tokens(text)
    assert p.g2p.language == g.g2p.language
    assert len(p.encode(text)) > 0 and len(g.encode(text)) > 0


# --- checkpoints ------------------------------------------------------------

def test_retention_keeps_recent_and_milestone_checkpoints():
    pol = RetentionPolicy(keep_last=2, keep_every=25_000)
    steps = [5_000, 10_000, 25_000, 50_000, 75_000, 80_000, 85_000]
    assert pol.survivors(steps) == [25_000, 50_000, 75_000, 80_000, 85_000]
    assert pol.to_delete(steps) == [5_000, 10_000]


def test_retention_never_deletes_everything():
    pol = RetentionPolicy(keep_last=2, keep_every=25_000)
    assert pol.survivors([100, 200]) == [100, 200]


def test_a_half_written_checkpoint_is_not_resumable(tmp_path):
    """The failure this guards against is a kill during the save itself."""
    ck = CheckpointDir(tmp_path)
    ck.save(10, lambda d: (d / "state.pt").write_bytes(b"x"), {})
    (tmp_path / "step_20").mkdir()               # no COMPLETE marker
    assert ck.steps() == [10]
    assert ck.resume_step() == 10


def test_resume_step_is_zero_on_an_empty_directory(tmp_path):
    assert CheckpointDir(tmp_path / "nothing").resume_step() == 0


def test_prune_moves_rather_than_unlinks(tmp_path):
    """Deletion is blocked inside connected folders, and an operation that
    silently fails to free space is worse than one that moves it."""
    ck = CheckpointDir(tmp_path, RetentionPolicy(keep_last=1, keep_every=0))
    for s in (10, 20, 30):
        ck.save(s, lambda d: (d / "state.pt").write_bytes(b"x"), {})
    assert ck.prune() == [10, 20]
    assert ck.steps() == [30]
    assert (tmp_path / "_trash" / "step_10").exists()


# --- configs ----------------------------------------------------------------

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def test_every_config_parses_and_carries_its_budget():
    for p in sorted(CONFIGS.glob("r*.yaml")):
        cfg = load_config(p)
        assert cfg["run_id"] == p.stem
        assert cfg["max_steps"] == 100_000
        assert cfg["batch_frames"] == 12_000
        assert cfg["lr"] == pytest.approx(2e-4)
        assert isinstance(cfg["deviations"], list)


def test_vits_configs_declare_their_deviations():
    """An undeclared architectural difference is a bug, not a detail."""
    for p in sorted(CONFIGS.glob("r*.yaml")):
        cfg = load_config(p)
        if cfg["architecture"] == "vits":
            assert cfg["sample_rate"] == 16_000
            assert any("discriminator" in d for d in cfg["deviations"])


def test_the_toy_adapter_cannot_produce_a_result():
    from src.train.adapters import assert_not_toy
    with pytest.raises(SystemExit):
        assert_not_toy({"architecture": "toy", "run_id": "rXX"})
    assert_not_toy({"architecture": "vits", "run_id": "r02"})
