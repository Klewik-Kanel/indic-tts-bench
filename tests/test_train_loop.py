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


# --- live progress ----------------------------------------------------------

def test_progress_writes_whole_lines_when_not_a_terminal():
    """A nohup log full of carriage returns is unreadable and unsearchable."""
    import io
    from src.train.progress import Progress
    buf = io.StringIO()
    p = Progress("r01", 1000, stream=buf, file_every=250)
    for s in range(1, 1001):
        p.update(s, 1.0, 2e-4)
    p.close(1.0)
    out = buf.getvalue()
    assert "\r" not in out
    assert out.count("\n") == len([l for l in out.splitlines() if l])
    assert "step       1/1,000" in out
    assert "step   1,000/1,000" in out


def test_progress_reports_every_file_every_steps_and_the_ends():
    import io
    from src.train.progress import Progress
    buf = io.StringIO()
    p = Progress("r01", 1000, stream=buf, file_every=250)
    for s in range(1, 1001):
        p.update(s, 1.0, 2e-4)
    body = [l for l in buf.getvalue().splitlines() if "step" in l]
    assert len(body) == 5          # step 1, then 250, 500, 750, 1000


def test_elapsed_formatting_survives_a_nan_estimate():
    from src.train.progress import _hms
    assert _hms(float("nan")) == "--:--:--"
    assert _hms(3661) == "1:01:01"
    assert _hms(-5) == "0:00:00"


# --- the export bundle ------------------------------------------------------

def test_bundle_audio_params_match_the_batching_module():
    """A vocoder fed mels on a different hop makes audio that is subtly wrong,
    which nobody notices in a demo room."""
    from src.export.bundle import AUDIO
    from src.train.batching import HOP_LENGTH
    assert AUDIO["hop_length"] == HOP_LENGTH


# --- the coqui adapters' shared per-step work -------------------------------

def test_both_coqui_adapters_derive_mel_once_per_step():
    """format_batch_on_device must run for BOTH coqui models, exactly once.

    Written after it did not. With `prepare` defined on FastSpeech2Adapter only,
    VITS inherited the no-op default; its discriminator branch ran regardless,
    because that branch indexes only keys collate supplies, and its generator
    branch died on KeyError: 'mel'. The failure needed a GPU and four steps to
    appear. This test needs neither, and no upstream package either.
    """
    from src.train import adapters

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def format_batch_on_device(self, t):
            self.calls += 1
            return {**t, "mel": "derived"}

    for cls in (adapters.FastSpeech2Adapter, adapters.VitsAdapter):
        model = StubModel()
        out = cls().prepare(model, {"spec": 1})
        assert model.calls == 1, f"{cls.__name__} did not derive mel"
        assert "mel" in out, f"{cls.__name__} returned a batch without mel"


def test_the_base_adapter_prepare_is_a_no_op():
    """Only the coqui models need per-step batch work; the toy must not."""
    from src.train import adapters

    t = {"ids": 1}
    assert adapters.AdapterBase().prepare(object(), t) is t
