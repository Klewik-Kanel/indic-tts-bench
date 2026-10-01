"""The two-optimiser path, which exists for VITS and must not change anything else.

VITS is adversarial: coqui's `Vits.train_step` takes an `optimizer_idx`, where 0
is the discriminator and 1 is the generator, and the generator branch reads
outputs the discriminator branch cached. The loop therefore has to step two
optimisers per batch, in that order, on one prepared batch.

These tests use a miniature GAN with the same shape and none of the weight, so
the machinery is tested without a GPU, an audio stack or an upstream package.
Three things can be wrong here and each is silent in a loss curve:

  - only one of the two optimisers actually steps, so half the model never trains
  - the gradient clip is computed over both halves at once, so grad_clip means
    something different in a VITS run than in a FastSpeech 2 run
  - a checkpoint stores one optimiser state and resume restarts the other's Adam
    moments from zero, which puts a transient in the loss that cannot later be
    told apart from a real effect

torch is required for these and only these; the rest of the suite still runs
without it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="the two-optimiser path needs torch")

from src.train import adapters, batching          # noqa: E402
from src.train.runner import train                # noqa: E402

CFG = {
    "run_id": "rtest",
    "architecture": "toy_gan",
    "language": "hindi",
    "input_repr": "phoneme",
    "data": "train",
    "sample_rate": 22050,
    "batch_frames": 2_000,
    "max_steps": 4,
    "lr": 1e-3,
    "warmup_steps": 2,
    "grad_clip": 1.0,
    "precision": "fp32",
    "seed": 1234,
}


def _manifest(tmp_path: pathlib.Path, n: int = 6) -> pathlib.Path:
    """A manifest with no audio behind it. The test adapter never opens a file."""
    p = tmp_path / "train.tsv"
    rows = ["id\twav22\twav16\tseconds\ttext"]
    for i in range(n):
        rows.append(f"u{i}\tw/{i}.wav\tw16/{i}.wav\t2.0\tकमल")
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


class ToyGanAdapter(adapters.AdapterBase):
    """A generator and a discriminator, named the way upstream names them.

    `disc.` prefix on purpose: VitsAdapter splits parameters by that prefix
    because that is how Vits.get_optimizer splits them, so the test exercises
    the same rule rather than a convenient one.
    """

    name = "toy_gan"
    n_optimizers = 2
    calls: list[int]

    def build(self, vocab_size: int, cfg: dict):
        import torch.nn as nn

        class G(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 8, padding_idx=0)
                self.gen = nn.Linear(8, 4)
                self.disc = nn.Linear(4, 1)

            def forward(self, ids):
                return self.gen(self.emb(ids).mean(1))

        self.calls = []
        m = G()
        # Kept so a test can ask which parameters moved without rebuilding the
        # model at a guessed vocabulary size.
        self.initial = {k: v.detach().clone() for k, v in m.state_dict().items()}
        return m

    def collate(self, batch, enc, cfg) -> dict:
        ids = adapters.pad_ids([enc.encode(u.text) for u in batch])
        return {"ids": torch.from_numpy(ids)}

    def _split(self, model):
        disc, gen = [], []
        for name, p in model.named_parameters():
            (disc if name.startswith("disc.") else gen).append(p)
        return disc, gen

    def optimizers(self, model, cfg: dict) -> list:
        disc, gen = self._split(model)
        return [adapters.adamw(disc, cfg), adapters.adamw(gen, cfg)]

    def param_groups(self, model) -> list:
        disc, gen = self._split(model)
        return [disc, gen]

    def loss(self, model, t: dict, optimizer_idx: int = 0):
        self.calls.append(optimizer_idx)
        fake = model(t["ids"])
        if optimizer_idx == 0:
            # Discriminator: scores a DETACHED generator output, exactly as
            # coqui's idx-0 branch does, so this backward leaves the
            # generator's graph intact for the idx-1 branch.
            return model.disc(fake.detach()).pow(2).mean()
        return model.disc(fake).pow(2).mean() * -1.0 + fake.pow(2).mean()


def _run(tmp_path, adapter, out, steps=4, cfg=None):
    return train(dict(cfg or CFG), adapter, out_dir=out, max_steps=steps,
                 log_every=1, ckpt_every=steps, device="cpu",
                 manifest=_manifest(tmp_path))


# --- both optimisers actually step -----------------------------------------

def test_both_halves_of_the_model_move(tmp_path):
    """A single-optimiser loop would leave one half of a GAN untrained."""
    a = ToyGanAdapter()
    out = tmp_path / "run"
    _run(tmp_path, a, out)

    trained = torch.load(out / "checkpoints/step_4/state.pt",
                         weights_only=False)["model"]
    moved = {k for k, v in a.initial.items()
             if not torch.equal(trained[k].cpu(), v)}
    assert any(k.startswith("disc.") for k in moved), \
        "the discriminator never changed, so its optimiser never stepped"
    assert any(k.startswith("gen.") for k in moved), \
        "the generator never changed, so its optimiser never stepped"


def test_the_optimisers_are_stepped_in_index_order(tmp_path):
    """0 then 1, every step. The generator branch reads the cache the
    discriminator branch wrote, so the reverse order scores a stale batch."""
    a = ToyGanAdapter()
    _run(tmp_path, a, tmp_path / "run", steps=3)
    assert a.calls == [0, 1, 0, 1, 0, 1]


# --- clipping is per optimiser ---------------------------------------------

def test_both_gradient_norms_are_logged(tmp_path):
    """One scalar cannot show which half of a GAN is diverging."""
    out = tmp_path / "run"
    _run(tmp_path, ToyGanAdapter(), out)
    recs = [json.loads(l) for l in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs, "nothing was logged"
    for r in recs:
        assert len(r["losses"]) == 2
        assert len(r["grad_norms"]) == 2


def test_the_default_is_still_one_optimiser(tmp_path):
    """The other three architectures must not acquire a second optimiser.

    Checked on the base class rather than by training, so this test needs no
    audio: the defaults are what every non-adversarial adapter inherits.
    """
    import torch.nn as nn

    model = nn.Linear(4, 4)
    base = adapters.AdapterBase()
    assert base.n_optimizers == 1
    assert len(base.optimizers(model, CFG)) == 1
    assert len(base.param_groups(model)) == 1
    t = {"x": 1}
    assert base.prepare(model, t) is t          # a no-op by default

    assert adapters.FastSpeech2Adapter.n_optimizers == 1
    assert adapters.VitsAdapter.n_optimizers == 2


# --- the checkpoint carries both states ------------------------------------

def test_the_checkpoint_holds_one_state_per_optimiser(tmp_path):
    out = tmp_path / "run"
    _run(tmp_path, ToyGanAdapter(), out)
    blob = torch.load(out / "checkpoints/step_4/state.pt", weights_only=False)
    assert "optimizers" in blob and len(blob["optimizers"]) == 2
    assert all(st["state"] for st in blob["optimizers"]), \
        "an optimiser state with no moments means that optimiser never stepped"


def test_resume_is_bit_exact_with_two_optimisers(tmp_path):
    """The same guarantee r01 already has, extended to the adversarial path.

    If the discriminator's Adam moments restart from zero on resume, its updates
    differ, the generator sees a different discriminator, and the generator's
    weights drift. So comparing the generator's weights is enough to catch a
    dropped discriminator state.
    """
    killed = tmp_path / "killed"
    torch.manual_seed(CFG["seed"])
    _run(tmp_path, ToyGanAdapter(), killed, steps=2)
    _run(tmp_path, ToyGanAdapter(), killed, steps=4)

    ref = tmp_path / "ref"
    torch.manual_seed(CFG["seed"])
    _run(tmp_path, ToyGanAdapter(), ref, steps=4)

    a = torch.load(ref / "checkpoints/step_4/state.pt", weights_only=False)["model"]
    b = torch.load(killed / "checkpoints/step_4/state.pt", weights_only=False)["model"]
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst == 0.0, f"resumed weights differ by {worst:g}"


def test_an_old_single_state_checkpoint_refuses_a_two_optimiser_resume(tmp_path):
    """r01 and r04 were checkpointed before this path existed.

    Reading one of those into a two-optimiser run would silently leave the
    discriminator's moments at zero, so it stops instead.
    """
    out = tmp_path / "run"
    _run(tmp_path, ToyGanAdapter(), out, steps=2)
    p = out / "checkpoints/step_2/state.pt"
    blob = torch.load(p, weights_only=False)
    torch.save({"model": blob["model"], "optimizer": blob["optimizers"][0]}, p)
    with pytest.raises(SystemExit):
        _run(tmp_path, ToyGanAdapter(), out, steps=4)


def test_a_mismatched_optimiser_count_refuses_to_resume(tmp_path):
    out = tmp_path / "run"
    _run(tmp_path, ToyGanAdapter(), out, steps=2)
    p = out / "checkpoints/step_2/state.pt"
    blob = torch.load(p, weights_only=False)
    torch.save({"model": blob["model"], "optimizers": blob["optimizers"][:1]}, p)
    with pytest.raises(SystemExit):
        _run(tmp_path, ToyGanAdapter(), out, steps=4)


# --- prefetching must not change the run ------------------------------------

def test_prefetch_preserves_order_and_completeness():
    """Prefetching changes WHEN a batch is prepared, never which one."""
    from src.train.runner import _prefetch

    src = list(range(200))
    assert list(_prefetch(iter(src), depth=3)) == src
    assert list(_prefetch(iter([]), depth=2)) == []


def test_prefetch_reraises_a_producer_error_instead_of_hanging():
    from src.train.runner import _prefetch

    def bad():
        yield 1
        raise ValueError("feature cache is lying")

    got = []
    with pytest.raises(ValueError, match="lying"):
        for x in _prefetch(bad()):
            got.append(x)
    assert got == [1]


def test_prefetch_can_be_switched_off(monkeypatch):
    """TRAIN_PREFETCH=0 is the escape hatch if a step ever looks unrepeatable."""
    from src.train.runner import _prefetch

    monkeypatch.setenv("TRAIN_PREFETCH", "0")
    assert list(_prefetch(iter([1, 2, 3]))) == [1, 2, 3]


def test_a_prefetched_run_matches_a_serial_one_bit_for_bit(tmp_path, monkeypatch):
    """The strongest form: same weights with prefetch on and off."""
    monkeypatch.setenv("TRAIN_PREFETCH", "0")
    torch.manual_seed(CFG["seed"])
    serial = tmp_path / "serial"
    _run(tmp_path, ToyGanAdapter(), serial, steps=4)

    monkeypatch.delenv("TRAIN_PREFETCH", raising=False)
    torch.manual_seed(CFG["seed"])
    pre = tmp_path / "pre"
    _run(tmp_path, ToyGanAdapter(), pre, steps=4)

    a = torch.load(serial / "checkpoints/step_4/state.pt", weights_only=False)["model"]
    b = torch.load(pre / "checkpoints/step_4/state.pt", weights_only=False)["model"]
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst == 0.0, f"prefetching changed the run by {worst:g}"
