#!/usr/bin/env python3
"""Run configuration: the unit of reproducibility.

Every training run is one YAML file in configs/. Nothing else decides what a
run does. A results row carries its run id and its config hash, so any number
in the paper can be traced back to the exact settings that produced it.

Three things this enforces.

**A fixed budget across architectures.** The study claims to compare
architectures under an identical budget, so the budget fields live here and are
identical by construction rather than by hand. `assert_budget_matched` checks a
set of runs actually share them.

**Deviations are declared, not discovered.** Some architectures force a
difference: VITS trains a discriminator, so it has an extra optimiser the
others do not. Hiding that would make the comparison dishonest; leaving it
undocumented would make it unreproducible. Each config carries a `deviations`
list that is emitted into the paper's deviations table, and a run with an
undeclared architectural difference is a bug.

**The hash covers what matters.** It is computed over every field that changes
the trained model, and deliberately not over cosmetic ones like `notes`, so an
edit to a comment does not invalidate a completed run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = HERE / "configs"
PROCESSED = HERE / "data" / "processed"

ARCHITECTURES = ("fastspeech2", "vits", "matcha", "hifigan")
LANGUAGES = ("hindi", "marathi")
INPUT_REPRS = ("phoneme", "grapheme", "none")

# The shared budget. Every acoustic-model run gets these, and the paper's claim
# rests on them being identical, so they are defined once here.
BUDGET = {
    "max_steps": 100_000,
    "batch_frames": 12_000,        # batching by mel frames, not by utterance,
                                   # so a long utterance does not silently get
                                   # a smaller effective batch than a short one
    "lr": 2.0e-4,
    "lr_schedule": "warmup_inverse_sqrt",
    "warmup_steps": 4_000,
    "precision": "bf16",           # see PRECISION_NOTE below
    "grad_clip": 1.0,
}

# Fields that do not affect the trained model and so stay out of the hash.
# run_id belongs here too: the hash identifies the SETTINGS, so two runs whose
# settings are identical must hash identically and be caught as duplicates.
# With run_id inside the hash, a duplicated run looks like a distinct one,
# which is exactly how the ladder's top rung silently became a second copy of
# the main run and would have cost GPU hours twice.
COSMETIC = {"notes", "created", "run_id"}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    architecture: str
    language: str
    input_repr: str
    data: str                       # a ladder rung name, or "train"
    seed: int = 0
    init_from: str = ""             # pretrained checkpoint, "" means scratch
    sample_rate: int = 22050
    max_steps: int = BUDGET["max_steps"]
    batch_frames: int = BUDGET["batch_frames"]
    lr: float = BUDGET["lr"]
    lr_schedule: str = BUDGET["lr_schedule"]
    warmup_steps: int = BUDGET["warmup_steps"]
    precision: str = BUDGET["precision"]
    grad_clip: float = BUDGET["grad_clip"]
    merge_nukta: bool = False
    aligner: str = "internal_mas"   # see ALIGNER_NOTE below
    vocoder: str = ""               # "" for end-to-end architectures
    deviations: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"{self.run_id}: unknown architecture {self.architecture!r}")
        if self.language not in LANGUAGES:
            raise ValueError(f"{self.run_id}: unknown language {self.language!r}")
        if self.input_repr not in INPUT_REPRS:
            raise ValueError(f"{self.run_id}: unknown input {self.input_repr!r}")
        if self.architecture == "hifigan" and self.input_repr != "none":
            raise ValueError(f"{self.run_id}: a vocoder takes no text input")
        if self.architecture != "hifigan" and self.input_repr == "none":
            raise ValueError(f"{self.run_id}: an acoustic model needs an input representation")

    # -- data ---------------------------------------------------------------

    def data_path(self) -> pathlib.Path:
        stem = "train" if self.data == "train" else f"ladder/{self.data}"
        return PROCESSED / self.language / f"{stem}.tsv"

    def assert_data_exists(self) -> None:
        p = self.data_path()
        if not p.exists():
            raise SystemExit(f"{self.run_id}: missing training data {p}")
        lock = PROCESSED / self.language / "SPLITS.lock"
        if not lock.exists():
            raise SystemExit(f"{self.run_id}: {self.language} splits are not frozen")

    # -- identity -----------------------------------------------------------

    def hashable(self) -> dict:
        d = dataclasses.asdict(self)
        for k in COSMETIC:
            d.pop(k, None)
        d["deviations"] = sorted(d.get("deviations") or ())
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.hashable(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_yaml(self) -> str:
        d = dataclasses.asdict(self)
        lines = [
            "# Generated by src/train/config.py. The run is the unit of",
            "# reproducibility: edit this file, not the training code.",
            f"config_hash: {self.config_hash()}",
        ]
        for k, v in d.items():
            if isinstance(v, (tuple, list)):
                if not v:
                    lines.append(f"{k}: []")
                else:
                    lines.append(f"{k}:")
                    lines.extend(f"  - {item}" for item in v)
            elif isinstance(v, str):
                lines.append(f"{k}: {v!r}" if (v == "" or ":" in v) else f"{k}: {v}")
            elif isinstance(v, bool):
                lines.append(f"{k}: {str(v).lower()}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"


def assert_budget_matched(runs: list[RunConfig]) -> None:
    """Every acoustic-model run must share the declared budget.

    This is the study's central claim expressed as an assertion. If it ever
    fails, the comparison is between architectures AND budgets, and nothing in
    the results table means what the paper says it means.
    """
    acoustic = [r for r in runs if r.architecture != "hifigan"]
    for field_name, expected in BUDGET.items():
        got = {getattr(r, field_name) for r in acoustic}
        if got != {expected}:
            raise SystemExit(
                f"budget field {field_name!r} is not identical across runs: "
                f"{sorted(got)}; the fixed-budget claim would be false"
            )


CONTROL_NOTE = """The Marathi control needs two arms, not one.

Plan v2 listed the Marathi control as two runs, both phonemic, and that is what
the matrix encoded until 1 October. It cannot answer the question it exists to
answer.

The control works by comparison of effects, not of systems. In Hindi we measure
the gap between phonemic and graphemic input. If that gap is large, it could be
because schwa deletion matters, or because Devanagari is simply hard for a
character-level model for reasons that have nothing to do with schwa. Marathi
separates the two, because it uses the same script and the same converter but
does not apply medial schwa deletion. So the claim rests on: a large gap in
Hindi next to a small gap in Marathi.

With only a phonemic Marathi arm there is no Marathi gap to put next to it.
The arm can be compared against Hindi in absolute quality, but that comparison
is confounded by language, speaker and corpus all at once, which is exactly
what the control was supposed to remove.

So r18 and r19 are the Marathi graphemic arms, and the control is now four runs
rather than two. They were not an oversight in the science; they were missing
from the encoding of it."""


PRECISION_NOTE = """Mixed precision is bf16, not fp16.

The plan said fp16 because it was written for Kaggle's T4 and P100, neither of
which has usable bf16. Training moved to an A100, which does, and on that card
bf16 is the better choice for three reasons.

It has the exponent range of fp32, so activations and gradients cannot overflow
to inf the way they can in fp16. That removes the loss-scaling machinery
entirely, and with it a failure mode that is particularly unpleasant here: a
scaler that keeps halving its scale looks like slow convergence rather than
like a bug, and would cost a run before anyone noticed.

It costs nothing in speed. The A100's tensor cores run bf16 and fp16 at the
same rate, so the step budget is unaffected.

And it is the same for every run, which is what matters most. The claim this
study rests on is that the architectures were compared under an identical
budget; precision is part of that budget, so it is set once here rather than
per architecture.

Recorded limitation: bf16 has fewer mantissa bits than fp16 (8 against 10), so
it is slightly less precise per value. For acoustic modelling at this scale
that is not the binding constraint, and the overflow behaviour is."""


ALIGNER_NOTE = """Alignment is learned inside every model, not supplied by an
external forced aligner.

The original plan had FastSpeech 2 take durations from the Montreal Forced
Aligner. Two things argue against it, and a third makes it impractical here.

First, it is an uncontrolled asymmetry. VITS and Matcha learn alignment
internally through monotonic alignment search. Giving FastSpeech 2 externally
supervised durations hands one architecture information the others never see,
inside a comparison whose whole point is the architecture.

Second, it is asymmetric across languages too. MFA publishes a pretrained
Hindi acoustic model but none for Marathi, so the Marathi arm would have used a
model trained on 9 h of its own data while Hindi used one trained on far more.
That difference would sit directly inside the control comparison.

Third, MFA cannot run in this environment at all: the Python package installs
but its Kaldi bindings (_kalpy) have no aarch64 Linux wheel and do not build,
since MFA ships through conda.

So FastSpeech 2 uses an internal unsupervised aligner of the kind described by
Badlani et al., "One TTS Alignment to Rule Them All" (ICASSP 2022), which is
what current implementations do. Every architecture then learns its own
alignment from the same data under the same budget.

Recorded limitation: an internal aligner is generally weaker than MFA at very
small data sizes, so the lowest ladder rungs may be affected more than the top
ones. That is a property of the ladder to report, not a reason to reintroduce
the asymmetry."""


# --- the 17 runs from plan v2, after removing the duplicated top rung -------

VITS_DEVIATIONS = (
    "trains an adversarial discriminator, so it has a second optimiser the "
    "other architectures do not have",
    "native sample rate is 16 kHz, inherited from the MMS checkpoint, so its "
    "output carries no energy above 8 kHz",
)
MATCHA_DEVIATIONS = (
    "flow-matching decoder has a sampling-steps hyperparameter at inference "
    "that the others do not; fixed at 10 for every evaluation",
)


def plan_runs() -> list[RunConfig]:
    runs: list[RunConfig] = []

    def vits(rid: str, lang: str, repr_: str, data: str, **kw) -> RunConfig:
        return RunConfig(
            run_id=rid, architecture="vits", language=lang, input_repr=repr_,
            data=data, sample_rate=16000,
            init_from=f"facebook/mms-tts-{'hin' if lang == 'hindi' else 'mar'}",
            deviations=VITS_DEVIATIONS, **kw)

    # Hindi main: three architectures, phoneme input
    runs.append(RunConfig(run_id="r01", architecture="fastspeech2", language="hindi",
                          input_repr="phoneme", data="9h", vocoder="hifigan_hindi",
                          notes="durations learned internally; see ALIGNER_NOTE"))
    runs.append(vits("r02", "hindi", "phoneme", "9h"))
    runs.append(RunConfig(run_id="r03", architecture="matcha", language="hindi",
                          input_repr="phoneme", data="9h", vocoder="hifigan_hindi",
                          init_from="matcha_ljspeech", deviations=MATCHA_DEVIATIONS))

    # Hindi ablation: the same two architectures on raw graphemes
    runs.append(RunConfig(run_id="r04", architecture="fastspeech2", language="hindi",
                          input_repr="grapheme", data="9h", vocoder="hifigan_hindi",
                          notes="the ablation arm; differs from r01 only in input"))
    runs.append(vits("r05", "hindi", "grapheme", "9h"))

    # Vocoder, shared by the non-end-to-end architectures
    runs.append(RunConfig(run_id="r06", architecture="hifigan", language="hindi",
                          input_repr="none", data="9h",
                          notes="fine-tuned on ground-truth mels; shared by r01 and r03"))

    # Data ladder. The 9 h rung is NOT listed: it is r01 and r02, which train
    # on exactly this data with exactly this budget. Listing it again would be
    # a second copy of the same run under a different id, paid for twice.
    rid = 7
    for arch in ("fastspeech2", "vits"):
        for rung in ("5h", "1h", "30min", "10min"):
            name = f"r{rid:02d}"
            if arch == "vits":
                runs.append(vits(name, "hindi", "phoneme", rung))
            else:
                runs.append(RunConfig(run_id=name, architecture=arch, language="hindi",
                                      input_repr="phoneme", data=rung,
                                      vocoder="hifigan_hindi",
                                      notes="ladder rung; the 9 h rung is r01"))
            rid += 1

    # Marathi control. BOTH arms, for the reason in CONTROL_NOTE: a control
    # that only has the phonemic arm cannot measure an effect, and an effect
    # is the thing being compared across the two languages.
    runs.append(RunConfig(run_id="r15", architecture="fastspeech2", language="marathi",
                          input_repr="phoneme", data="9h", vocoder="hifigan_marathi",
                          notes="control arm: identical settings, schwa deletion off"))
    runs.append(vits("r16", "marathi", "phoneme", "9h"))
    runs.append(RunConfig(run_id="r17", architecture="hifigan", language="marathi",
                          input_repr="none", data="9h",
                          notes="shared by r15"))
    runs.append(RunConfig(run_id="r18", architecture="fastspeech2", language="marathi",
                          input_repr="grapheme", data="9h", vocoder="hifigan_marathi",
                          notes="control ablation; differs from r15 only in input"))
    runs.append(vits("r19", "marathi", "grapheme", "9h"))
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write configs/*.yaml")
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args(argv)

    runs = plan_runs()
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate run ids")
    by_hash: dict[str, list[str]] = {}
    for r in runs:
        by_hash.setdefault(r.config_hash(), []).append(r.run_id)
    dupes = {h: v for h, v in by_hash.items() if len(v) > 1}
    if dupes:
        raise SystemExit(
            "runs with identical settings under different ids, which would be "
            f"trained twice for the same result: {dupes}")
    assert_budget_matched(runs)

    if args.check_data:
        for r in runs:
            r.assert_data_exists()

    CONFIGS.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs, budget matched across "
          f"{len([r for r in runs if r.architecture != 'hifigan'])} acoustic models\n")
    print(f"{'run':<5}{'arch':<13}{'lang':<9}{'input':<10}{'data':<8}{'sr':<7}hash")
    for r in runs:
        print(f"{r.run_id:<5}{r.architecture:<13}{r.language:<9}{r.input_repr:<10}"
              f"{r.data:<8}{r.sample_rate:<7}{r.config_hash()}")
        if args.write:
            (CONFIGS / f"{r.run_id}.yaml").write_text(r.to_yaml(), encoding="utf-8")

    if args.write:
        print(f"\nwrote {len(runs)} files to {CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
