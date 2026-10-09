# Artifact state, exported from the tracker
This is the contents of the live tracking artifact "Schwa Deletion Benchmark"
(`claude.ai/artifact/Bgsh94cpAb2KKdF9BQjH6M`), exported as text on 9 October
2026 so that it survives the move to another Claude account. An artifact is
private to the account that published it, so the page itself does not transfer:
this file is the part that mattered.

The artifact's own state last moved on **3 October, 18:15 IST**, so where it
disagrees with `RESULTS.md` or `handoff/02-STATE-OF-PLAY.md`, those are newer.
Four days of work followed it and were never written back to the card.

Three collections: a decision log, a blocker list, and per-run and per-task
state. All three follow.

---

## Decision log (35 entries)

### 13 Sep — Front end, provisional.

49/49 against the author's own answer key. Not a result: it measures self-consistency, since the rule and the gold forms were written by the same author in the same sitting.

### 14 Sep — Splits frozen.

Hindi 5,085 train / 300 test / 100 dev; Marathi 5,159 / 300 / 100. Membership from a salted SHA-256 of the utterance id, so a rebuild reproduces test.tsv byte for byte. The ladder's top rung is 9 h, not 10: it is the largest round figure both languages clear after trimming.

### 14 Sep — Forced alignment dropped.

MFA would hand FastSpeech 2 externally supervised durations that VITS and Matcha never see, inside a comparison whose subject is the architecture. It is also asymmetric across languages, since MFA publishes a Hindi model and no Marathi one. Alignment is learned inside every model.

### 1 Oct — Front end scored against speaker judgement: 39/47 = 83.0%.

Replaces the 49/49 figure. Seven of the eight mismatches are speakers keeping a schwa the rule deletes, which is citation-form reading, so 83.0% is a lower bound. Four of the eight are one environment: word-final schwa after a cluster in Sanskrit-derived words. block_before stays empty rather than being fitted to the test set.

### 1 Oct — Resume proven bit-exact.

A killed run restarted from its checkpoint gives weights bit-identical to one never interrupted, across 11 tensors. The data order comes from (seed, step) rather than from the checkpoint, so a checkpoint cannot disagree with its manifest.

### 1 Oct — r04 complete.

FastSpeech 2, Hindi, graphemic. 100,000 steps, final loss 302.46, checkpoint step_100000. r01 was at 87.6% and finishing. Together they are the Hindi ablation pair.

### 1 Oct — Two gaps found by reading the repository.

hifigan is declared in ARCHITECTURES with configs at r06 and r17 but has no entry in ADAPTERS. And runner.py builds one AdamW over model.parameters(), while VitsAdapter.loss is written against a two-optimiser path that was never built, so the discriminator is never stepped on its own.

### 2 Oct — Two-optimiser path built; VITS optimizer_idx was inverted.

Verified against coqui-tts 0.27.5 rather than remembered: optimizer_idx 0 is the discriminator and 1 is the generator, and the idx-1 branch reads outputs the idx-0 branch cached. The adapter called idx=0 believing it was the generator. runner.py also had one AdamW over all parameters, so the discriminator was never stepped on its own. Now: per-architecture optimiser lists, per-optimiser gradient clipping, both learning rates taken from the budget, checkpoints holding one state per optimiser and refusing a mismatched resume. 8 new tests pass, including a bit-exact resume with two optimisers. Not yet run against real VITS on the DGX.

### 2 Oct — Vocoder decided: pretrained HiFi-GAN. Matcha stays.

r06 and r17 leave the matrix and the deviation goes in the methods table. Blocking before any audio from it is scored: the checkpoint's mel configuration must match 22.05 kHz, hop 256, FFT 1024, window 1024, 80 mels. A mismatch sounds plausible and scores wrongly. Matcha-TTS stays in the matrix, so MatchaAdapter is still to be written and the paper keeps three architectures.

### 2 Oct — VITS reached the generator branch; r07 + r08 training instead.

The dry run failed at vits.py:1129 with KeyError: 'mel', inside the optimizer_idx == 1 branch, which means idx 0 ran. The inverted optimizer_idx was the real original defect and is fixed. The new failure was mine: prepare() was on FastSpeech2Adapter instead of the shared CoquiAdapter, so VITS never ran format_batch_on_device. Fixed in 9ceac4e with two stub-model tests that catch it without a GPU or an upstream package. The fallback launched r07 (ladder 5 h) and r08 (ladder 1 h) at 00:20 IST, both due near 07:00.

### 2 Oct — VITS completes a training step. r01 confirmed at 100,000.

4 steps on CPU, loss 244.6698 -> 243.1716 in 3m21s. The blocker that held six of the runs is cleared. r01 finished at 100,000 steps, final loss 316.0127, 5 checkpoints; r04 at 302.4603. Still unproven for real VITS: bit-exact resume, which is proven only for the toy GAN so far. Note for the paper: r01 and r04 final losses are not comparable to each other, since the phonemic and graphemic arms have different vocabularies and sequence lengths. The arms are compared on audio metrics, not on training loss.

### 2 Oct — The DGX is shared, and the step rate is not ours to set.

A third process holds 11,192 MiB on the card, above the 10,084 MiB our TRAIN_GPU_FRACTION=0.25 cap allows, so it is not ours. GPU at 99% with no throttling while our rungs crawled; host load average 263 to 294 on 256 cores against roughly 8 cores of ours. Our combined throughput is 2.83 it/s against 8.22 on an idle card, so we are getting about 34% of the card. The step budget is untouched and no comparison between runs changes. Only the calendar does: a pair takes about 19.6 h instead of 6.76, and the eight remaining pairs become 6.5 days rather than 2.25. Every wall-clock figure in the plan descends from 4.11 it/s measured on an idle card and is no longer reliable.

### 2 Oct — Vocoder mel front end is now checked rather than assumed.

The vocoder decision left one blocking item: the pretrained checkpoint's mel configuration must match this project's, and --vocoder took a bare string that was never read. src/export/vocoder.py reads either lineage's key spellings (coqui audio.fft_size/num_mels/mel_fmax, jik876 n_fft/hop_size/fmax) from a json file, a directory, or a checkpoint's embedded config, and compares all seven parameters against features.py, which is what actually computes the training spectrograms. Absence counts as a mismatch: a config that does not state its hop cannot be checked. bundle.py --vocoder now fails the export on a mismatch and records mel_verified in the manifest. 20 assertions pass. Warning worth having in advance, inference and not verified: HiFi-GANs published for LJSpeech commonly cap the mel bank at 8 kHz against this project's 11,025, with sample rate, hop, FFT and window all agreeing, which is the hardest version to notice by ear.

### 2 Oct — The fixed budget is fixed in frames, and a frame is not a fixed amount of audio.

Found while writing the adapter parity check. batch_frames is 12,000 in all 19 configs and assert_budget_matched enforces that the integer is identical, but a frame is 256 samples at the run's own sample rate. At 22,050 Hz that is 139.32 s of audio per step; at 16,000 Hz it is 192.00 s. The ratio is 1.3781, exactly 22,050/16,000, so every VITS run sees 37.8% more audio per step than its FastSpeech 2 counterpart under a budget the code certifies as identical. Over 100,000 steps: 3,870 h of audio against 5,333 h. The phonemic-versus-graphemic contrast is measured within one architecture and is unaffected, and the parity test now asserts that those arms see identical batches. Any FastSpeech 2 against VITS statement is affected, including the two ladders against each other. Three routes are in RESULTS.md and the decision is open.

### 2 Oct — No published HiFi-GAN matches this project's mel band.

This morning's 8 kHz warning was an inference. Now measured, and it is not one checkpoint's quirk: jik876's config_v1, v2 and v3, Matcha's bundled vocoder, and Matcha's own LJSpeech training mel all build the filter bank to 8 kHz. This project builds to sr/2 = 11,025 and matches on sample rate, FFT, hop, window and mel count. About 22 of the 80 bands lie above 8 kHz, so an off-the-shelf vocoder has never seen roughly a quarter of what our acoustic models predict. Cause worth naming: fmax = sr/2 was defensible while r06 and r17 were going to be trained on it, and dropping them to save 7.0 h removed the only component fitted to this convention. Scope: 9 of 17 runs are mel-based and affected (r01, r03, r04, r07-r10, r15, r18); the 8 VITS runs are end to end and are not. Four of the nine have already trained on sr/2 mels. Four routes with costs are in RESULTS.md; route 1, moving the project to 8 kHz, costs 39.2 h of retraining at the contended rate or 13.5 h idle.

### 2 Oct — Matcha-TTS adapter written. NotImplementedError is gone from the matrix.

Written against matcha-tts 0.0.7.2's real API rather than its documentation: get_losses returns duration, prior and flow-matching losses and upstream sums them unweighted; mel is channels-first like VITS and unlike FastSpeech 2; encoder and cfm are read by attribute while decoder is splatted as a mapping, which only reading flow_matching.py reveals. Hyperparameters come from upstream's shipped configs, not from Decoder's code defaults, which differ on num_heads and act_fn. It refuses two guesses: corpus mel statistics now come from the new src/train/melstats.py over our own cache, with the mel parameters recorded so a stale file is refused, because upstream's LJSpeech numbers and the 0.0/1.0 defaults would both train against the wrong centre and scale without erroring; and init_from is refused while it names a bare label or while the mel bands differ, since 8 kHz weights under an sr/2 bank are a different model rather than a warm start. 18 assertions pass without torch, matcha or audio. Unverified: that matcha-tts installs beside coqui-tts 0.27.5 on the DGX, since it pulls lightning, torchvision, torchmetrics and torchaudio and its alignment search is a Cython extension.

### 2 Oct — Mel band resolved by fitting the vocoder to us, not us to the vocoder.

Constraint: keep the trained parameters, finite GPU, credibility first. That picks one route. r06 and r17 come back, warm-started from a published checkpoint rather than trained from scratch. Costs 19.6 h contended against 39.2 h for retraining to an 8 kHz convention, and discards none of the 21.0 h already spent on r01, r04, r07 and r08. It dissolves all three axes of the mismatch at once, because a vocoder fitted to our own ground-truth mels is fitted to our band, our amplitude convention and our framing together, whatever they are. It also removes the deviation that said the vocoder is not matched to the corpus or speaker, rather than adding one. The warm start is close: our amplitude convention is already exactly jik876's log(clamp(mel, 1e-5)), so only the band and 128 samples of padding differ. Step count gets a stopping criterion on held-out mel reconstruction error rather than a step budget, since the vocoder is not an object of comparison, and that exemption from the fixed-budget rule goes in the methods table. Mel-domain metrics need no vocoder, so r01 and r04 can be scored today.

### 2 Oct — HiFi-GAN adapter written. Every architecture in ARCHITECTURES now has one.

Three things read out of coqui's source rather than trusted. GAN.get_optimizer's docstring says the generator is first; its code returns [optimizer2, optimizer1], which is the discriminator, and train_step indexes the code's order, so following the docstring would have inverted the optimisers exactly as the VITS adapter once did. GAN.train_disc defaults to False and is set by coqui's own trainer from total_steps_done, which we do not use, so left alone the discriminator never trains and the GAN is quietly a generator with a reconstruction loss. And the L1 spectrogram term in GeneratorLoss computes its own mel, which at coqui's defaults is already ours (fmin 0.0, mel_fmax None, which librosa reads as sr/2) but is now set explicitly. collate crops a fixed 8192-sample segment, 32 frames at hop 256, because a vocoder trained on padded batches learns to produce the padding; the crop offset is hashed from (seed, step, utterance id) rather than drawn from an RNG, because collate runs on the prefetch thread ahead of the loop and a global draw would be captured in a checkpoint at the wrong position and silently end the bit-exact resume guarantee. A warm start loads what matches by name and shape, prints the counts, and refuses when no generator tensor matched, since that is a cold start wearing a checkpoint's name.

### 2 Oct — The vocoder stops when it converges, and the step it reached is reported.

The whole cost argument for fine-tuning rather than retraining depends on the vocoder not quietly taking 100,000 steps, which would turn 19.6 h into the 39.2 h the other route costs. earlystop.Plateau takes numbers and returns a decision, with a min_delta because without one fourth-decimal noise reads as improvement forever, and with a check that could not run ignored rather than counted as a failure to improve. The signal is held-out mel reconstruction error: generate audio from dev-set mels, recompute the mel of what came out through the same code path that produced every training mel, and take the mean absolute difference. features._spec_and_mel is now that single implementation, so the validation cannot drift from training through a second copy of the formula. The runner writes stopped.json beside the checkpoints with the step, the reason and the sequence it was decided from. eval_every, patience and min_delta are launch flags and not config fields, so they stay out of config_hash and cannot be mistaken for part of the budget, which the vocoder is already excluded from.

### 2 Oct — Matcha to future work. The other 14 runs are queued, and weights now leave the machine.

r03 is commented out of plan_runs rather than deleted, with the adapter, its 18 tests and configs/r03.yaml kept, so it reverses by uncommenting three lines. It joins StyleTTS 2 in a FUTURE_WORK dict so the paper's future-work section comes from the repository rather than from memory. Matrix: 18 runs, 8 FastSpeech 2, 8 VITS, 2 vocoders, 140 GPU-h planned, budget assertion still passing. The remaining 14 runs are queued as 7 pairs in priority order: Hindi VITS ablation, both Marathi control pairs, the VITS ladder, the FastSpeech 2 ladder, then the vocoders. The ladder is last because it is the shock absorber, so losing the card mid-queue costs the weakest claim rather than the central one. The queue is idempotent, logs a failing pair and continues rather than stalling, waits for our own runs rather than for the card to empty, and waits a bounded hour for a foreign tenant before proceeding with whatever memory is free. It cannot prevent a tenant arriving; nothing can. Weights and results are copied to a Hugging Face repo on a timer by scripts/offload.py, closing phase 3 task t5: small files every pass, then one checkpoint per run, the final one or the newest, plus export bundles since the front end consumes those.

### 2 Oct — The tenant left, the ladder pair finished, and VITS runs at about 1.6 it/s.

r07 and r08 both reached 100,000 steps, at 4.92 and 4.26 it/s against r01's 4.07 on the same architecture and budget, which is the clearest evidence that the foreign tenant was real and was costing roughly a fifth of the card. r02 and r05 were restarted from step_5000 with the intra-op thread pool capped at 16, since torch sizes that pool at one thread per physical core per process and two runs on one host ask for twice what exists. They now run at 1.55 and 1.62 it/s with GPU utilisation averaging 91.5% over twenty samples, so the card is near its ceiling and there is nothing further to reclaim. The pair lands around 09:00 to 09:40 IST on 3 October. The rate improved about 5% after the cap, but the host's own load fell from 339 to 100 in the same window, so the gain is not attributable to the cap alone. Correction worth keeping: /proc/loadavg is not namespaced inside a container, so it reports the whole host including tenants whose processes ps cannot see. Load average was never evidence about our own runs, in either direction, and was twice read as if it were.

### 2 Oct — r06 verified end to end. The vocoder path is closed.

Warm start loads 234 of 404 tensors, which is the whole generator; the 170 in the MPD and MSD discriminators start fresh because a generator-only checkpoint cannot supply them, and the run says so in its own output. Both optimisers step, the loss falls from 41.72 to 39.85, and a kill-and-resume reproduces an uninterrupted 4-step run bit-identically across all 404 tensors. That last check is the one that matters here: the segment crop is hashed from (seed, step, utterance id) rather than drawn from an RNG, and killing a run is the only way to know it reproduces. Three defects were fixed to get there, each silent or misattributed on its own: TextEncoder.for_config raises on input_repr 'none' and the loop called it before any adapter was built, so r06 died before init_from was ever read; a straight load_state_dict matched zero tensors because torch renamed weight-norm storage from X.weight_g to X.parametrizations.weight.original0 and every published checkpoint predates that; and coqui's loss modules hold an STFT window built on the CPU at construction while the loop moved only the model, so the first spectral loss on the GPU failed on a device mismatch. The resume check ran on CPU because the card had 2.75 GiB free beside r02 and r05 and --kill-at runs three trainings; checkpointing, RNG restoration and the crop are device-independent.

### 3 Oct — The queue froze on a shell idiom, and the grapheme arm had never expanded numbers.

pgrep -c prints its count on stdout and exits 1 when that count is zero, so the '|| echo 0' in ours_running returned two lines and the -eq 0 guard in wait_for_our_runs errored instead of returning true. The chain froze the moment a pair finished and would have spun out its full 30 h budget; gpu_free.sh had the same line feeding an arithmetic expansion. Then r18 died in collate at step 1: normalize runs devanagari_digits_to_ascii and grapheme_inventory is the U+0900 block plus punctuation, so an ASCII digit is outside it by construction, and phonemize expanded numbers before tokenising while graphemes did not. One missing call. The larger half is that for any utterance containing a numeral the phoneme arm heard a spoken number while the grapheme arm saw a digit glyph, which is exactly the confound src/train/text.py exists to exclude. Measured over the full manifests: hindi has 0 digit runs and 0 OOV in either arm across 5,485 utterances, so r01, r04, r07 and r08 are untouched; marathi has 19 digit runs over at most 27 of 5,559. One Latin z in ma_007005 was settled by listening to the recording rather than by grammar — the speaker nasalises, so it is anusvara — and the repair is scoped to the Devanagari-flanked case so an unflanked z fails the coverage check instead of being silently nasalised. scripts/check_text_coverage.py now gates every run in a pair before launch, because the dry run only ever exercised the first of a pair, which is how r18 reached the GPU at all.

### 3 Oct — The demo runs on CPU, and Hugging Face closed the free Gradio tier.

Correction: the demo was first framed as live inference hosted on the DGX against a static page of pre-rendered audio, on the grounds that the card is temporary. That was the wrong axis. Inference is a CPU workload and the GPU is for training; the DGX was only where the checkpoints and the environment happened to be. Hugging Face then restricted free cpu-basic in July 2026, so Gradio and Docker Spaces need PRO while Static Spaces stay free. The permanent artifact is therefore a Static Space of pre-rendered audio, with space/app.py kept for a notebook when live free text is wanted. Not a consolation: a fixed, pre-registered sentence set is what a listening test requires, so this is material the dissertation needed regardless. src/export/synthesize.py is the one synthesis path for both, rebuilding the model through the adapter that trained it and loading the bundle's own vocab.json rather than a rebuilt one. coqui's inference surface was probed rather than assumed: ForwardTTS returns mel as [B, T_frames, n_mels] and Vits returns waveform as [B, 1, N], and 24 tokens gave 48 frames and 12,288 samples, which confirms hop 256 from the other direction. The mel transpose happens once on the way out, because the HiFi-GAN generator takes [B, n_mels, T] and a mel handed over the wrong way round sounds like speech and is subtly wrong. Bundles are 332.5 MB of fp32. Sentence selection is by stated rule: held-out dev split only, filtered to 1.5-6.0 s, banded by medial schwa deletion count and taken round-robin from the richest band down, sorted by utterance id, deterministic given the split, with word-final deletion counted but never selected on because it is near-categorical in Hindi and both arms learn it. Hindi dev gives 47 of 100 utterances in the window, banded 5/20/13/9, and the twelve chosen span 6 medial sites down to 0, the three zero-medial sentences standing as the control. Deletion sites come from the training path: delete_schwas always returned them and phonemize_word discarded them, so that function is refactored rather than reimplemented, and all 65 existing G2P tests pass unchanged.

### 3 Oct — First audio from r02 and r05. Sequence length is not the confound.

Twelve sentences through both arms plus the natural recording at 16 kHz: 36 clips, 4.9 MB. First listening, one listener: r02 clear on namak and slightly off on kamal, r05 largely unintelligible. That is the direction the study predicts, and both front ends were correct, n schwa m schwa k against the four bare characters, so what is audible is the models rather than the pipeline. One sentence and one listener, so this is consistent with the hypothesis and is not evidence for it; MCD, F0 and the listening test settle it. Recorded mainly because it rules out a confound: 655 phoneme tokens against 639 grapheme across the twelve sentences, a ratio of 1.025, largest per-sentence gap 9.8% and most within one or two tokens, so neither arm is working from a materially different sequence length. One structural asymmetry does remain and is intrinsic to the contrast rather than a defect: r05 carries 136 symbols against r02's 78, so the graphemic arm fits a larger embedding table from the same nine hours. The CPU timings taken alongside are unusable, since the same sentence measured 1.90 s and 0.65 s on a contended box, 2.92x apart, so nothing there says whether a two-core host is viable.

### 3 Oct — The VITS runs never warm-started from MMS, and the config says they did.

configs/r02.yaml carries init_from: facebook/mms-tts-hin, that string is inside config_hash, the deviations list asserts the 16 kHz rate is inherited from the MMS checkpoint, and VitsAdapter's docstring says both declared deviations are real in the code. VitsAdapter.build calls Vits.init_from_config and never reads cfg['init_from']. The only warm-start paths in src/ are HiFiGanAdapter._warm_start, MatchaAdapter, and runner.py's resume-from-checkpoint; nothing anywhere loads facebook/mms-tts-hin and the project contains no from_pretrained call. So r02 and r05 trained from random initialisation, and slurred articulation is what an undertrained VITS sounds like. Undertrained is not itself the defect: a fixed-budget study reports what the budget buys, and the 100,000 steps were met while convergence was never claimed. The defects are the false provenance, and that 16 kHz existed only because MMS is 16 kHz and is now an orphaned choice putting VITS on a different bandwidth and a different audio-per-step figure from FastSpeech 2 for no stated reason. Correction to how l14 was restated on 3 Oct: at 12,000 frames, 16 kHz gives 192.00 s of audio per step and 22.05 kHz gives 139.32 s, so the VITS runs saw more audio per step, not less. Scope: only r02 and r05 are spent and would need redoing, roughly 8 to 13 h for the pair on a free card by scaling the measured FastSpeech 2 rates, which is an inference and not a measurement of VITS on a free card; r11 to r14, r16 and r19 have not started, so changing them costs nothing. Curing it for Marathi alone is the one option that is clearly wrong, since it would put initialisation inside the Hindi-against-Marathi comparison that r16 and r19 exist for.

### 3 Oct — Vocoders moved ahead of VITS; no warm start, no extra steps.

Two decisions and a reordering. The vocoders were queued last while they were an upper bound on 10 GPU-h each with nothing depending on them; both halves of that stopped being true, since the FastSpeech 2 arms are mute without one and their mels turned out to be over-smoothed in a way only a learned prior can carry, and they stop on a plateau criterion rather than spending a budget. They are now next, which also means the Marathi VITS pair cannot launch on an unsettled configuration. Second: no MMS warm start and no extra steps for VITS. A warm start imports training compute the budget does not count and extra steps for one architecture breaks the symmetry outright, and the equal-compute claim is worth more than nicer audio. The artefacts stay and get reported. What that leaves open is the sample rate, which is now the sharper question rather than a settled one: 16 kHz was adopted because MMS is 16 kHz, so with the warm start deliberately abandoned the rate is unmotivated, and it hands VITS both a narrower band and 37.8% more audio per step than FastSpeech 2. Moving to 22.05 kHz costs no compute at all, frames times hop being 3,072,000 samples per step at any rate, and costs only the 9.7 h to redo r02 and r05 since the other six VITS runs have not started. It would delete a deviation from the methods table instead of adding an explanation to it. scripts/probe_mms_match.py was written before the warm start was declined and is kept: it reports what a transformers VitsModel could ever transfer to coqui's Vits, which is a question the future-work section will be asked.

### 3 Oct — Remaining work reviewed end to end; six items were missing from the plan.

Phase 7, the demo and listening interface, is 6 of 10. Phases 0 to 2 are complete. Phase 3 is 13 of 16 after two new diagnostics were added. Phase 4 is 3 of 5 with the seed floor queued and the Hindi vocoder still to run. Phase 5 has not started beyond r07 and r08. Phase 6 is 2 of 9. Phases 8 and 9 have not started. Twenty runs now: six finished, two running, twelve queued. What was missing rather than merely unfinished, and is now on the board: an ASR model has to be chosen and downloaded before ASR-WER can be computed, and nothing tracked that; UTMOSv2 and NISQA are trained on English speech and their validity on Devanagari is unexamined, so they are either checked or labelled; the per-position error analysis that the phone-frequency finding implies, splitting deletion-derived word-final consonants from the rest, did not exist as a task; the listening test may need an institutional ethics process whose lead time no amount of compute can absorb; the paper's deviations table now has to emit corrections and future work from the configs rather than be retyped, or the correction mechanism added this afternoon achieves nothing; and two diagnostics are outstanding, the gradient-clipping measurement and the 5 h to 1 h loss cliff, the second of which gates whether any ladder figure drawn from training loss means anything. Two possible gaps are raised as questions rather than tasks because they may be deliberate: the stress-test set is Hindi only while Marathi carries the cross-language claim, and the ladder is Hindi only.

### 3 Oct — Scheduled against the 11 Oct deadline: the GPU is not the constraint.

GPU access runs a few more days and result gathering plus paper writing must start by Sunday 11 Oct. The remaining queue comes to 49.1 h of training against 174 h of wall clock, so it lands around Tuesday with roughly 125 h to spare: r15 and r18 finishing at 1.5 h, the vocoders at an upper bound of 10, the seed floor at 4.2, Marathi VITS at 9.7, the two VITS ladder pairs at 9.7 each and the FastSpeech 2 ladder pair at 4.2. Training is therefore not the critical path and the ladder does not need to be cut. What is on the critical path is the evaluation harness. Phase 6 stands at 2 of 9, with MCD and F0 existing as functions and nothing that walks the runs, synthesises the frozen test split, computes the metrics and writes a table. It needs no GPU and can be built in parallel; if it is not, Sunday arrives with twenty trained models and no numbers. One thing is available sooner than planned: VITS is end to end, so r02 and r05 can be scored as soon as the harness exists and do not wait on r06, which means the central phonemic-versus-graphemic comparison is the first result obtainable rather than the last. That also exposes a gap: noise floors are architecture-specific and the queued seed pair is a FastSpeech 2 cell, so the VITS comparison would currently be reported without a scale. No consent process is needed for the listening test.

### 3 Oct — A VITS seed floor, queued first; a check that would have caught the MMS gap; the dry run dropped.

r22 and r23 are r02's cell at seeds 1 and 2, ahead of the vocoders in the queue. A noise floor is architecture-specific, so the FastSpeech 2 pair says nothing about how far apart two VITS runs land, and VITS is the half that can be scored first because it is end to end. Sequenced first, the chain is r15 and r18 finishing at 1.5 h then this pair at 9.7 h, so the central contrast becomes complete and interpretable about eleven hours from now rather than on Wednesday. The matrix is 22 runs and the slack before 11 October goes from 5.2 days to 4.8. assert_init_from_is_honest now sits at adapters.for_config, the single point where a config meets an adapter, so it covers training, dry runs and evaluation alike: a config may not claim a warm start its adapter never performs. That is the check that would have caught the MMS gap on day one instead of after someone listened to the audio. The decided case has to pass or the queued VITS runs could not start, so a mismatch is allowed exactly when a corrections entry mentions init_from, meaning somebody wrote down what is wrong and why; a correction about something else does not excuse it, since a check any correction satisfies is decoration, and reads_init_from defaults to False so a new adapter that forgets the flag refuses rather than silently claiming to honour a warm start. All 22 runs pass it. The 500-step dry run per architecture is removed from the plan rather than performed: eight runs reaching 100,000 steps is stronger evidence than a 500-step job would be, and the queue already dry-runs two steps before each pair, so doing it now would tick a box and tell us nothing. Also written: scripts/evaluate.py, the evaluation harness, which is the actual critical path to 11 October.

### 3 Oct — The 29.9x loss cliff is not a cliff: the logged loss is mostly a pitch error.

coqui's ForwardTTS loss is a sum of five terms and two of them are mean squared errors in physical units, f0 in hertz and the L2 norm of each linear frame, both at alpha 0.1, neither normalised anywhere because upstream normalises them inside a dataset class this testbed does not use. Measured on twelve utterances per rung through features.py itself: 0.1*var(f0) is 569.13 at the 1 h rung and 654.00 at 9 h, 0.1*var(energy) is 45.38 and 45.65, and the mel L1 against a mean predictor is 1.81 and 1.97. A mean predictor therefore scores 616 to 702, so r01's 314.60 implies an f0 error of 51.7 Hz against a speaker sd of 80.9 and r08's 10.04 implies 10.0 Hz. The rungs are nested subsets of one speaker so the units are identical; what differs is the pass count, which the fixed step budget makes inverse to rung size: 430 epochs at 9 h, 774 at 5 h, 3,870 at 1 h, 7,740 at 30 min, 23,220 at 10 min. 541 utterances seen 3,870 times have their f0 memorised, and that is the whole 29.9. The mel term is 0.6% of r01's loss and 18% of r08's, so the same number does not mean the same thing at two rungs: the ladder is plotted against held-out metrics on test.tsv, never against training loss. Second consequence, inference: FastSpeech 2's mel decoder is optimised against under 1% of the gradient while an unnormalised pitch head takes most of the rest, which is sufficient to explain those arms sounding like noise and is a property of the objective rather than the data. Normalising would change the config hash of ten runs, so the objective is kept fixed for the rest of the matrix and the per-term breakdown is now logged at every step instead.

### 3 Oct — First intelligibility numbers, and a punctuation defect that had inverted them.

Two CTC recognisers, never averaged: IndicConformer 600M multilingual pinned at e9b71b36, MIT, primary, and facebook/mms-1b-all pinned at 3d33597e, CC-BY-NC-4.0, cross-check. CTC rather than Whisper on purpose, since a sequence-to-sequence decoder carries an implicit language model and repairs exactly the slurred schwa being counted. Ground truth is transcribed too, because without the recogniser's own floor an absolute rate has no denominator. The defect: punctuation survived normalize(), the recognisers emit none, and 41% and 48% of the error on the first two ground-truth utterances was commas and a full stop. Worse, word_stats asks whether a deletion site is the LAST segment of a word and a trailing comma is one more segment, so all six final-class words sampled from the test split were reclassified final to medial. The final class had been excluding every clause-final word, the position where final deletion is most audible, and medial had been contaminated with them. The symptom was in the first table and I read it as small-sample noise: punctuation was 0.0000 of final-class characters against 0.0690 and 0.0561 elsewhere. Zero of 283 is not chance. After the fix, 8 words and 44 characters move back to final over ten utterances, and the two recognisers stop disagreeing: final-site excess grapheme minus phoneme was -0.0038 and +0.1871, and is now -0.0347 and -0.0351. The effect relocates to MEDIAL sites, where it is +0.1147 and +0.1150 floor-corrected, with the phonemic arm's errors avoiding those words and the graphemic arm's concentrating on them. That is the asymmetry the phonology predicts: final deletion is inferable from the orthography, medial deletion is not. The nuisance variable sits in the other class, since the floor's own final-site excess is +0.0629 and +0.0806 on real studio audio while its medial excess is -0.0307 and -0.0019. Scale, honestly: the medial class is 96 characters in 21 word tokens and the counts are 16 edits against 4, and 36 against 21. No interval is computable from ten utterances and none is claimed.

### 3 Oct — Springer conference draft written from what is measured.

paper/springer_draft.tex, LNCS class, the progress report's bibliography reformatted to LNCS style and its voice kept. Every figure traces to RESULTS.md, a run log or a table under results/tables; nothing is estimated. The structure follows the evidence rather than the synopsis: the headline is not the phoneme-versus-grapheme effect, which rests on ten utterances with no seed floor under it, but the evaluation design. MCD cannot resolve this contrast at 0.31 sigma, so the paper argues for the duration-bias slope and the site-partitioned floor-corrected error rate, then reports the medial-versus-final asymmetry both recognisers agree on. Section 6 carries the two measurement defects as the portable contribution, since a loss dominated by physical-unit MSE terms and a partition computed on punctuated references are both mistakes other people will make, and in the second case the wrong answer was the one the hypothesis predicted. Threats to validity names the seed floor as the largest gap and says plainly that no number is offered as a finding. Corrected while drafting: I had written ten completed runs; verify_queue says eight done and two running, so it reads eight, two and twelve not started. Open before submission: first person singular throughout per the progress report, which needs changing to we if the supervisor is a co-author, and the ORCID and email are placeholders.

### 3 Oct — Synthesis was unseeded, so the medial finding was one draw and it reversed.

Two runs of scripts/score_intelligibility.py with identical arguments, code, weights and the same ten utterances disagreed. r05's corpus CER under IndicConformer moved 0.1330 to 0.1478 and its medial-site errors fell from 16 of 96 characters to 5. The medial excess that both recognisers had agreed on at +0.1147 and +0.1150, reported three hours ago as the strongest evidence available, came back -0.0569 and +0.0661, reversing sign on one; the final excess reversed on the other, -0.0347 to +0.0386. The ground-truth floor was byte-identical across both passes because it reads fixed files, which is what localised the fault. Cause: VITS samples, its stochastic duration predictor and its flow both draw noise at inference, and Bundle.synthesize called inference with no seed. So those were never two measurements that disagreed, they were two draws from a distribution nobody had looked at, and the agreement inside each was an artefact of drawing once. Fixed: the seed is derived from (run_id, text, draw) by sha256, not from a global RNG and not from hash() which Python salts per process, so an utterance's audio does not depend on how many preceded it and --limit 10 measures the same audio as the full split. One seeded draw is reproducible and still is not the system, so --draws N makes each pass its own draw and reports the spread on corpus CER and on each class's excess; with --draws 1 on a VITS run the harness prints what happened today. What survived: the graphemic arm is worse on corpus CER in four cells out of four across both draws, +0.0283 and +0.0197 on IndicConformer and +0.0283 and +0.0566 on MMS. What did not: every site-partitioned number. The paper's medial claim is withdrawn, its excess table removed, and the failure written up as section 6.3 with the two runs side by side. The lesson worth keeping is the one about corroboration: agreement between two independent instruments is not evidence of a stable effect when both are reading the same single sample.

### 4 Oct — The VITS seed floor exists, and both vocoders stopped on their own criterion.

r22 and r23 reached 100,000. Final training loss across the three identical-but-for-seed VITS runs: r02 27.3800, r22 24.9429, r23 25.4861, so mean 25.9363, sd 1.2794, range 2.4371, coefficient of variation 4.93%. That is the smallest difference this setup can resolve, and it is the number plan v3 said no gap could be reported without. Caveat that matters: the arm gap cannot be read on this quantity. r02 and r05 have different vocabularies and sequence lengths, so their losses were never comparable to each other (logged 2 Oct, l11), and the 0.1953 between them is not 0.153 sigma of anything. The floor has to be applied to the held-out metrics instead, which is now possible because r22 and r23 are end to end and need no vocoder: score all four VITS runs and compare the spread across seeds to the r02-against-r05 gap on the same measure. Separately, r06 and r17 both show STOPPED with checkpoints at step 18000 and 22000, which is earlystop.Plateau firing on held-out mel reconstruction error rather than a crash; the status parser misreads their stop line, printing 'at step no 0.246545' where that number is the reconstruction error. If stopped.json confirms it, every waveform metric for the FastSpeech 2 arms is unblocked, along with their demo audio and the listening test, and r01, r04, r07, r08, r15 and r18 become scorable for the first time. The queue moved on by itself to r20 and r21, the FastSpeech 2 seed floor, at 2,600 steps with about 7.1 and 7.5 h left.

---

## Blockers (23, open ones first)
### Still open

**b4** — Decide what the fixed budget means: 12,000 frames at each run's native rate, with the 37.8% audio asymmetry declared in the deviations table; or audio seconds, which makes the 16 kHz budget 8,707 frames and invalidates every finished VITS run; or drop the cross-architecture claim. Only the first costs no GPU time. Changed on 3 Oct: the 16 kHz rate was justified by an MMS warm start that never happened, so if the VITS re-run moves to 22.05 kHz this question dissolves rather than needing an answer. Decide b12 first.

**b6** — Decide where the seed variance pair goes. Plan v3's standing rules say no difference between two runs is reportable until that number exists, and it was meant to precede the ladder. The ladder is already running, and its smallest rungs are where variance is widest and the claims weakest.

**b8** — One vocoder or two. r06 and r17 were separate because the speaker differs by language, and a Hindi-fitted vocoder on Marathi audio would put a speaker mismatch inside the control that exists to isolate language. Two is the defensible answer; one is roughly half the GPU. This is the only open question left on the vocoder.

**b14** — Publish the Static Space from the DGX, which holds both the rendered audio and the write token. The Space already exists but was created with the Gradio SDK, so hf upload's implicit create_repo is refused with 402 Payment Required; pass space_sdk='static' through HfApi with exist_ok, or delete it and recreate as Static. The page reads its own README frontmatter, which says sdk: static, so the upload converts it.

    hf upload Klewik/Indic-tts-demo space_static/ . --repo-type=space

**b16** — Run the clipping diagnostic? 25 minutes, outside the matrix so nothing's comparability is touched: 5,000 steps on the 10-minute rung at grad_clip 1.0 against clip 100. Gradient norms of 97 to 1139 against a clip of 1.0 mean essentially every step is scaled down by two to three orders of magnitude, and the note that AdamW largely cancels a uniform rescale is marked [Inference] because it was an argument rather than a measurement. If clipping is the limiter it is the most plausible single explanation for every run flattening at step 50,000, and it is reportable either way without retraining anything.

**b19** — ANSWERED in part on 3 Oct: Hindi only for the paper, Marathi deferred to dissertation part 2. Still open as a wording question rather than a work question: the paper's threats section has to say the resource claim and the front-end accuracy figure are Hindi-specific, since the contested-schwa stress-test set is Hindi only and the ladder is r07 to r14, all Hindi. Confirm that is how it should read.

**b20** — PARTLY CLEARED 3 Oct. The harness exists: scripts/evaluate.py walks the bundles and writes MCD with a chance level, F0 and the duration-bias slope; scripts/score_intelligibility.py adds the site-partitioned character error rate with the recogniser's own floor. Phase 6 is now 8 of 13. What remains and still needs no GPU: the full test split with a bootstrap interval over utterances, in place of the 10-utterance pass everything so far rests on; stress-test error rates per architecture; the MOS proxies, either validated on Devanagari or labelled English-trained; and the macOS real-time factor. The seed floor is the one item the card has to supply, and r22 and r23 are running.

**b22** — Re-score with draws and no limit, because every intelligibility number so far is one unseeded draw. The full test split at 5 draws is 300 utterances x 2 arms x 2 recognisers x 5, which is CPU only and safe beside training but is hours rather than minutes; --limit 50 --draws 5 first would say whether the draw spread is wide enough to swamp the contrast before the long pass is spent. Until that exists, the paper reports the overall ordering and nothing partitioned.

    /workspace/venv/bin/python scripts/score_intelligibility.py --lang hindi --draws 5

**b23** — Confirm the two vocoders stopped on the plateau criterion rather than failing, and read the step and reason each recorded. If they did, export bundles for r22 and r23 and re-run the intelligibility harness over all four VITS runs with --draws, which gives the seed spread and the draw spread on the same metric for the first time. The FastSpeech 2 arms also become scorable, so r01 against r04 is available and the mel-only skip path stops applying.

    cat /workspace/runs/r06/stopped.json /workspace/runs/r17/stopped.json

### Closed

**b1** — Vocoder decided on 2 Oct: pretrained HiFi-GAN, deviation declared. Closed.

**b2** — Morning check pasted on 2 Oct: r07 and r08 reached 100,000 and the tenant had left. Closed.

**b3** — Commits pushed on 2 Oct; origin/main and local both at 761ce0d. Closed.

    git push

**b5** — Mel band settled on 2 Oct: fit the vocoder to our mels by fine-tuning, keeping every trained parameter. Closed.

**b7** — Matcha moved to future work on 2 Oct, so the install question is moot. Closed.

    pip install matcha-tts && python -m src.train.melstats hindi

**b9** — Checkpoint downloaded, VOCODER_INIT set, configs regenerated, and r06 dry-run verified end to end on 2 Oct including bit-exact resume. Closed.

**b10** — HF_REPO set to Klewik/indic-tts-bench and offload running after every pair; 48 of 51 files uploaded on 2 Oct, export bundles since. Closed.

**b11** — queue_all.sh owns all 7 pairs and the duplicate chain was killed on 2 Oct. Closed.

**b12** — DECIDED 3 Oct and closed. No MMS warm start, no extra steps, no rate change; r02 and r05 keep their trained weights and Marathi keeps its own vocoder. Either lever would have imported or added training compute the other architectures do not get, and the equal-compute claim is worth more than better audio. The false init_from and the 16 kHz justification are recorded in a new non-hashed corrections field on all eight VITS runs rather than edited in place, so no finished run's provenance moved. Closed.

**b13** — Resolved 3 Oct by reordering the queue rather than by killing it: the vocoders r06 and r17 now sit ahead of the Marathi VITS pair, so r16 and r19 cannot launch on an unsettled configuration and the rate decision has the whole vocoder run to be made in. Closed.

**b15** — Answered 3 Oct: GPU access runs for a few more days, and result gathering plus paper writing must start by Sunday 11 Oct. Scheduled against that, the whole remaining queue is 49.1 h of training against 174 h of wall clock, so it finishes around Tuesday with roughly 125 h of slack. Training is therefore NOT the critical path. Closed in favour of b20.

**b17** — RESOLVED 3 Oct and closed. Not a cliff. The logged FastSpeech 2 loss is dominated by unnormalised mean squared errors on f0 in hertz and on frame energy, so the mel term is 0.6% of r01's total and 18% of r08's; the rungs are nested so the units match, and the 1 h rung simply memorises f0 over 3,870 passes. The ladder is plotted against held-out metrics on test.tsv, never against training loss, and the per-term breakdown is logged from r06 onward. See log entry l31.

**b18** — Answered 3 Oct: no consent or ethics process is required for the listening test. Closed.

**b21** — DECIDED 3 Oct: added as r22 and r23, r02's cell at seeds 1 and 2, and queued ahead of the vocoders. VITS is end to end so r02 and r05 can be scored with no vocoder, which makes the phonemic-versus-graphemic contrast the first obtainable result rather than the last; sequencing the floor first means it lands about eleven hours from now instead of Wednesday. 9.7 h, taking the slack before 11 Oct from 5.2 days to 4.8. Matrix is 22 runs. Closed.

---

## Run state as the card last recorded it (3 Oct)

| run | state |
|---|---|
| r01 | done |
| r02 | done |
| r04 | done |
| r05 | done |
| r06 | done |
| r07 | done |
| r08 | done |
| r15 | done |
| r17 | done |
| r18 | done |
| r20 | running |
| r21 | running |
| r22 | done |
| r23 | done |

Runs absent from this table were still queued on 3 October.

## Phase checklist: 54 items ticked (3 Oct)

Phase ids: p0 front end, p1 access, p2 corpora, p3 harness, p4 main runs, p5 ladder and control, p6 evaluation, pg demo, p7 listening test, p8 write-up. The task text for each id is in `handoff/artifacts/schwa-benchmark.html`, in the `PHASES` array.

- **p0**: t1, t2, t3, t4, t5, t6, t7
- **p1**: t1, t2, t3, t4, t5, t6
- **p2**: t1, t2, t3, t4, t5, t6, t7, t8
- **p3**: t1, t10, t11, t12, t13, t14, t16, t2, t3, t4, t5, t6, t8, t9
- **p4**: t1, t2, t5
- **p6**: t1, t10, t11, t13, t2, t3, t7, t9
- **p8**: t1
- **pg**: t1, t2, t3, t4, t5, t6, t8
