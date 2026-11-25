# Why `load` exists: two theories, and the honest state of the evidence

Read this before interpreting a number from [`train_load.ipynb`](../train_load.ipynb).

The `tia` direction asks a frozen foundation model what it sees. This one asks the opposite
question: **does a small set of features, chosen because a psychological theory says they should
matter, carry as much signal as a 94M-parameter encoder?** If it does, the encoder was not what was
doing the work. If it does not, the theory's cues are not in this data.

## Theory 1 — Cognitive load (Vrij)

Lying is more demanding than telling the truth: the liar has to invent, monitor the invention, and
avoid contradicting what was said before. Under that load, speech slows down and breaks up.

Predicted observable: **longer and more frequent pauses, slower speech rate, more filled hesitations,
less fluent delivery.**

The important qualification is Vrij's own: the approach works best when the interviewer **elicits**
the load — asking unexpected questions, asking for a reverse-order account, keeping the interview
unpredictable. The evidence for *passive observation* of these cues is much weaker than the evidence
for *eliciting* them. A classifier can only observe, so this direction tests the weaker half.

## Theory 2 — Vocal arousal

Concealed emotion shows in the voice as tension: the larynx tightens, the vocal folds vibrate less
regularly, the signal gets noisier.

Predicted observable: **higher and more variable fundamental frequency, more jitter and shimmer,
lower harmonic-to-noise ratio.**

This is the voice-stress hypothesis, and its evidence is genuinely mixed. Pitch does rise under
stress and cognitive demand, but the step from "stress" to "deception" is where the literature
becomes thin: many truthful situations are stressful, and many lies are not.

## The evidence, in one place

- **Behavioural cues are weak.** DePaulo et al. (2003) meta-analysed 158 cues to deception; most
  effects are small, and no single cue separates truth from lies on its own.
- **Humans are barely above chance.** Bond and DePaulo (2006), across more than 200 studies, put
  human accuracy at 54% against a 50% baseline.
- **Training does not fix it.** A direct test of Ekman's Micro-Expressions Training Tool (Jordan et
  al., 2019) found the trained group *slightly below chance* and no better than a bogus-training
  control — the same result the `tia` diagnostics reach from a different direction, where the 41
  hand annotations predict deception at 51.1 ACC.
- **Eliciting beats observing.** The cognitive approach's positive results come from interview
  protocols that make lying harder, not from passive measurement.

## What this direction can and cannot show

**Can show:** whether a compact, interpretable, theory-chosen feature set matches or beats a frozen
foundation model on the same splits. That is a real and useful comparison, and it costs an hour of
CPU.

**Cannot show:** that the cognitive approach "works". The eliciting half of it — the interview
protocol — is not in the data: DOLOS clips are answers without the questions.

**Cannot show:** anything about micro-expressions. They last 40–200 ms; the `tia` cache samples the
face every 312 ms on the median clip, so they are not representable at all. The annotated proxies
(`raising`, `frequent_blinking`) are measured here through their acoustic consequences only.

## The number to beat, and the number to fear

| Reference | Subject AUC | Parameters |
|---|---:|---:|
| `tia` audio branch (WavLM, frozen) | 58.9 (one seed) | 107k trainable + 94M frozen |
| `tia` fusion (WavLM + DINOv2) | 55.2 (three seeds) | 165k trainable |
| **MUMIN oracle** (41 hand annotations) | 52.5 | — |
| majority class | 50.0 | — |

The oracle is the ceiling that matters. It says: **even a human coding 41 behaviours while watching
the clip cannot predict the label.** Any feature set that reaches 55–59 is doing as well as the
behavioural signal allows; anything above 60 should be treated as a reason to check the split, not
as a discovery.

## What happened when it ran

Subject AUC **54.0** over three seeds, against 52.5 for the annotation oracle and 55.2 for the frozen
fusion: the features do as well as the hand-coded behaviour allows, and no better. That is the answer
the ceiling predicted.

The unexpected result is elsewhere. These features gain only **+4.7 AUC** from the leaky split, while
the frozen encoders gain **+25.2** on the same folds. Acoustic statistics do not encode a face or a
scene, so there is nothing for the clip-level split to leak. See
[`protocol-effect.md`](../../cross-modal-fusion/docs/protocol-effect.md) for the full comparison and
its one caveat (a smaller model has less capacity to exploit a leak even when one exists).
