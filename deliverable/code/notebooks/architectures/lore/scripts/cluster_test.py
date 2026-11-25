"""Speaker-cluster significance test for the lore subject-independent runs.

    python scripts/cluster_test.py --dataset ../../../../dolos/dataset.csv

Why: clips of one speaker are not independent, so a binomial test on the pooled holdout clips is
optimistic. This script resamples whole speakers (``subject`` column of ``dataset.csv``) and, on
each resample, measures the model against the majority rate *of that same resample*:

- ACC - majority: the gain over always predicting the more frequent class;
- AUC - 0.5:      threshold-free discrimination.

Inputs are the pooled out-of-fold predictions written by ``scripts/fuse.py``
(``results/fusion-video-lora-r8+audio-lora-r8/preds/cv_lore{,-audio,-video}.csv``): one row per
holdout clip, ``y_pred`` already thresholded at that clip's own fold threshold. Output: a table on
stdout and ``<fusion dir>/cluster_test.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
FUSION = HERE / "results" / "fusion-video-lora-r8+audio-lora-r8"
RUNS = {"video only": "cv_lore-video.csv", "audio only": "cv_lore-audio.csv",
        "fused (tuned)": "cv_lore.csv"}


def cluster_test(y, pred, prob, groups, n_boot, seed):
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    acc_gain, auc_gain = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=unique.size)])
        yy = y[idx]
        majority = max(yy.mean(), 1 - yy.mean())
        acc_gain.append((pred[idx] == yy).mean() - majority)
        if 0 < yy.sum() < yy.size:
            auc_gain.append(roc_auc_score(yy, prob[idx]) - 0.5)
    out = {}
    for name, draws, point in (
        ("acc_minus_majority", np.array(acc_gain),
         (pred == y).mean() - max(y.mean(), 1 - y.mean())),
        ("auc_minus_half", np.array(auc_gain), roc_auc_score(y, prob) - 0.5),
    ):
        out[name] = {
            "point": float(point),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            # one-sided: share of speaker resamples in which the model does not beat the reference
            "p_one_sided": float((draws <= 0).mean()),
            "n_boot": int(draws.size),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="dolos/dataset.csv")
    ap.add_argument("--fusion-dir", type=Path, default=FUSION)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.dataset, usecols=["clip_id", "y", "subject"]).set_index("clip_id")
    report = {}
    print(f"{'model':<15}{'n':>5}{'spk':>5}{'ACC':>7}{'maj':>7}  "
          f"{'ACC-maj [95% CI]':<24}{'p':>7}  {'AUC':>6}  {'AUC-0.5 [95% CI]':<24}{'p':>7}")
    for name, fname in RUNS.items():
        preds = pd.read_csv(args.fusion_dir / "preds" / fname).set_index("clip_id")
        df = preds.join(meta, how="inner")
        if len(df) != len(preds):
            raise SystemExit(f"{fname}: {len(preds) - len(df)} clips missing from dataset.csv")
        y = df["y"].to_numpy(int)
        res = cluster_test(y, df["y_pred"].to_numpy(int), df["y_prob"].to_numpy(float),
                           df["subject"].to_numpy(), args.n_boot, args.seed)
        a, u = res["acc_minus_majority"], res["auc_minus_half"]
        acc = (df["y_pred"] == y).mean()
        maj = max(y.mean(), 1 - y.mean())
        report[name] = {"n_clips": int(len(df)), "n_speakers": int(df["subject"].nunique()),
                        "acc": float(acc), "majority": float(maj), **res}
        print(f"{name:<15}{len(df):>5}{df['subject'].nunique():>5}{acc*100:>7.1f}{maj*100:>7.1f}  "
              f"{a['point']*100:+5.1f} [{a['lo']*100:+5.1f}, {a['hi']*100:+5.1f}]{'':<7}"
              f"{a['p_one_sided']:>7.3f}  {(u['point']+.5)*100:>6.1f}  "
              f"{u['point']*100:+5.1f} [{u['lo']*100:+5.1f}, {u['hi']*100:+5.1f}]{'':<7}"
              f"{u['p_one_sided']:>7.3f}")
    out = args.fusion_dir / "cluster_test.json"
    out.write_text(json.dumps({"n_boot": args.n_boot, "seed": args.seed, "runs": report}, indent=2))
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
