"""Late fusion of the video and audio branches, and the cross-validation report.

Per fold, from the per-clip logits each branch saved:

* each branch's logits are standardised with the mean/std of its own ``val`` logits, so the two
  scales are comparable;
* fused score ``w * z_video + (1 - w) * z_audio``; ``w`` in {0, 0.05, ..., 1} is chosen by AUC on
  ``val`` (ties go to the weight closest to 0.5); ``w = 0.5`` is reported too, as an untuned reference;
* the threshold is chosen on ``val`` (``lib.metrics.best_threshold``), the holdout is scored with
  ``lib.metrics.evaluate`` — exactly as for a single branch.

The board number is the mean +- std over the fold holdouts (``cv_summary``). The pooled view
(every working-set clip once, each judged with its own fold's threshold) gives the per-gender
breakdown, the confusion matrix and the majority baseline on the same clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("acc", "f1", "auc", "balanced_acc")
WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 2)


# --------------------------------------------------------------------------- #
# summaries (also used by train.py)
# --------------------------------------------------------------------------- #
def cv_summary(holdouts: list[dict]) -> dict:
    """Mean and sample std (ddof=1, as the team template) of each metric over the folds."""
    out: dict = {"n_folds": len(holdouts)}
    for metric in METRICS:
        values = np.array([h[metric] for h in holdouts], dtype=float)
        out[metric] = float(np.nanmean(values))
        out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def format_cv_table(name: str, results: list[dict]) -> str:
    lines = [f"{name}", f"{'fold':>4}  {'ACC':>5}  {'F1':>5}  {'AUC':>5}  {'bal':>5}  {'thr':>5}  {'best ep':>7}"]
    for r in results:
        h = r["holdout"]
        lines.append(f"{r['fold']:>4}  {h['acc'] * 100:5.1f}  {h['f1'] * 100:5.1f}  {h['auc'] * 100:5.1f}  "
                     f"{h['balanced_acc'] * 100:5.1f}  {r['threshold']:5.2f}  {r.get('best_epoch', '-'):>7}")
    s = cv_summary([r["holdout"] for r in results])
    lines.append(f"{'CV':>4}  {s['acc'] * 100:5.1f}  {s['f1'] * 100:5.1f}  {s['auc'] * 100:5.1f}  "
                 f"{s['balanced_acc'] * 100:5.1f}   (ACC ± {s['acc_std'] * 100:.1f} over {s['n_folds']} folds)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> dict[int, dict]:
    """``fold -> {"val", "holdout", "metrics"}`` for every completed fold of a run."""
    run_dir = Path(run_dir)
    folds = {}
    for fold_dir in sorted(run_dir.glob("fold*")):
        marker = fold_dir / "metrics.json"
        if not marker.exists():
            continue
        # the paper's CSVs list a few clips twice (e.g. ls_wilty_ep14_lie11 in test_fold1), and the
        # official split keeps them as listed: one row per clip here, or the 1:1 merges below fail
        read = lambda name: pd.read_csv(fold_dir / name).drop_duplicates("clip_id", ignore_index=True)
        folds[int(fold_dir.name[4:])] = {
            "val": read("val_preds.csv"),
            "holdout": read("holdout_preds.csv"),
            "metrics": json.loads(marker.read_text(encoding="utf-8")),
        }
        if (fold_dir / "paper_test_preds.csv").exists():                # official protocol only
            folds[int(fold_dir.name[4:])]["paper_test"] = read("paper_test_preds.csv")
    if not folds:
        raise FileNotFoundError(f"no completed fold in {run_dir}")
    return folds


def _sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan")


# --------------------------------------------------------------------------- #
# per fold
# --------------------------------------------------------------------------- #
def _fold_result(fold: int, val_y, val_p, hold: pd.DataFrame, hold_p, tools, extra=None) -> dict:
    evaluate, best_threshold = tools
    threshold = best_threshold(val_y, val_p)
    preds = pd.DataFrame({"clip_id": hold.clip_id, "y_true": hold.y_true.astype(int),
                          "y_prob": np.round(hold_p, 6), "y_pred": (np.asarray(hold_p) >= threshold).astype(int),
                          "fold": fold, "gender": hold.gender, "subject": hold.subject})
    result = {"fold": fold, "threshold": threshold, "holdout": evaluate(hold.y_true, hold_p, threshold),
              "preds": preds}
    result.update(extra or {})
    return result


def branch_fold(fold: int, data: dict, tools) -> dict:
    """A single branch, threshold re-chosen on val (identical to what train.py did)."""
    return _fold_result(fold, data["val"].y_true, data["val"].prob, data["holdout"], data["holdout"].prob, tools,
                        {"best_epoch": data["metrics"].get("best_epoch")})


def fuse_fold(fold: int, video: dict, audio: dict, tools, weight: float | None = None) -> dict:
    keys = ["clip_id", "y_true"]
    val = video["val"].merge(audio["val"][keys + ["logit"]], on=keys, suffixes=("_v", "_a"), validate="1:1")
    hold = video["holdout"].merge(audio["holdout"][keys + ["logit"]], on=keys, suffixes=("_v", "_a"),
                                  validate="1:1")
    if len(val) != len(video["val"]) or len(hold) != len(video["holdout"]):
        raise RuntimeError(f"fold {fold}: the two branches were not evaluated on the same clips")

    def standardise(column: str):
        mean, std = val[column].mean(), val[column].std() + 1e-6
        return (val[column] - mean) / std, (hold[column] - mean) / std

    zv_val, zv_hold = standardise("logit_v")
    za_val, za_hold = standardise("logit_a")
    if weight is None:
        ranked = [(_auc(val.y_true, w * zv_val + (1 - w) * za_val), -abs(w - 0.5), w) for w in WEIGHTS]
        weight = float(max(ranked)[2])
    val_p = _sigmoid(weight * zv_val + (1 - weight) * za_val)
    hold_p = _sigmoid(weight * zv_hold + (1 - weight) * za_hold)
    return _fold_result(fold, val.y_true, val_p, hold, hold_p, tools, {"video_weight": weight})


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def pooled_report(preds: pd.DataFrame, tools) -> dict:
    """Every holdout clip once, each judged with its own fold's threshold."""
    evaluate, _ = tools
    y, decision = preds.y_true.to_numpy(), preds.y_pred.to_numpy().astype(float)
    pooled = evaluate(y, decision, 0.5)                   # decisions are 0/1: threshold 0.5 keeps them
    pooled["auc"] = _auc(y, preds.y_prob)                 # ranking quality across folds (rough)
    majority = float(max(y.mean(), 1 - y.mean()))
    per_gender = {}
    for gender, group in preds.groupby("gender"):
        g = evaluate(group.y_true.to_numpy(), group.y_pred.to_numpy().astype(float), 0.5)
        per_gender[gender] = {"n": g["n"], "acc": g["acc"], "balanced_acc": g["balanced_acc"],
                              "lie_rate": float(group.y_true.mean())}
    return {"pooled": pooled, "majority": majority, "per_gender": per_gender}


def summarize(name: str, folds: list[dict], tools) -> dict:
    preds = pd.concat([f["preds"] for f in folds], ignore_index=True)
    out = {"name": name, "cv": cv_summary([f["holdout"] for f in folds]), **pooled_report(preds, tools),
           "per_fold": [{k: v for k, v in f.items() if k != "preds"} for f in folds]}
    return out, preds


def format_comparison(reports: list[dict]) -> str:
    header = (f"{'model':<26} {'ACC (CV)':>12} {'F1':>5} {'AUC':>5} {'bal':>5} | {'pooled':>6} "
              f"{'majority':>8} | {'male':>5} {'female':>6}")
    lines = [header, "-" * len(header)]
    for r in reports:
        cv, g = r["cv"], r["per_gender"]
        lines.append(f"{r['name']:<26} {cv['acc'] * 100:5.1f} ± {cv['acc_std'] * 100:4.1f} {cv['f1'] * 100:5.1f} "
                     f"{cv['auc'] * 100:5.1f} {cv['balanced_acc'] * 100:5.1f} | {r['pooled']['acc'] * 100:6.1f} "
                     f"{r['majority'] * 100:8.1f} | {g.get('male', {}).get('acc', float('nan')) * 100:5.1f} "
                     f"{g.get('female', {}).get('acc', float('nan')) * 100:6.1f}")
    return "\n".join(lines)


def _run_cost(folds: dict[int, dict]) -> tuple[int, float]:
    first = next(iter(folds.values()))["metrics"]
    minutes = sum(f["metrics"]["seconds"]["train_total"] for f in folds.values()) / 60
    return int(first["parameters"]["trainable"]), minutes


def _gpu(run_dir: Path) -> str:
    try:
        env = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["environment"]
        return env.get("gpu", env.get("device", "?"))
    except Exception:
        return "?"


def _paper_style_line(name: str, accs, f1s, aucs) -> str:
    accs, aucs = np.asarray(accs, dtype=float) * 100, np.asarray(aucs, dtype=float) * 100
    f1s = np.asarray([np.nan if f is None else f for f in f1s], dtype=float) * 100
    std = accs.std(ddof=1) if len(accs) > 1 else 0.0
    f1 = f"{np.nanmean(f1s):.1f}" if not np.isnan(f1s).all() else "  n/a"
    return (f"  {name:<6} ACC {accs.mean():.1f} ± {std:.1f}  F1 {f1}  AUC {aucs.mean():.1f}   "
            f"(per fold ACC {', '.join(f'{a:.1f}' for a in accs)})")


def paper_style_fusion(video: dict[int, dict], audio: dict[int, dict], weights: dict[int, float],
                       tools) -> list[dict] | None:
    """Per fold: each branch's test predictions at its own best-test epoch, fused as
    ``w * logit_v + (1 - w) * logit_a`` (``w`` = the val-tuned weight of that fold), threshold 0.5.
    ``None`` when a fold lacks ``paper_test_preds.csv`` (runs trained before it was written)."""
    evaluate, _ = tools
    common = sorted(set(video) & set(audio))
    if not common or any("paper_test" not in video[k] or "paper_test" not in audio[k] for k in common):
        return None
    out = []
    keys = ["clip_id", "y_true"]
    for k in common:
        v, a = video[k]["paper_test"], audio[k]["paper_test"]
        both = v[keys + ["logit"]].merge(a[keys + ["logit"]], on=keys, suffixes=("_v", "_a"), validate="1:1")
        if len(both) != len(v) or len(both) != len(a):
            raise RuntimeError(f"fold {k}: the two branches were not tested on the same clips")
        w = weights[k]
        prob = _sigmoid(w * both.logit_v + (1 - w) * both.logit_a)
        out.append({"fold": k, "video_weight": w, "epochs": (int(v.epoch.iloc[0]), int(a.epoch.iloc[0])),
                    **evaluate(both.y_true.to_numpy(), np.asarray(prob), 0.5)})
    return out


def _paper_style_lines(runs: dict[str, dict[int, dict]], weights: dict[int, float] | None = None,
                       tools=None) -> str:
    """Official protocol: the paper's own selection (best epoch on the test fold, threshold 0.5)."""
    lines = ["", "paper-style: best epoch chosen ON THE TEST FOLD, threshold 0.5, as the paper's train_test.py",
             "(optimistic; comparable with the paper's 64.75% only in this sense):"]
    for name, folds in runs.items():
        styles = [folds[k]["metrics"].get("paper_style") for k in sorted(folds)]
        if not all(styles):
            continue
        lines.append(_paper_style_line(name, [s["test_acc_at_05"] for s in styles],
                                       [s.get("test_f1_at_05") for s in styles], [s["test_auc"] for s in styles]))
    if weights and tools and len(runs) == 2:
        fused = paper_style_fusion(runs["video"], runs["audio"], weights, tools)
        if fused is None:
            lines.append("  fused  n/a (no paper_test_preds.csv: these folds were trained before it was saved)")
        else:
            lines.append(_paper_style_line("fused", [f["acc"] for f in fused], [f["f1"] for f in fused],
                                           [f["auc"] for f in fused]))
            lines.append("         (each branch at its own best test epoch, w tuned on val per fold: "
                         + ", ".join(f"fold {f['fold']} w={f['video_weight']:.2f}" for f in fused) + ")")
    return "\n".join(lines)


def fuse_runs(video_dir: Path | None, audio_dir: Path | None, out_dir: Path, lib_dir: Path,
              owner: str = "—", direction: str = "lore", folds: list[int] | None = None) -> dict:
    """Report for whatever branches exist; fused rows when both do. Writes preds + board row."""
    from .config import use_team_lib

    use_team_lib(lib_dir)
    from lib.metrics import best_threshold, evaluate, format_row

    tools = (evaluate, best_threshold)
    runs = {name: load_run(d) for name, d in (("video", video_dir), ("audio", audio_dir)) if d}
    if folds:                                                # e.g. the pilot: fold 1 only
        runs = {name: {k: v for k, v in got.items() if k in folds} for name, got in runs.items()}
        empty = [name for name, got in runs.items() if not got]
        if empty:
            raise FileNotFoundError(f"no completed fold among {folds} for: {', '.join(empty)}")
    out_dir = Path(out_dir)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    reports, predictions = [], {}
    for name, folds in runs.items():
        report, preds = summarize(f"{name} only", [branch_fold(k, folds[k], tools) for k in sorted(folds)], tools)
        reports.append(report)
        predictions[name] = preds
    if len(runs) == 2:
        common = sorted(set(runs["video"]) & set(runs["audio"]))
        for label, weight in (("fused (w tuned on val)", None), ("fused (w = 0.5)", 0.5)):
            fused = [fuse_fold(k, runs["video"][k], runs["audio"][k], tools, weight) for k in common]
            report, preds = summarize(label, fused, tools)
            reports.append(report)
            predictions[label] = preds

    best = reports[-2] if len(runs) == 2 else reports[0]      # the tuned fusion is the direction's result
    best_preds = predictions[best["name"]]
    best_preds[["clip_id", "y_true", "y_prob", "y_pred", "fold"]].to_csv(out_dir / "preds" / f"cv_{direction}.csv",
                                                                         index=False)
    for name in runs:
        predictions[name][["clip_id", "y_true", "y_prob", "y_pred", "fold"]].to_csv(
            out_dir / "preds" / f"cv_{direction}-{name}.csv", index=False)

    params, minutes, gpus = 0, 0.0, set()
    for name, folds in runs.items():
        p, m = _run_cost(folds)
        params, minutes = params + p, minutes + m
        gpus.add(_gpu(video_dir if name == "video" else audio_dir))
    method = " + ".join({"video": "VideoMAE-B", "audio": "HuBERT-B"}[n] for n in runs) + " · LoRA"
    if len(runs) == 2:
        method += " · late fusion"
    cv = best["cv"]
    row = format_row(direction, {"acc": cv["acc"], "acc_std": cv["acc_std"], "f1": cv["f1"], "auc": cv["auc"],
                                 "balanced_acc": cv["balanced_acc"]}, owner=owner, method=method,
                     params=params, majority=best["majority"],
                     runtime=f"{minutes:.0f} min train ({', '.join(sorted(gpus))})")

    table = format_comparison(reports)
    official = any(d is not None and "official" in Path(d).name for d in (video_dir, audio_dir))
    if official:
        tuned = next((r for r in reports if r["name"] == "fused (w tuned on val)"), None)
        weights = {f["fold"]: f["video_weight"] for f in tuned["per_fold"]} if tuned else None
        table += "\n" + _paper_style_lines(runs, weights, tools)
        row = "NOT FOR THE RESULTS BOARD: official clip-level folds (subject leakage), paper comparison only"
    payload = {"reports": reports, "board_row": row, "selected": best["name"],
               "runs": {k: str(v) for k, v in (("video", video_dir), ("audio", audio_dir)) if v}}
    (out_dir / "fusion_report.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    if not official:
        (out_dir / "board_row.md").write_text(row + "\n", encoding="utf-8")
    (out_dir / "comparison.txt").write_text(table + "\n", encoding="utf-8")
    return {"table": table, "board_row": row, "reports": reports, "out_dir": str(out_dir)}
