"""Training of one branch (video or audio): one fold, and the resumable cross-validation driver.

Per fold, following the team protocol (``lie-detector/docs/evaluation.md``):

* fit on ``train`` (the two captains included), class prior handled by ``pos_weight``;
* early stopping on the AUC of ``val``; the decision threshold is chosen on ``val`` with
  ``lib.metrics.best_threshold`` (never 0.5 by default, never on the holdout);
* the fold's score is the ``holdout``, computed with ``lib.metrics.evaluate``.

Everything a fold produces goes to ``<runs>/<run_name>/fold<k>/``; ``metrics.json`` is written
last and marks the fold as done. ``checkpoint.pt`` (trainable weights, optimizer, scheduler, RNG,
history) is rewritten after every epoch and deleted when the fold ends, so a Colab/Kaggle
disconnect costs at most the epoch in progress: re-running the same command resumes from it.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Paths, RunConfig
from .data import AudioClips, AudioStore, BucketBatchSampler, FaceClips, Study, audio_collate, subject_weights
from .faces import file_sha256
from .fusion import cv_summary, format_cv_table
from .models import assert_lora_active, build_branch, count_parameters

IGNORED_FOR_RESUME = {"num_workers", "grad_checkpointing"}   # may change between sessions, results do not


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_settings(device: torch.device):
    """bf16 on Ampere or newer (A100, L4); fp16 + GradScaler on older GPUs (T4); fp32 on CPU."""
    if device.type != "cuda":
        return None, False
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16, False
    return torch.float16, True


def environment(device: torch.device) -> dict:
    import peft
    import transformers

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1)
    return info


def _git_commit(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class Runner:
    """Cross-validation of one branch configuration."""

    def __init__(self, cfg: RunConfig, paths: Paths, device: str | None = None, verbose: bool = True,
                 stop_after_epochs: int | None = None):
        self.cfg = cfg
        self.stop_after_epochs = stop_after_epochs      # simulate a session cut: raise after N epochs
        # LORE_DEADLINE (unix time): do not start an epoch that would end after it (Kaggle's 12 h limit)
        self.deadline = float(os.environ["LORE_DEADLINE"]) if os.environ.get("LORE_DEADLINE") else None
        self.paths = paths
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp_dtype, self.use_scaler = amp_settings(self.device)
        self.verbose = verbose
        self.study = Study(paths, cfg.protocol)      # puts the team lib on sys.path
        from lib.metrics import best_threshold, evaluate, majority_baseline

        self._evaluate, self._best_threshold, self._majority = evaluate, best_threshold, majority_baseline
        self.run_dir = Path(paths.runs) / cfg.run_name
        self._store: AudioStore | None = None

    # ------------------------------------------------------------------ utils
    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    # ------------------------------------------------------------------- data
    def audio_store(self) -> AudioStore:
        if self._store is None:
            start = time.time()
            self._store = AudioStore(self.paths.audio, self.study.working_ids())
            self.log(f"audio: {len(self.study.working_ids())} waveforms in RAM ({time.time() - start:.0f} s)")
        return self._store

    def dataset(self, ids: list[str], train: bool, offset: int = 0):
        labels = self.study.labels(ids)
        if self.cfg.modality == "video":
            return FaceClips(ids, labels, self.paths.faces, self.cfg.num_frames, train=train,
                             offset=offset, rrc_scale=self.cfg.rrc_scale)
        return AudioClips(ids, labels, self.audio_store(), train=train,
                          train_seconds=self.cfg.audio_train_seconds, eval_seconds=self.cfg.audio_eval_seconds)

    def loader(self, dataset, train: bool, seed: int) -> DataLoader:
        cfg, pin = self.cfg, self.device.type == "cuda"
        size = cfg.batch_size if train else cfg.batch_size * 2
        if cfg.modality == "audio":
            sampler = BucketBatchSampler(dataset.lengths(), size, shuffle=train, seed=seed)
            return DataLoader(dataset, batch_sampler=sampler, collate_fn=audio_collate,
                              num_workers=cfg.num_workers, pin_memory=pin)
        return DataLoader(dataset, batch_size=size, shuffle=train,
                          generator=torch.Generator().manual_seed(seed), num_workers=cfg.num_workers,
                          pin_memory=pin, persistent_workers=cfg.num_workers > 0)

    def to_device(self, batch):
        if self.cfg.modality == "video":
            frames, y, positions = batch
            return (frames.to(self.device, non_blocking=True),), y.to(self.device), positions
        waves, lengths, y, positions = batch
        return (waves.to(self.device, non_blocking=True), lengths.to(self.device)), y.to(self.device), positions

    def example_inputs(self, ids: list[str]):
        item = self.dataset(ids[:1], train=False)[0]
        if self.cfg.modality == "video":
            return (item[0].unsqueeze(0).to(self.device),)
        return (item[0].unsqueeze(0).to(self.device), torch.tensor([item[1]], device=self.device))

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, model, loader) -> np.ndarray:
        """Logits in dataset order (audio batches are length-sorted, so positions are restored)."""
        model.eval()
        logits, positions = [], []
        for batch in loader:
            inputs, _, pos = self.to_device(batch)
            with self.autocast():
                out, _ = model(*inputs)
            logits.append(out.float().cpu())
            positions.append(pos.cpu())
        order = torch.cat(positions).argsort()
        return torch.cat(logits)[order].numpy()

    def score(self, model, ids: list[str], seed: int) -> np.ndarray:
        """Final logits; with ``eval_tta`` the video logits are averaged over every frame offset."""
        offsets = [0]
        if self.cfg.modality == "video" and self.cfg.eval_tta:
            offsets = list(range(self.dataset(ids[:1], train=False).stride))
        total = 0.0
        for offset in offsets:
            total = total + self.predict(model, self.loader(self.dataset(ids, False, offset), False, seed))
        return total / len(offsets)

    # ------------------------------------------------------------------- runs
    def _check_deadline(self, k: int, epoch: int, history: list[dict]) -> None:
        """Stop cleanly (checkpoint already on disk) if the next epoch would overrun ``LORE_DEADLINE``.

        The estimate is the slowest epoch seen so far plus a 15-minute margin for the final
        val/holdout scoring; before the first epoch of a fresh fold it is a flat 45 minutes.
        """
        if self.deadline is None:
            return
        needed = (max(h["train_s"] + h["val_s"] for h in history) if history else 30 * 60) + 15 * 60
        left = self.deadline - time.time()
        if needed > left:
            raise EpochBudgetReached(f"fold {k}: epoch {epoch} not started, {left / 60:.0f} min left before "
                                     f"LORE_DEADLINE (need ~{needed / 60:.0f}); resume in a new session")

    def write_run_json(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "run.json"
        config = self.cfg.to_dict()
        if target.exists():
            defaults = _jsonable(RunConfig(modality=self.cfg.modality, run_name="x").to_dict())
            previous = {**defaults, **json.loads(target.read_text(encoding="utf-8"))["config"]}
            current = _jsonable(config)
            differs = sorted(k for k in set(current) | set(previous)
                             if k not in IGNORED_FOR_RESUME and current.get(k) != previous.get(k))
            if differs:
                raise RuntimeError(f"{self.run_dir} already holds a run with a different configuration "
                                   f"({differs}); use another run name")
        payload = {
            "config": config,
            "paths": self.paths.as_dict(),
            "environment": environment(self.device),
            "data": self.study.identity(),
            "team_repo_commit": _git_commit(self.paths.lib),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.cfg.modality == "video":
            payload["face_cache"] = {
                "name": self.paths.faces.name,
                "index_sha256_12": file_sha256(self.paths.faces / "index.csv")[:12],
                "config": json.loads((self.paths.faces / "cache_config.json").read_text(encoding="utf-8")),
            }
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def run(self, folds: list[int] | None = None) -> dict:
        folds = list(folds or range(1, self.study.n_folds + 1))
        self.write_run_json()
        self.log(f"run {self.cfg.run_name} on {self.device} (amp: {self.amp_dtype}) -> {self.run_dir}")
        results = [self.train_fold(k) for k in folds]
        summary = cv_summary([r["holdout"] for r in results])
        summary.update(run=self.cfg.run_name, folds=folds,
                       trainable_params=results[0]["parameters"]["trainable"],
                       train_minutes=round(sum(r["seconds"]["train_total"] for r in results) / 60, 1))
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log("\n" + format_cv_table(self.cfg.run_name, results))
        return summary

    def train_fold(self, k: int) -> dict:
        cfg = self.cfg
        fold_dir = self.run_dir / f"fold{k}"
        marker = fold_dir / "metrics.json"
        if marker.exists():
            self.log(f"fold {k}: already done ({marker}), skipped")
            return json.loads(marker.read_text(encoding="utf-8"))
        fold_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed + k)

        train_ids, val_ids, hold_ids = self.study.fold(k, debug=cfg.debug, limit=cfg.limit, seed=cfg.seed)
        y_train, y_val, y_hold = (self.study.labels(i) for i in (train_ids, val_ids, hold_ids))
        self.log(f"\n=== fold {k}: train {len(train_ids)} ({y_train.mean() * 100:.1f}% lie)  "
                 f"val {len(val_ids)}  holdout {len(hold_ids)} ===")

        model = build_branch(cfg).to(self.device)
        parameters = count_parameters(model)
        lora_check = assert_lora_active(model, *self.example_inputs(train_ids))
        self.log(f"trainable {parameters['trainable']:,} / {parameters['total']:,} "
                 f"({parameters['trainable_pct']}%)  lora check: {lora_check}")

        weights = (subject_weights(self.study.subjects(train_ids)) if cfg.subject_balanced
                   else np.ones(len(train_ids), dtype=np.float32))
        pos_weight = float(weights[y_train == 0].sum() / max(weights[y_train == 1].sum(), 1e-6))
        weight_t = torch.tensor(weights, device=self.device)
        pos_weight_t = torch.tensor(pos_weight, device=self.device)

        train_dl = self.loader(self.dataset(train_ids, train=True), True, cfg.seed + k)
        val_dl = self.loader(self.dataset(val_ids, train=False), False, cfg.seed + k)
        # official protocol only: also score the test fold every epoch, to report the paper's own
        # (optimistic) "best epoch chosen on test" number next to ours. Never used for any decision.
        paper_dl = (self.loader(self.dataset(hold_ids, train=False), False, cfg.seed + k)
                    if cfg.protocol == "official" else None)

        adapters, head = model.parameter_groups()
        groups = ([{"params": adapters, "lr": cfg.lr_lora}] if adapters else []) + \
                 [{"params": head, "lr": cfg.lr_head}]
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        steps_per_epoch = math.ceil(len(train_dl) / cfg.grad_accum)
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup = min(total_steps - 1, int(round(cfg.warmup_epochs * steps_per_epoch)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _warmup_cosine(step, warmup, total_steps))
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        trainable = [p for p in model.parameters() if p.requires_grad]

        best = {"auc": -1.0, "epoch": 0, "state": model.trainable_state()}
        history, waited, first_epoch, stopped = [], 0, 1, False
        checkpoint_path = fold_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_trainable_state(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            history, best, waited, stopped = ckpt["history"], ckpt["best"], ckpt["waited"], ckpt["stopped"]
            first_epoch = ckpt["epoch"] + 1
            _set_rng_state(ckpt["rng"], train_dl)
            pd.DataFrame(history).to_csv(fold_dir / "log.csv", index=False)
            self.log(f"fold {k}: resumed from epoch {ckpt['epoch']} (best val AUC {best['auc']:.3f} "
                     f"at epoch {best['epoch']})")
        for epoch in range(first_epoch, cfg.epochs + 1):
            if stopped:
                break
            self._check_deadline(k, epoch, history)
            if isinstance(train_dl.batch_sampler, BucketBatchSampler):
                train_dl.batch_sampler.set_epoch(epoch)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            model.train()
            start, running, seen = time.time(), 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_dl, 1):
                inputs, y, positions = self.to_device(batch)
                with self.autocast():
                    logits, _ = model(*inputs)
                per_clip = F.binary_cross_entropy_with_logits(logits.float(), y.float(),
                                                              pos_weight=pos_weight_t, reduction="none")
                w = weight_t[positions.to(self.device)]
                loss = (per_clip * w).sum() / w.sum()
                scaler.scale(loss / cfg.grad_accum).backward()
                running += float(loss.detach()) * len(y)
                seen += len(y)
                if step % cfg.grad_accum == 0 or step == len(train_dl):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            train_seconds = time.time() - start

            start = time.time()
            val_logits = self.predict(model, val_dl)
            val_seconds = time.time() - start
            val_auc = self._evaluate(y_val, _sigmoid(val_logits))["auc"]
            val_auc = val_auc if val_auc == val_auc else 0.0                    # NaN: one class only
            val_loss = float(F.binary_cross_entropy_with_logits(
                torch.tensor(val_logits), torch.tensor(y_val, dtype=torch.float32)))
            peak = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
            history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 4),
                            "val_loss": round(val_loss, 4), "val_auc": round(val_auc, 4),
                            "lr_adapters": optimizer.param_groups[0]["lr"], "train_s": round(train_seconds, 1),
                            "val_s": round(val_seconds, 1), "peak_gpu_gb": round(peak, 2)})
            if paper_dl is not None:
                start = time.time()
                test_logits = self.predict(model, paper_dl)
                test = self._evaluate(y_hold, _sigmoid(test_logits), 0.5)
                earlier = [h["test_acc_at_05"] for h in history[:-1] if "test_acc_at_05" in h]
                history[-1].update(test_acc_at_05=round(test["acc"], 4), test_f1_at_05=round(test["f1"], 4),
                                   test_auc=round(test["auc"], 4))
                history[-1]["val_s"] = round(history[-1]["val_s"] + time.time() - start, 1)
                if not earlier or history[-1]["test_acc_at_05"] > max(earlier):
                    # the test predictions of the paper-style epoch, for a paper-style fusion (fusion.py);
                    # written now, not at the end, so a resumed fold keeps them
                    pd.DataFrame({"clip_id": hold_ids, "y_true": y_hold, "logit": np.round(test_logits, 6),
                                  "prob": np.round(_sigmoid(test_logits), 6), "epoch": epoch}
                                 ).to_csv(fold_dir / "paper_test_preds.csv", index=False)
            pd.DataFrame(history).to_csv(fold_dir / "log.csv", index=False)
            improved = val_auc > best["auc"]
            self.log(f"fold {k} epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  "
                     f"val AUC {val_auc:.3f}{' *' if improved else ''}  {train_seconds:.0f}+{val_seconds:.0f} s  "
                     f"peak {peak:.1f} GB")
            if improved:
                best, waited = {"auc": val_auc, "epoch": epoch, "state": model.trainable_state()}, 0
            else:
                waited += 1
                stopped = waited >= cfg.patience
            _save_atomic({"epoch": epoch, "model": model.trainable_state(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "history": history, "best": best, "waited": waited, "stopped": stopped,
                          "rng": _rng_state(train_dl)}, checkpoint_path)
            if stopped:
                self.log(f"fold {k}: early stop, best epoch {best['epoch']}")
                break
            if self.stop_after_epochs and epoch - first_epoch + 1 >= self.stop_after_epochs:
                raise EpochBudgetReached(f"fold {k}: stopped after epoch {epoch} (checkpoint saved)")

        model.load_trainable_state(best["state"])
        val_logits = self.score(model, val_ids, cfg.seed + k)
        hold_logits = self.score(model, hold_ids, cfg.seed + k)
        p_val, p_hold = _sigmoid(val_logits), _sigmoid(hold_logits)
        threshold = self._best_threshold(y_val, p_val)
        metrics = {
            "fold": k, "run": cfg.run_name, "modality": cfg.modality,
            "best_epoch": best["epoch"], "epochs_run": len(history), "best_val_auc": round(best["auc"], 4),
            "threshold": threshold,
            "holdout": self._evaluate(y_hold, p_hold, threshold),
            "holdout_at_0.5": self._evaluate(y_hold, p_hold, 0.5),
            "val": self._evaluate(y_val, p_val, threshold),
            "majority_holdout": self._majority(y_hold),
            "sizes": {"train": len(train_ids), "val": len(val_ids), "holdout": len(hold_ids)},
            "pos_weight": round(pos_weight, 4),
            "parameters": parameters,
            "lora_check": lora_check,
            "seconds": {"train_total": round(sum(h["train_s"] for h in history), 1),
                        "train_per_epoch": round(float(np.mean([h["train_s"] for h in history])), 1),
                        "val_per_epoch": round(float(np.mean([h["val_s"] for h in history])), 1)},
            "peak_gpu_gb": max(h["peak_gpu_gb"] for h in history),
        }
        if paper_dl is not None:
            best_test = max(history, key=lambda h: h["test_acc_at_05"])
            metrics["paper_style"] = {
                "note": "best epoch picked ON THE TEST FOLD at threshold 0.5, as the paper's train_test.py "
                        "does: optimistic, only for comparison with the paper",
                "test_acc_at_05": best_test["test_acc_at_05"], "test_f1_at_05": best_test.get("test_f1_at_05"),
                "test_auc": best_test["test_auc"],
                "epoch": best_test["epoch"]}
        for name, ids, y, logits in (("val", val_ids, y_val, val_logits), ("holdout", hold_ids, y_hold, hold_logits)):
            pd.DataFrame({"clip_id": ids, "y_true": y, "logit": np.round(logits, 6),
                          "prob": np.round(_sigmoid(logits), 6), "subject": self.study.subjects(ids),
                          "gender": self.study.genders(ids)}).to_csv(fold_dir / f"{name}_preds.csv", index=False)
        torch.save(best["state"], fold_dir / "trainable.pt")
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        checkpoint_path.unlink(missing_ok=True)
        h = metrics["holdout"]
        self.log(f"fold {k} holdout: ACC {h['acc'] * 100:.1f}  F1 {h['f1'] * 100:.1f}  AUC {h['auc'] * 100:.1f}  "
                 f"bal {h['balanced_acc'] * 100:.1f}  (thr {threshold:.3f}, majority "
                 f"{metrics['majority_holdout'] * 100:.1f})")

        del model, optimizer, train_dl, val_dl, paper_dl
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics


class EpochBudgetReached(RuntimeError):
    """Raised by ``Runner(stop_after_epochs=N)`` once N epochs ran in this process (tests, manual budgets)."""


def _rng_state(train_dl: DataLoader) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "loader": train_dl.generator.get_state() if train_dl.generator is not None else None}
    return state


def _set_rng_state(state: dict, train_dl: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["loader"] is not None and train_dl.generator is not None:
        train_dl.generator.set_state(state["loader"])


def _save_atomic(payload: dict, target: Path) -> None:
    """Write next to the target, then rename: a session killed mid-save leaves the old checkpoint intact."""
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def _jsonable(value):
    """Tuples become lists in JSON: compare configurations in their JSON form."""
    return json.loads(json.dumps(value, default=str))
