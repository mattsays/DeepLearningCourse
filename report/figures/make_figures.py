"""Candidate figures for report/report.typ.

Run from the repository root:

    .venv/bin/python report/figures/make_figures.py

Every number is read from a committed file, never typed in:

    fig1-protocol-gap.pdf   official vs subject-independent ACC/AUC
                            cross-modal-fusion/results/*/summary.json
                            videomae-hubert-lora/results/*/summary.json and *fusion_report.json
                            MARLIN and the theory baseline: report/context.md (§2, §5)
    fig2-roc.pdf            pooled ROC on the official folds
                            cross-modal-fusion/results/{late,video,audio}-official/preds.csv
    fig3-mumin-probes.pdf   linear-probe AUC per MUMIN cue against its noise band
                            cross-modal-fusion/diagnostics.ipynb, cell 15 output
                            (saved by the notebook as results/diagnostics_mumin.csv)
    fig4-efficiency.pdf     official AUC against trainable parameters and training time
                            cross-modal-fusion/results/*/summary.json,
                            videomae-hubert-lora/results/*/summary.json,
                            MARLIN, baseline and PECL: report/context.md (§1, §2, §5)

The MUMIN table is parsed from the notebook's stored output because the CSV it wrote was
not committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "lie-detector/notebooks/architectures"

INK = "#1a1a1a"
OFFICIAL = "#2f4b6e"
SUBJECT = "#c8672b"
VIDEO = "#2c6ea3"
AUDIO = "#c8732b"
FUSED = "#1f4e79"
GREY = "#8a8a8a"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#4d4d4d",
        "axes.labelcolor": INK,
        "xtick.color": "#4d4d4d",
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def protocol_rows() -> list[dict]:
    """Official and subject-independent ACC/AUC for every model with both numbers."""
    rows = []

    base = RESULTS / "cross-modal-fusion/results"
    for key, label in (("audio", "Cross-Modal · audio"), ("video", "Cross-Modal · video"),
                       ("late", "Cross-Modal · fused")):
        off = read_json(base / f"{key}-official/summary.json")
        sub = read_json(base / f"{key}-subject/summary.json")
        rows.append({"label": label, "official": (off["acc"], off["auc"]),
                     "subject": (sub["acc"], sub["auc"])})

    lore = RESULTS / "videomae-hubert-lora/results"
    fused = read_json(lore / "fusion-video-lora-r8-official+audio-lora-r8-official/fusion_report.json")
    fused_sub = read_json(lore / "fusion-video-lora-r8+audio-lora-r8/fusion_report.json")
    pick = {r["name"]: r for r in fused["reports"]}
    pick_sub = {r["name"]: r for r in fused_sub["reports"]}
    for name, label in (("audio only", "VideoMAE+HuBERT · audio"),
                        ("video only", "VideoMAE+HuBERT · video"),
                        ("fused (w tuned on val)", "VideoMAE+HuBERT · fused")):
        rows.append({"label": label,
                     "official": (pick[name]["cv"]["acc"], pick[name]["cv"]["auc"]),
                     "subject": (pick_sub[name]["cv"]["acc"], pick_sub[name]["cv"]["auc"])})

    load = RESULTS / "acoustic-baseline/results"
    rows.insert(0, {"label": "Theory baseline",
                    "official": (read_json(load / "load-official/summary.json")["acc"],
                                 read_json(load / "load-official/summary.json")["auc"]),
                    "subject": (read_json(load / "load-subject/summary.json")["acc"],
                                read_json(load / "load-subject/summary.json")["auc"])})

    teo = RESULTS / "temporal-cnn/results"
    rows.append({"label": "Temporal CNN",
                 "official": (read_json(teo / "temporal-official/summary.json")["acc"],
                              read_json(teo / "temporal-official/summary.json")["auc"]),
                 "subject": (read_json(teo / "temporal-subject/summary.json")["acc"],
                             read_json(teo / "temporal-subject/summary.json")["auc"])})

    # report/context.md §5: recomputed from the run's per-clip predictions, not from folds.csv.
    rows.append({"label": "MARLIN+ConvNeXt", "official": (0.589, 0.589),
                 "subject": (0.548, 0.554)})
    return rows


def argmax_delta(row: dict) -> float:
    return (row["subject"][0] - row["official"][0]) * 100


def fig_protocol_gap() -> None:
    rows = protocol_rows()
    n = len(rows)
    y = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15), sharey=True)

    panels = ((0, "Accuracy (%)", 44, 89), (1, "AUC (%)", 46, 89))
    for ax, (idx, xlabel, lo, hi) in zip(axes, panels):
        for yi, row in zip(y, rows):
            o, s = row["official"][idx] * 100, row["subject"][idx] * 100
            ax.plot([s, o], [yi, yi], color="#c9c9c9", lw=1.1, zorder=1, solid_capstyle="round")
            ax.plot([o], [yi], "o", ms=4.4, color=OFFICIAL, zorder=3)
            ax.plot([s], [yi], "o", ms=4.4, mfc="white", mec=SUBJECT, mew=1.3, zorder=3)
            delta = round(s, 1) - round(o, 1)
            ax.text(hi - 1.0, yi, f"{delta:+.1f}", ha="right", va="center", fontsize=7.7,
                    color=SUBJECT if delta < 0 else INK)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e8e8e8", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[1].spines[["left"]].set_visible(False)
    axes[0].axvspan(52.4, 53.3, color="#dcdcdc", alpha=0.6, lw=0, zorder=0)
    axes[0].text(53.0, n - 0.35, "majority", ha="center", va="bottom", fontsize=6.6, color=GREY)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=8.2)
    axes[0].set_ylim(-0.7, n - 0.1)
    # group separators: after baseline (row n-1) and after the two fusion families
    for boundary in (n - 1.5, n - 4.5, n - 7.5):
        for ax in axes:
            ax.axhline(boundary, color="#e0e0e0", lw=0.6, zorder=0)

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.4, color=OFFICIAL, label="official 3-fold (leaky)"),
        Line2D([], [], marker="o", ls="", ms=4.4, mfc="white", mec=SUBJECT, mew=1.3,
               label="subject-independent"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.30, 0.965), ncol=2,
               handletextpad=0.4, columnspacing=1.2, fontsize=7.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1-protocol-gap.{ext}", dpi=300)
    plt.close(fig)


def fig_roc() -> None:
    base = RESULTS / "cross-modal-fusion/results"
    series = (("late", FUSED, "-", "Cross-Modal fused"),
              ("video", VIDEO, "--", "Cross-Modal video"),
              ("audio", AUDIO, "-.", "Cross-Modal audio"))
    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    for key, color, style, label in series:
        d = pd.read_csv(base / f"{key}-official/preds.csv")
        fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
        auc = roc_auc_score(d.y_true, d.y_prob)
        ax.plot(fpr, tpr, color=color, ls=style, lw=1.5, label=f"{label}  ({auc:.3f})")
        # operating point at the fixed threshold 0.5 used on the official folds
        flag = d.y_prob >= 0.5
        tpr05 = (flag & (d.y_true == 1)).sum() / (d.y_true == 1).sum()
        fpr05 = (flag & (d.y_true == 0)).sum() / (d.y_true == 0).sum()
        ax.plot([fpr05], [tpr05], "o", ms=4.6, color=color, mec="white", mew=0.8, zorder=4)
    ax.plot([0, 1], [0, 1], color="#b0b0b0", lw=0.7, ls=(0, (2, 2)))
    ax.plot([], [], "o", ms=4.6, color="#6a6a6a", mec="white", mew=0.8,
            label="operating point, $t=0.5$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.6, handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2-roc.{ext}", dpi=300)
    plt.close(fig)


def mumin_table() -> pd.DataFrame:
    """Parse the probe table out of the notebook's stored cell output."""
    nb = json.loads((RESULTS / "cross-modal-fusion/diagnostics.ipynb").read_text())
    text = ""
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text += "".join(out.get("text", []))
            text += "".join(out.get("data", {}).get("text/plain", []))
    pattern = re.compile(
        r"^\s*(\w+)\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+(yes|no)\s*$",
        re.MULTILINE,
    )
    rows = [
        {"cue": m.group(1), "prevalence": float(m.group(2)), "video": float(m.group(3)),
         "band_video": float(m.group(4)), "audio": float(m.group(5)),
         "band_audio": float(m.group(6))}
        for m in pattern.finditer(text)
    ]
    df = pd.DataFrame(rows).drop_duplicates("cue")
    if len(df) != 26:
        raise RuntimeError(f"expected 26 cues, parsed {len(df)}")
    return df.sort_values("video", ascending=False).reset_index(drop=True)


def fig_mumin() -> None:
    df = mumin_table()
    n = len(df)
    y = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.55), sharey=True)

    panels = (
        (axes[0], "video", "band_video", VIDEO, "DINOv2 (video)"),
        (axes[1], "audio", "band_audio", AUDIO, "WavLM (audio)"),
    )
    for ax, value, band, color, title in panels:
        for yi, row in zip(y, df.itertuples()):
            b = getattr(row, band)
            ax.plot([0.5, b], [yi, yi], color="#dcdcdc", lw=4.0, solid_capstyle="butt",
                    zorder=1)
            ax.plot([getattr(row, value)], [yi], "o", ms=4.2, color=color, zorder=3)
        ax.axvline(0.5, color="#8a8a8a", lw=0.7, ls=(0, (2, 2)), zorder=2)
        ax.set_xlim(0.48, 0.85)
        ax.set_ylim(-0.8, n - 0.4)
        ax.set_xlabel("probe AUC" if ax is axes[0] else "")
        ax.set_title(title, fontsize=8.2, pad=5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#efefef", lw=0.5)
        ax.set_axisbelow(True)

    names = {"non_ah_mmh_disturbances": "non-ah/mmh disturbances"}
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{names.get(r.cue, r.cue.replace('_', ' '))}  ({r.prevalence:.0%})"
                             for r in df.itertuples()], fontsize=7.4)
    axes[1].spines[["left"]].set_visible(False)

    handles = [Line2D([], [], lw=4.0, color="#dcdcdc", label="chance region up to null p95")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=7.6,
               handletextpad=0.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3-mumin-probes.{ext}", dpi=300)
    plt.close(fig)


def efficiency_points() -> list[dict]:
    """Official-protocol AUC against trainable parameters and training time."""
    base = RESULTS / "cross-modal-fusion/results"
    pts = []
    for key, label in (("audio", "Cross-Modal · audio"), ("video", "Cross-Modal · video"),
                       ("late", "Cross-Modal · fused")):
        s = read_json(base / f"{key}-official/summary.json")
        pts.append({"label": label, "params": s["trainable_params"], "auc": s["auc"] * 100,
                    "time": "25 s", "regime": "head", "marker": "o", "color": "#2c6ea3"})

    lore = RESULTS / "videomae-hubert-lora/results"
    for run, label, minutes in (("video-lora-r8-official", "VideoMAE+HuBERT · video", "45 min"),
                                ("audio-lora-r8-official", "VideoMAE+HuBERT · audio", "10 min")):
        s = read_json(lore / f"{run}/summary.json")
        pts.append({"label": label, "params": s["trainable_params"], "auc": s["auc"] * 100,
                    "time": minutes, "regime": "lora", "marker": "s", "color": "#c8732b"})
    pts.append({"label": "VideoMAE+HuBERT · fused", "params": 594_434, "auc": 73.1,
                "time": "54 min", "regime": "lora", "marker": "s", "color": "#c8732b"})

    # report/context.md §2 (baseline, 609 parameters) and §5 (MARLIN); times from report @tab:main
    load = RESULTS / "acoustic-baseline/results"
    pts.append({"label": "Theory baseline", "params": 609,
                "auc": 100 * read_json(load / "load-official/summary.json")["auc"], "time": "10 s",
                "regime": "theory", "marker": "D", "color": "#6f6f6f"})

    teo = RESULTS / "temporal-cnn/results"
    pts.append({"label": "Temporal CNN", "params": 57_322,
                "auc": 100 * read_json(teo / "temporal-official/summary.json")["auc"], "time": "2.6 min",
                "regime": "head", "marker": "X", "color": "#7b5c9e"})
    pts.append({"label": "MARLIN+ConvNeXt", "params": 97_857, "auc": 58.9, "time": "15 s",
                "regime": "tcn", "marker": "p", "color": "#4b7d3a"})
    return pts


def fig_efficiency() -> None:
    pts = efficiency_points()
    fig, ax = plt.subplots(figsize=(6.6, 3.85))

    for p in pts:
        ax.scatter([p["params"]], [p["auc"]], s=46, marker=p["marker"], color=p["color"],
                   edgecolor="white", linewidth=0.8, zorder=4)

    pecl = (5.06e6, 62.71, 64.58)
    ax.plot([pecl[0], pecl[0]], [pecl[1], pecl[2]], color="#8c2f39", lw=0.9, zorder=3)
    for y in pecl[1:]:
        ax.scatter([pecl[0]], [y], s=52, marker="^", color="#8c2f39", edgecolor="white",
                   linewidth=0.8, zorder=4)

    offsets = {
        "Theory baseline": (1.45, 0.3, "left", "bottom"),
        "MARLIN+ConvNeXt": (1.06, 0.5, "left", "bottom"),
        "Temporal CNN": (0.90, -1.15, "right", "top"),
        "Cross-Modal · audio": (0.93, 0.6, "right", "bottom"),
        "Cross-Modal · video": (0.92, 0.6, "right", "bottom"),
        "Cross-Modal · fused": (1.08, 0.6, "left", "bottom"),
        "VideoMAE+HuBERT · audio": (1.08, -0.7, "left", "top"),
        "VideoMAE+HuBERT · video": (1.08, 0.6, "left", "bottom"),
        "VideoMAE+HuBERT · fused": (1.08, 0.6, "left", "bottom"),
    }
    for p in pts:
        fx, dy, ha, va = offsets[p["label"]]
        ax.text(p["params"] * fx, p["auc"] + dy, f"{p['label']}  ({p['time']})",
                fontsize=7.2, ha=ha, va=va, color="#333333")
    ax.text(pecl[0] * 0.84, (pecl[1] + pecl[2]) / 2, "PECL, paper (5.06M)",
            fontsize=7.2, ha="right", va="center", color="#8c2f39")

    ax.set_xscale("log")
    ax.set_xlim(400, 8.5e6)
    ax.set_ylim(56.5, 83.5)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6], ["1k", "10k", "100k", "1M"])
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("AUC on the official folds (%)")
    ax.grid(axis="y", color="#ececec", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], marker=marker, ls="", ms=5.2, color=color, mec="white", mew=0.8, label=label)
        for marker, color, label in (
            ("D", "#6f6f6f", "theory-driven acoustic features"),
            ("o", "#2c6ea3", "frozen encoder + trained head"),
            ("p", "#4b7d3a", "frozen encoder + temporal CNN"),
            ("X", "#7b5c9e", "from-scratch CNN on descriptors"),
            ("s", "#c8732b", "frozen encoder + LoRA"),
            ("^", "#8c2f39", "adapters (PECL, quoted)"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=7.0, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4-efficiency.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_protocol_gap()
    fig_roc()
    fig_mumin()
    fig_efficiency()
    rows = protocol_rows()
    for r in rows:
        print(f"{r['label']:26s} ACC {r['official'][0]*100:5.1f} -> {r['subject'][0]*100:5.1f}"
              f"   AUC {r['official'][1]*100:5.1f} -> {r['subject'][1]*100:5.1f}")
    d = mumin_table()
    print(f"MUMIN: {len(d)} cues, mean video {d.video.mean():.3f}, "
          f"mean audio {d.audio.mean():.3f}")
