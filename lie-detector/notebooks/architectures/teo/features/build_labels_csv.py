"""
Builds data/labels.csv (columns: video_id,label; 1=deception, 0=truth)
from DOLOS train_dolos.xlsx (two-row header, data from row 2).

Schema (verified):
    row 0: category headers (ignored)
    row 1: real column names
    row 2+: data
    col 0: filename (e.g. AN_WILTY_EP15_truth1)
    col 1: label ('truth' or 'deception')
"""
import argparse
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="dolos/train_dolos.xlsx")
    p.add_argument("--out", default="data/labels.csv")
    args = p.parse_args()

    # Skip row 0 (category headers); row 1 becomes the column header.
    df = pd.read_excel(args.xlsx, header=1)

    # Column names may have trailing whitespace / zero-width spaces
    df.columns = [str(c).replace("\u200b", "").strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        name = str(r.iloc[0]).strip()
        label_raw = str(r.iloc[1]).strip().lower()
        if not name or name == "nan":
            continue
        stem = os.path.splitext(os.path.basename(name))[0].lower()
        if label_raw in ("deception", "lie", "deceptive", "1"):
            label = 1
        elif label_raw in ("truth", "honest", "0"):
            label = 0
        else:
            print(f"[skip] unknown label {label_raw!r} for {name}")
            continue
        rows.append({"video_id": stem, "label": label})

    out_df = pd.DataFrame(rows).drop_duplicates("video_id")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} rows to {args.out}")
    print(out_df["label"].value_counts().to_string())

if __name__ == "__main__":
    main()