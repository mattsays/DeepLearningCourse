"""
Builds cache/{stem}_mumin.npy (shape [41], float32 binary vector per clip)
from DOLOS train_dolos.xlsx (two-row header, data from row 2).

Column layout (verified):
    row 0: category headers (ignored)
    row 1: real column names
    col 0: filename
    col 1: label
    col 2: gender (not a MUMIN feature — excluded)
    cols 3..43: 41 MUMIN binary features (order preserved by index)
"""
import argparse
import os
import numpy as np
import pandas as pd

FIRST_MUMIN_COL = 3          # inclusive
LAST_MUMIN_COL  = 43         # inclusive
N_MUMIN         = LAST_MUMIN_COL - FIRST_MUMIN_COL + 1   # = 41


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="dolos/train_dolos.xlsx")
    p.add_argument("--cache", default="cache",
                   help="Feature cache dir; per-clip T is read from *_feats.npy.")
    p.add_argument("--out_cache", default=None,
                   help="Where to write mumin caches (default: same as --cache).")
    args = p.parse_args()

    out_cache = args.out_cache or args.cache
    os.makedirs(out_cache, exist_ok=True)

    df = pd.read_excel(args.xlsx, header=1)
    df.columns = [str(c).replace("\u200b", "").strip() for c in df.columns]

    written = 0
    missing_feats = 0
    skipped = 0

    for _, r in df.iterrows():
        name = str(r.iloc[0]).strip()
        if not name or name == "nan":
            continue
        stem = os.path.splitext(os.path.basename(name))[0].lower()

        # If the feature cache for this clip is missing, skip — the model
        # cannot train on it anyway.
        feat_path = os.path.join(args.cache, f"{stem}_feats.npy")
        if not os.path.exists(feat_path):
            missing_feats += 1
            continue

        # Extract MUMIN binary vector, coerce strings -> float
        mumin_vals = r.iloc[FIRST_MUMIN_COL:LAST_MUMIN_COL + 1].tolist()
        try:
            vec = np.array(
                [float(str(v).strip() or 0.0) for v in mumin_vals],
                dtype=np.float32,
            )
        except ValueError:
            skipped += 1
            print(f"[skip] {stem}: non-numeric MUMIN values {mumin_vals}")
            continue

        if vec.shape[0] != N_MUMIN:
            skipped += 1
            print(f"[skip] {stem}: expected {N_MUMIN} MUMIN dims, got {vec.shape[0]}")
            continue

        np.save(os.path.join(out_cache, f"{stem}_mumin.npy"), vec)
        written += 1

    print(f"wrote {written} mumin caches to {out_cache}")
    print(f"skipped {missing_feats} clips without feats, "
          f"{skipped} clips with malformed MUMIN")

if __name__ == "__main__":
    main()