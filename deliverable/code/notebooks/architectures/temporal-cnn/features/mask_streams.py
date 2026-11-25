"""Single-stream copies of the TemporalCNN feature cache, for the unimodal ablation.

    python mask_streams.py --cache ../../../../../dolos/cached_features/teo-landmarks-egemaps

Each ``{clip_id}_feats.npy`` is T x 53: columns 0-44 are the 15 landmark regions (x, y, z) and
columns 45-52 the 8 eGeMAPS descriptors (``extract_features.py``). This writes two sibling caches,
``<cache>-video`` (audio columns set to 0) and ``<cache>-audio`` (landmark columns set to 0), and
copies the ``_mumin.npy`` files. A zeroed input channel contributes nothing to a convolution, so the
network is the same as one built without that stream, apart from its unused weights.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

VIDEO = slice(0, 45)
AUDIO = slice(45, 53)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    targets = {"video": AUDIO, "audio": VIDEO}       # stream kept -> columns zeroed
    for keep, zero in targets.items():
        out = a.cache.with_name(f"{a.cache.name}-{keep}")
        out.mkdir(exist_ok=True)
        n = 0
        for f in a.cache.glob("*_feats.npy"):
            x = np.load(f)
            assert x.shape[1] == 53, (f.name, x.shape)
            x = x.copy()
            x[:, zero] = 0.0
            np.save(out / f.name, x)
            n += 1
        for f in a.cache.glob("*_mumin.npy"):
            shutil.copyfile(f, out / f.name)
        print(f"{keep}-only: {n} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
