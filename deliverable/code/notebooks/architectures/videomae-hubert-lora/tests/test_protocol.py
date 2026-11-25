"""Protocol and data checks on the local DOLOS copy (skipped when ``dolos/`` is not there).

    python tests/test_protocol.py          (or: python -m pytest tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths  # noqa: E402
from lore.faces import make_sprite, split_sprite  # noqa: E402

PATHS = Paths.resolve()
HAVE_DATA = (PATHS.data / "splits" / "manifest.json").exists()


def test_sprite_roundtrip():
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(32)]
    back = split_sprite(make_sprite(frames), 32, 16)
    assert back.shape == (32, 16, 16, 3)
    assert all(int(back[i, 0, 0, 0]) == i for i in range(32))


def _study():
    from lore.data import Study

    return Study(PATHS)


def test_folds_never_touch_the_test_set():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    study = _study()
    test_ids = set(study.protocol.test())
    test_subjects = {study.clips[c].subject for c in test_ids}
    assert not test_ids & set(study.working_ids())
    for k in range(1, study.n_folds + 1):
        train, val, hold = study.fold(k)
        subjects = [set(study.subjects(part)) for part in (train, val, hold)]
        assert not (subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2])
        assert not (set(train) | set(val) | set(hold)) & test_ids
        assert not set().union(*subjects) & test_subjects


def test_debug_subsets_are_stratified_subsets():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    study = _study()
    full = study.fold(1)
    small = study.fold(1, debug=True)
    for part_small, part_full, n in zip(small, full, (160, 40, 40)):
        assert len(part_small) == min(n, len(part_full))
        assert set(part_small) <= set(part_full)
        assert 0 < study.labels(part_small).mean() < 1


def test_audio_files_exist_for_every_protocol_clip():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    study = _study()
    missing = [c for c in study.working_ids() if not (PATHS.audio / f"{c}.wav").exists()]
    assert not missing, missing[:5]
    empty = [c for c in study.working_ids() if study.clips[c].audio is None]
    print(f"clips whose dataset.csv audio_path is empty but the wav exists: {len(empty)}")


def test_official_protocol_is_the_papers_split_with_a_carved_val():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    from lore.data import Study

    study, again = Study(PATHS, "official"), Study(PATHS, "official")
    assert study.n_folds == 3
    for k in range(1, 4):
        paper_train, paper_test = study.protocol.official(k)
        train, val, test = study.fold(k)
        assert test == list(paper_test)                                   # the test fold is untouched
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        assert set(train) | set(val) == set(paper_train) - set(paper_test)
        assert abs(len(val) / (len(train) + len(val)) - 0.15) < 0.01
        assert again.fold(k)[1] == val                                    # same val in every run
    assert set(Study(PATHS).fold(1)[0]) != set(study.fold(1)[0])          # default is still the subject protocol


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
