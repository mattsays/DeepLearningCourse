# AGENTS.md — `lie-detector/`

Project-level rules for the lie-detection work. They refine the repository-level
[`../AGENTS.md`](../AGENTS.md) for this subtree; where the two differ, this file wins.

Scope: everything under `lie-detector/`. The specs themselves (layout, preprocessing contract,
evaluation protocol, plan) live in [`README.md`](README.md).

---

1. **Never commit data**: no `DOLOS/`, no `data/`, no `clips/`, no checkpoints. They are git-ignored.
   Each member needs a local copy of `DOLOS/` — obtained from ROSE Lab with their own account, or passed
   privately inside the group. It cannot travel through this repository: redistribution is not allowed,
   and git history is forever.

2. **One notebook per direction**, named `train_<direction>.ipynb`, inside its own folder, plus a short
   `README.md` next to it with the final table row and a paragraph of interpretation.

3. **Never edit shared files alone.** `preprocessing.ipynb` and this README change through a pull
   request or after telling the group — four people editing one notebook is how work gets lost.

4. **Reproducibility**: fixed seeds, pinned versions, and the split file hash written into each result.

5. **Report negative results.** "Clip encoding plateaus at 60%, like the paper" is a finding, not a
   failure.
