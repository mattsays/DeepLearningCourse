"""Direction `lore` — Paradigm 2: spatio-temporal Transformer (VideoMAE) + HuBERT, adapted with LoRA.

Modules:
    config  paths (local or Colab) and the run configuration
    faces   face-frame cache: 32 face crops per clip stored as one sprite JPEG
    data    splits/labels through the team's ``lie-detector/lib``, video and audio datasets
    models  VideoBranch / AudioBranch, LoRA wrapping, ``assert_lora_active``
    train   one-fold training loop and the resumable 5-fold driver
    fusion  late fusion of the two branches, cross-validation report, board row
"""

__version__ = "0.1.0"
