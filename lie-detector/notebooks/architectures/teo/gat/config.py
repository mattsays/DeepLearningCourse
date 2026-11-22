from dataclasses import dataclass

@dataclass
class Config:
    # Feature layout
    n_landmark_regions: int = 15
    landmark_dim: int = 3
    n_audio_feats: int = 8

    # Sliding window
    window_sec: float = 1.5
    fps: int = 10
    overlap: float = 0.5


    # Model
    hidden: int = 32
    heads: int = 4
    num_mumin: int = 41
    dropout: float = 0.3

    # Training
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mumin_loss_weight: float = 0.3
    patience: int = 20

    @property
    def feat_per_frame(self) -> int:
        return self.n_landmark_regions * self.landmark_dim + self.n_audio_feats

    @property
    def node_feat_dim(self) -> int:
        # Each node is projected from a 3-dim vector (landmark xyz or padded audio)
        return max(self.landmark_dim, 1)