"""All the knobs. Pass overrides to `train(...)` as keyword arguments,
e.g. `train(n_classes=4096, p_bursty=0.5, n_layers=1)`."""
from dataclasses import dataclass

import torch

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()   # Apple Silicon
          else "cpu")


@dataclass
class Config:
    # ---- data ----
    dataset: str = "omniglot"   # "omniglot" (character embeddings) or "gaussian" (abstract clusters)
    omniglot_path: str = "data/omniglot.pt"   # built by `python -m icl_iwl.omniglot`
    n_classes: int = 1024     # K: number of training classes (Omniglot: at most 6492 - 64)
    n_labels: int = 32        # L: classes share labels (label = class % L)
    item_dim: int = 64        # D: dimensionality of an item (Omniglot sets this from the file)
    noise: float = 0.2        # extra isotropic noise on items (Gaussian task: the within-class spread)
    p_bursty: float = 1.0     # fraction of training sequences that are bursty
    n_bursty: int = 3         # B: copies of the query's class in a bursty context (<= n_context // 2)
    zipf: float = 0.0         # class frequency ~ 1/rank^zipf (0 = uniform)
    n_context: int = 8        # item-label pairs before the query
    # ---- model ----
    model: str = "transformer"  # "transformer" or "lstm"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 1
    use_mlp: bool = True
    pos_enc: str = "learned"    # "learned", "sinusoidal" or "none"
    # ---- training ----
    steps: int = 6000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_every: int = 250
    seed: int = 0
