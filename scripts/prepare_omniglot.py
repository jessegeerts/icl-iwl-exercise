"""Download Omniglot and build the item embeddings used by the task
(data/omniglot.pt): trains a small CNN embedder, ~1-2 min on a GPU.

    python scripts/prepare_omniglot.py [--dim 64] [--steps 3000] [--path data/omniglot.pt]
"""
import argparse

from icl_iwl.omniglot import DEFAULT_PATH, RAW_ROOT, build

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--path", default=DEFAULT_PATH)
ap.add_argument("--root", default=RAW_ROOT)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--width", type=int, default=64)
a = ap.parse_args()
build(a.path, a.root, a.dim, a.steps, a.width)
