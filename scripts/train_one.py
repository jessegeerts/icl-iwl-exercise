"""Train a single model from the command line with any Config overrides.

    python scripts/train_one.py n_classes=4096 p_bursty=0.5 n_layers=1 steps=2000
"""
import sys
from dataclasses import fields

from icl_iwl import Config, train, plot_runs

types = {f.name: f.type for f in fields(Config)}
overrides = {}
for arg in sys.argv[1:]:
    k, v = arg.split("=", 1)
    t = types[k]
    overrides[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)

_, _, log = train(**overrides)
name = " ".join(sys.argv[1:]) or "default"
plot_runs({name: [log]}, save="results/train_one.png")
