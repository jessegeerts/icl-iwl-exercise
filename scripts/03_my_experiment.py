"""Section 4: your experiment. Does the architecture matter?

Fill in the three lines below BEFORE you run anything (see EXERCISE.md), then
edit VARIANT (a config override) or icl_iwl/model.py (a code change) and run:

    python scripts/03_my_experiment.py
"""
from icl_iwl import compare

# --------------------------------------------------------------------------
# 1. What I changed:
# 2. My prediction (ICL / IWL, in which data regime):
# 3. Why:
# --------------------------------------------------------------------------

VARIANT = {"n_layers": 1}        # <- config overrides for your variant ({} if you edited model.py)
SEEDS = (0, 1, 2)                # at least 3: the ICL/IWL switch is abrupt and its timing varies
STEPS = 4000

# Two data regimes: one where the baseline does ICL, one where it does IWL.
REGIMES = {
    "icl_regime": dict(n_classes=4096, p_bursty=1.0),
    "iwl_regime": dict(n_classes=1024, p_bursty=0.0),
}

for name, data in REGIMES.items():
    print(f"\n=== {name}: {data} ===")
    compare({"baseline": {}, "my variant": VARIANT}, seeds=SEEDS, steps=STEPS,
            title=name, save=f"results/my_experiment_{name}.png", **data)

# --------------------------------------------------------------------------
# What happened:
# What I now think this component is for:
# --------------------------------------------------------------------------
