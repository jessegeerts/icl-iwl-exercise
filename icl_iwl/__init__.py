"""ICL vs IWL in a small transformer (after Chan et al. 2022, Reddy 2023).

Quick start::

    from icl_iwl import train, compare, plot_runs
    model, task, log = train(n_classes=4096, p_bursty=1.0, steps=4000)
    plot_runs({"baseline": [log]})
"""
from .config import DEVICE, Config
from .data import Task
from .model import Attention, Block, Model
from .train import accuracy, train
from .plotting import compare, plot_attention, plot_dynamics, plot_grid, plot_runs, plot_sweeps
from .omniglot import load_omniglot, plot_examples
from .analysis import induction_measures

__all__ = [
    "Config", "Task", "Attention", "Block", "Model",
    "DEVICE", "accuracy", "train",
    "compare", "plot_attention", "plot_dynamics", "plot_grid", "plot_runs", "plot_sweeps",
    "load_omniglot", "plot_examples", "induction_measures",
]


def reload():
    """Re-import the package after editing its files (for notebooks where
    `%autoreload` is unavailable). Re-run your `from icl_iwl import ...` afterwards."""
    import importlib
    import sys
    for name in ["config", "data", "model", "analysis", "train", "plotting", "omniglot"]:
        importlib.reload(sys.modules[f"{__name__}.{name}"])
    importlib.reload(sys.modules[__name__])
