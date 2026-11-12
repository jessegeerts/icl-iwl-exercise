"""Training loop and evaluation."""
import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from .config import DEVICE, Config
from .data import Task
from .model import Model
from .analysis import induction_measures


@torch.no_grad()
def accuracy(model, task, kind, n=1000):
    *inputs, target = task.batch(kind, n)
    return (model(*inputs).argmax(-1) == target).float().mean().item()


def train(cfg=None, verbose=True, model_cls=Model, **overrides):
    """Train one model. Returns `(model, task, log)` where `log` has keys
    "step", "loss", "icl", "iwl" plus the induction-head progress measures
    "ila1", "tila2", "cla" (see analysis.py), one entry per evaluation.

    Any `Config` field can be passed as a keyword override:
    `train(n_classes=4096, p_bursty=0.5, n_layers=1)`. `model_cls` lets you
    train your own model class (e.g. a subclass of `Model` defined in a
    notebook cell) instead of editing model.py: `train(model_cls=MyModel)`.
    """
    cfg = replace(cfg or Config(), **overrides)
    torch.manual_seed(cfg.seed)
    task = Task(cfg)
    cfg = replace(cfg, item_dim=task.item_dim)     # Omniglot fixes the item dimension
    model = model_cls(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log = {"step": [], "loss": [], "icl": [], "iwl": [], "ila1": [], "tila2": [], "cla": []}
    t0 = time.time()
    for step in range(cfg.steps + 1):
        *inputs, target = task.batch("train")
        loss = F.cross_entropy(model(*inputs), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg.eval_every == 0:
            model.eval()
            for k, v in [("step", step), ("loss", loss.item()), ("icl", accuracy(model, task, "icl")), ("iwl", accuracy(model, task, "iwl"))]:
                log[k].append(v)
            for k, v in induction_measures(model, task).items():     # Reddy's progress measures (NaN for non-transformers)
                log[k].append(v)
            model.train()
            if verbose:
                print(f"step {step:6d}  loss {loss.item():.3f}  ICL {log['icl'][-1]:.2f}  IWL {log['iwl'][-1]:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, task, log
