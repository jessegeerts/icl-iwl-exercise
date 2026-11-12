"""Fast CPU smoke tests: the task produces well-formed batches and every
model variant trains for a few steps without error. Uses the Gaussian task
(no data file needed) plus a fake Omniglot embedding tensor."""
import os

import pytest
import torch

from icl_iwl import Config, Task, Model, train

TINY = dict(n_classes=64, steps=5, eval_every=5, batch_size=16, d_model=16)
FAKE_EMB = torch.randn(64 + 70, 5, 16).half()   # 134 classes x 5 drawings x 16 dims


def _check_batches(task, cfg):
    for kind in ["train", "icl", "iwl"]:
        ctx, lab, q, target = task.batch(kind, 32)
        assert ctx.shape == (32, cfg.n_context, task.item_dim)
        assert lab.shape == (32, cfg.n_context) and q.shape == (32, task.item_dim)
        assert target.shape == (32,)
    _, lab, _, target = task.batch("icl", 256)
    assert (lab == target[:, None]).any(1).all()          # ICL: answer always in context
    _, lab, _, target = task.batch("iwl", 256)
    assert not (lab == target[:, None]).any()             # IWL: answer never in context


def test_gaussian_batches():
    cfg = Config(dataset="gaussian", **TINY)
    _check_batches(Task(cfg, device="cpu"), cfg)


def test_omniglot_batches_fake_embeddings():
    cfg = Config(dataset="omniglot", **TINY)
    task = Task(cfg, device="cpu", embeddings=FAKE_EMB)
    assert task.item_dim == 16 and task.n_novel == 70
    _check_batches(task, cfg)
    with pytest.raises(ValueError):
        Task(Config(dataset="omniglot", n_classes=100), device="cpu", embeddings=FAKE_EMB)


@pytest.mark.parametrize("overrides", [
    {}, {"n_layers": 1}, {"use_mlp": False}, {"pos_enc": "none"},
    {"pos_enc": "sinusoidal"}, {"model": "lstm"}, {"n_heads": 4}, {"n_bursty": 1}, {"n_bursty": 4},
])
def test_train_variants(overrides):
    _, _, log = train(verbose=False, dataset="gaussian", **TINY, **overrides)
    assert len(log["step"]) == 2 and 0 <= log["icl"][-1] <= 1
    assert len(log["ila1"]) == 2 and (overrides.get("model") == "lstm" or 0 <= log["tila2"][-1] <= 1)


def test_forward_shape():
    cfg = Config(dataset="gaussian", **TINY)
    task, model = Task(cfg, device="cpu"), Model(cfg)
    *inputs, target = task.batch("train", 8)
    assert model(*inputs).shape == (8, cfg.n_labels)


@pytest.mark.skipif(not os.path.exists("data/omniglot.pt"), reason="run `python scripts/prepare_omniglot.py` first")
def test_omniglot_file():
    _, task, log = train(verbose=False, **TINY)
    assert task.item_dim == 64 and task.n_novel >= 64
    assert len(log["step"]) == 2


def test_model_cls_hook():
    class NoLayerNorm(Model):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.ln_out = torch.nn.Identity()
    model, _, log = train(verbose=False, dataset="gaussian", model_cls=NoLayerNorm, **TINY)
    assert isinstance(model, NoLayerNorm) and len(log["step"]) == 2
