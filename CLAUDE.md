# icl-iwl-coding-exercise

Teaching repo: in-context vs in-weights learning in a small transformer
(Chan et al. 2022 / Reddy 2023). Students modify the model and compare against
the baseline. See README.md and EXERCISE.md.

- `icl_iwl/config.py` holds every knob as a dataclass; `train(**overrides)`
  accepts any field as a keyword argument.
- `icl_iwl/model.py` is where architecture changes go. When adding a variant,
  put it behind a new `Config` flag (default = current behaviour) so the
  baseline still runs and `compare` can train both.
- `icl_iwl/data.py` generates batches; "icl" batches must always contain the
  answer, "iwl" batches never. `tests/test_smoke.py` checks this.
- Items are Omniglot embeddings from `data/omniglot.pt` (gitignored; built by
  `python scripts/prepare_omniglot.py`, which trains the CNN in
  `icl_iwl/omniglot.py`). `dataset="gaussian"` needs no file; tests use it.
- Run `pytest` (CPU, ~10 s) after any change. Full experiments need a GPU.
- `Attention.forward` stores `self.pattern` for `plot_attention` and for the
  progress measures in `icl_iwl/analysis.py`; keep that if you rewrite attention.
