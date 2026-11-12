# The exercise

Work through the three parts in order. Parts 1 and 2 are mostly reading and
running; part 3 is yours.

## 0. Setup

Run `python scripts/prepare_omniglot.py` once. It downloads Omniglot and
trains a small CNN to embed every drawing (see `icl_iwl/omniglot.py`). The
transformer never sees pixels, only these 64-d embeddings. Look at
`plot_examples(task)` to see what a "class" and an "item" are.

## 1. Baseline (`scripts/01_baseline.py`)

A 2-layer transformer on fully bursty data with 4096 classes. Run it and look
at the two curves. Which strategy does the model end up with?

`train` also logs three progress measures from Reddy (2023) (`log["cla"]`,
`log["ila1"]`, `log["tila2"]`; see `icl_iwl/analysis.py`), and
`plot_dynamics(logs)` shows them next to the accuracy curves. Which one moves
first? Which ones move at the jump?

Then look at the attention plots. Positions are ordered item, label, item,
label, …, query (even positions are items, odd positions are labels, 16 is
the query). What is the query attending to in the last layer? What is the
first layer doing?

## 2. The data decides the strategy (`scripts/02_data_grid.py`)

Same model, different training data: `n_classes ∈ {1024, 4096}` and
`p_bursty ∈ {0, 0.5, 1}`.

**Before running:** which corner of this grid do you expect to give ICL, and
which IWL? Write it down. Then run it.

## 3. Your experiment: does the architecture matter? (`scripts/03_my_experiment.py`)

Pick **one** change to the model. Then, **before you run anything**, write
down (in the header of the script):

1. What you changed.
2. What you predict will happen to ICL and to IWL, and in which data regime.
3. *Why*: what does that component do that ICL or IWL needs?

Then run it against the baseline with `compare`, over at least 3 seeds, in at
least two data regimes (one where the baseline does ICL, one where it does
IWL). The template script already does this. A single run tells you very
little: the switch between strategies is abrupt and its timing varies from
seed to seed.

Some options that only need a config change (edit `VARIANT` in the script):

| Change | Config |
|---|---|
| One attention layer instead of two | `n_layers=1` |
| Remove the MLPs (attention-only) | `use_mlp=False` |
| No positional information | `pos_enc="none"` |
| Replace attention with recurrence | `model="lstm"` |
| More heads, wider or narrower model | `n_heads=4`, `d_model=16` |
| Regularisation | `weight_decay=0.1` |

Better still, change the model code yourself, either by editing
`icl_iwl/model.py` or by subclassing `Model` in a notebook cell and passing
`model_cls=MyModel` to `train` / `compare`: a different token format (item and label in one token), a different attention
rule, no LayerNorm, frozen embeddings, and so on. The *data* side is open too:
`n_bursty` sets how often the query's class repeats in a bursty context,
`noise` adds extra within-class variability, `zipf` makes some classes rare,
and `icl_iwl/omniglot.py` lets you change the embedder (its width, bottleneck
size, or how many drawings it trains on) and rebuild `data/omniglot.pt`. If you do, keep the baseline
reachable (e.g. behind a new `Config` flag) so that `compare` can still run
both. Run `pytest` after editing to check nothing is broken.

## What to hand in

One plot (from `results/`) and three sentences:

- **My prediction:**
- **What happened:**
- **What I think this component is for:**
