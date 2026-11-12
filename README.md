# In-context vs in-weights learning in a small transformer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jessegeerts/icl-iwl-exercise/blob/main/notebooks/icl_vs_iwl.ipynb)

A transformer can solve a classification problem in two ways:

- **In-weights learning (IWL):** memorise which label goes with which class, in the weights.
- **In-context learning (ICL):** look up the answer in the examples given in the prompt.

[Chan et al. (2022)](https://arxiv.org/abs/2205.05055) showed that which strategy a transformer picks depends on the *statistics of the training data*. This repo is a small version of that experiment (the task follows [Reddy, 2023](https://arxiv.org/abs/2312.03002)) that trains in about a minute on a GPU.

**The task.** Items are handwritten characters from [Omniglot](https://github.com/brendenlake/omniglot) (1623 characters from 50 alphabets, 20 drawings each; each 90° rotation counts as its own class, giving 6492 classes). A small CNN turns every drawing into a 64-dimensional embedding, and that embedding is what the transformer sees. Every class has a fixed label (32 labels, so classes share them). The model sees 8 (item, label) pairs followed by a query item, and has to output the query's label.

**Two tests, run throughout training:**

- **ICL accuracy:** brand-new classes with randomly assigned labels. The query's class appears in the context. Only a model that reads the context can get this right.
- **IWL accuracy:** trained classes, but the query's class and label are *absent* from the context. Only a model that has memorised the class can get this right.

**Two data knobs:**

- `p_bursty`: the fraction of training sequences that are "bursty", meaning the query's class appears `n_bursty` (default 3) times in the context.
- `n_classes`: how many classes there are to memorise.

## Getting started on Colab

1. Click the **Open in Colab** badge above (or open `notebooks/icl_vs_iwl.ipynb` from GitHub via File → Open notebook → GitHub in Colab).
2. Set the runtime to a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run the first two cells. They clone this repo into the Colab machine and install it (about 30 seconds; the Omniglot embeddings are included in the repo, so there is nothing to download).
4. Work through the notebook. Sections 1 to 3 are reading and running; section 4 is your experiment.

To edit the model code on Colab, open the **Files** pane on the left, go to `icl-iwl-exercise/icl_iwl/` and double-click `model.py`. Save with Ctrl/Cmd-S, then run the reload cell in section 4 (`icl_iwl.reload()` followed by the import line) so the notebook sees your change. Alternatively, define your variant as a subclass of `Model` in a notebook cell and pass it as `model_cls=` (there is a template cell in section 4).

Colab forgets everything when the runtime disconnects, so keep your notes and your plot in the notebook (File → Save a copy in Drive) or download them.

## Working locally

```bash
git clone https://github.com/jessegeerts/icl-iwl-exercise.git && cd icl-iwl-exercise
pip install -e .          # torch, torchvision, matplotlib and the icl_iwl package
pytest                    # ~10 s on CPU: checks everything runs
```

A GPU makes a big difference (≈15 s per run on a laptop GPU, minutes on CPU). CUDA and Apple's MPS are picked up automatically.

## Layout

```
icl_iwl/            the library
  config.py         Config: every knob (data, model, training)
  data.py           Task: generates train / icl / iwl batches
  model.py          Attention, Block, Model  <- edit this for architecture experiments
  train.py          train(), accuracy()
  plotting.py       plot_runs(), compare(), plot_attention(), plot_grid(), plot_dynamics()
  analysis.py       induction-head progress measures (Reddy 2023), logged during training
  omniglot.py       the CNN embedder that made data/omniglot.pt
data/omniglot.pt    the item embeddings (6492 classes x 10 drawings x 64 dims)
notebooks/icl_vs_iwl.ipynb   the exercise, for Colab
scripts/            the same sections as plain Python scripts, for local use
tests/test_smoke.py          fast CPU checks: run after you change the package
EXERCISE.md         what to do and what to hand in
```

## Usage

Everything is a keyword override of `Config`:

```python
from icl_iwl import train, compare, plot_runs, plot_attention, plot_examples

model, task, log = train(n_classes=4096, p_bursty=1.0, steps=4000)
plot_examples(task)              # a few drawings of a few training classes and their labels
plot_runs({"baseline": [log]})
plot_attention(model, task)

# several variants x several seeds, in one data regime
runs = compare({"baseline": {}, "1 layer": {"n_layers": 1}},
               seeds=(0, 1, 2), n_classes=4096, p_bursty=1.0, steps=4000)
```

## How the embeddings were made

`scripts/prepare_omniglot.py` downloads Omniglot and trains a 4-block CNN (`icl_iwl/omniglot.py`) to classify all 6492 rotated characters from drawings 0-9 of each one. Its 64-d bottleneck then embeds drawings 10-19, which the embedder never saw, so the within-class variation in the task is real. You don't need to run it (the result is in `data/omniglot.pt`), but you can, for example to change the embedding size.

## The exercise

See [EXERCISE.md](EXERCISE.md).
