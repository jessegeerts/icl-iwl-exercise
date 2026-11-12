"""Plots and the multi-seed `compare` helper."""
import os

import matplotlib
import matplotlib.pyplot as plt
import torch

from .train import train


def _finish(fig, save):
    """Save to `save` (if given) and show the figure in notebooks / GUIs."""
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved {save}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def plot_runs(runs, title="", save=None):
    """runs: {name: [log, log, ...]}, one log per seed."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for i, (name, logs) in enumerate(runs.items()):
        for ax, key in zip(axes, ["icl", "iwl"]):
            for log in logs:
                ax.plot(log["step"], log[key], color=f"C{i}", alpha=0.3, lw=1)
            mean = torch.tensor([log[key] for log in logs]).mean(0)
            ax.plot(logs[0]["step"], mean, color=f"C{i}", lw=2, label=name)
    for ax, name in zip(axes, ["ICL accuracy (novel classes)", "IWL accuracy (trained classes)"]):
        ax.set(title=name, xlabel="training step", ylim=(-0.03, 1.03))
    axes[0].set_ylabel("accuracy"); axes[1].legend(frameon=False); fig.suptitle(title)
    _finish(fig, save)


def compare(variants, seeds=(0, 1, 2), save=None, title="", **common):
    """variants: {name: {config overrides}}. Trains each variant on each seed
    and plots ICL/IWL curves (thin = seeds, thick = mean). An override dict
    may contain "model_cls" to train a custom model class."""
    runs = {}
    for name, overrides in variants.items():
        runs[name] = [train(verbose=False, seed=s, **{**common, **overrides})[2] for s in seeds]
        print(f"{name:30s} final ICL {sum(l['icl'][-1] for l in runs[name]) / len(seeds):.2f}   final IWL {sum(l['iwl'][-1] for l in runs[name]) / len(seeds):.2f}")
    plot_runs(runs, title=title, save=save)
    return runs


@torch.no_grad()
def plot_attention(model, task, save=None):
    """How much the query (last position) attends to items vs labels of each
    class in the last layer, then the full attention pattern of every layer,
    averaged over a batch of ICL test sequences. Positions are ordered
    item, label, item, label, ..., query (even = items, odd = labels)."""
    ctx, labels, query, target = task.batch("icl", 500)
    model.eval(); model(ctx, labels, query)
    blocks = list(model.core)
    correct = (labels == target[:, None])                      # which context pairs carry the right answer
    last = blocks[-1].attn.pattern.mean(1)[:, -1, :-1].view(len(target), -1, 2)   # query -> (item, label) of each pair
    print("Last layer, attention from the query to ...")
    for name, mask in [("pairs of the query's class", correct), ("pairs of the other class ", ~correct)]:
        print(f"  {name}:  items {(last[..., 0] * mask).sum(1).mean():.2f}   labels {(last[..., 1] * mask).sum(1).mean():.2f}")
    fig, axes = plt.subplots(1, len(blocks), figsize=(4.5 * len(blocks), 4))
    for i, (ax, b) in enumerate(zip([axes] if len(blocks) == 1 else axes, blocks)):
        ax.imshow(b.attn.pattern.mean((0, 1)).cpu(), cmap="viridis")
        ax.set(title=f"layer {i + 1} attention (batch mean)", xlabel="attended position", ylabel="from position")
    _finish(fig, save)


def plot_grid(grid, Ks, ps, save=None):
    """grid: {(n_classes, p_bursty): log}. Heatmaps of final ICL / IWL accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, key in zip(axes, ["icl", "iwl"]):
        vals = [[grid[K, p][key][-1] for p in ps] for K in Ks]
        ax.imshow(vals, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(Ks)):
            for j in range(len(ps)):
                ax.text(j, i, f"{vals[i][j]:.2f}", ha="center", va="center", color="w" if vals[i][j] < 0.6 else "k")
        ax.set(title=f"final {key.upper()} accuracy", xlabel="p_bursty", ylabel="n_classes",
               xticks=range(len(ps)), xticklabels=ps, yticks=range(len(Ks)), yticklabels=Ks)
    _finish(fig, save)


def plot_sweeps(sweeps, save=None, title=""):
    """sweeps: {axis_label: {value: [log per seed]}}. One column per sweep,
    final ICL accuracy on top and final IWL accuracy below (like Reddy 2023, Fig. 2)."""
    fig, axes = plt.subplots(2, len(sweeps), figsize=(3.2 * len(sweeps), 5), sharey=True, squeeze=False)
    for j, (label, points) in enumerate(sweeps.items()):
        xs = list(points)
        for i, key in enumerate(["icl", "iwl"]):
            ax = axes[i, j]
            finals = [[log[key][-1] for log in points[x]] for x in xs]
            mean = [sum(f) / len(f) for f in finals]
            ax.plot(xs, mean, "o-", color=f"C{i}")
            ax.fill_between(xs, [min(f) for f in finals], [max(f) for f in finals], color=f"C{i}", alpha=0.2)
            ax.set(ylim=(-0.03, 1.03), xlabel=label)
            if label.startswith("n_classes"):
                ax.set_xscale("log", base=2)
            if j == 0:
                ax.set_ylabel(f"final {key.upper()} accuracy")
    fig.suptitle(title)
    _finish(fig, save)


def plot_dynamics(logs, save=None, title="", align=False, threshold=0.75):
    """Accuracy and induction-head progress measures over training for one
    configuration, one line per seed. `align=True` shifts every seed so that
    step 0 is where ICL accuracy first exceeds `threshold` (Reddy 2023, Fig. 4;
    the pre-transition plateau sits at 0.5, so the default is 0.75)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for s, log in enumerate(logs):
        steps = torch.tensor(log["step"], dtype=torch.float)
        if align:
            above = [i for i, a in enumerate(log["icl"]) if a > threshold]
            steps = steps - (steps[above[0]] if above else steps[-1])
        for key, c, ls in [("icl", "C0", "-"), ("iwl", "C1", "-")]:
            axes[0].plot(steps, log[key], color=c, ls=ls, alpha=0.8, label=key.upper() if s == 0 else None)
        for key, c in [("cla", "C2"), ("ila1", "C3"), ("tila2", "C4")]:
            axes[1].plot(steps, log[key], color=c, alpha=0.8, label=key.upper() if s == 0 else None)
    xlabel = "training step" + (f" (0 = ICL accuracy crosses {threshold})" if align else "")
    axes[0].set(title="accuracy", xlabel=xlabel, ylim=(-0.03, 1.03)); axes[0].legend(frameon=False)
    axes[1].set(title="progress measures (ICL test sequences)", xlabel=xlabel, ylim=(-0.03, 1.03)); axes[1].legend(frameon=False)
    fig.suptitle(title)
    _finish(fig, save)
