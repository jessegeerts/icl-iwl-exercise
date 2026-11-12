"""Omniglot items.

A small CNN is trained from scratch to classify Omniglot characters (1623
characters x 4 rotations = 6492 classes; each rotation counts as a separate
class, as in Chan et al. 2022) using drawings 0-9 of every character. Its
64-dimensional bottleneck is then used to embed drawings 10-19, which the
embedder never saw, so the within-class variation in the task is real.

Build the embedding file once (downloads ~40 MB of images, then ~1-2 min on a
GPU, ~10 min on CPU):

    python scripts/prepare_omniglot.py        # writes data/omniglot.pt

`Task` loads it from `Config.omniglot_path`.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PATH = "data/omniglot.pt"
RAW_ROOT = "data/omniglot_raw"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path):
    """Relative paths are tried from the current directory, then from the repo root."""
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(REPO_ROOT, path)
        if os.path.exists(alt):
            return alt
    return path
ROTATIONS = (0, 90, 180, 270)
N_EMBED_TRAIN = 10        # drawings 0-9 train the embedder, 10-19 are used by the task
IMG = 64                  # embedder input resolution


def load_omniglot(path=DEFAULT_PATH):
    """Returns the dict saved by `build`: emb [n_classes, n_drawings, dim] (float16),
    names, dirs, rotation, ..."""
    path = _resolve(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build the Omniglot embeddings first with\n"
            f"    python scripts/prepare_omniglot.py\n"
            f"or use the abstract task with dataset='gaussian'.")
    return torch.load(path)


def _load_raw(root):
    """All 1623 characters as a uint8 tensor [1623, 20, 105, 105] (0 = ink, 255 = paper)."""
    import numpy as np
    from torchvision.datasets import Omniglot

    imgs, names, dirs = [], [], []
    for background in (True, False):
        ds = Omniglot(root, background=background, download=True)
        per_char = {}
        for i, (file, c) in enumerate(ds._flat_character_images):
            per_char.setdefault(c, []).append(i)
        sub = "images_background" if background else "images_evaluation"
        for c in sorted(per_char):
            imgs.append(torch.stack([torch.from_numpy(np.array(ds[i][0], dtype=np.uint8)) for i in per_char[c]]))
            names.append(ds._characters[c])
            dirs.append(os.path.join("omniglot-py", sub, ds._characters[c]))
    n = min(len(x) for x in imgs)
    return torch.stack([x[:n] for x in imgs]), names, dirs


class Encoder(nn.Module):
    """4 conv blocks -> `dim`-d embedding -> linear classifier (training only)."""

    def __init__(self, n_classes, dim=64, width=64):
        super().__init__()
        blocks, c = [], 1
        for _ in range(4):
            blocks += [nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            c = width
        self.conv = nn.Sequential(*blocks)                       # 64x64 -> 4x4
        self.proj = nn.Linear(width * (IMG // 16) ** 2, dim)
        self.head = nn.Linear(dim, n_classes)

    def embed(self, x):                                           # x: [B, 1, IMG, IMG] in [0, 1], ink = 1
        return self.proj(self.conv(x).flatten(1))

    def forward(self, x):
        return self.head(self.embed(x))


def _resize(imgs):
    """uint8 [C, E, 105, 105] -> uint8 [C, E, IMG, IMG] (done once, on CPU)."""
    C, E, H, W = imgs.shape
    x = F.interpolate(imgs.view(-1, 1, H, W).float(), size=(IMG, IMG), mode="area")
    return x.round().to(torch.uint8).view(C, E, IMG, IMG)


def _prep(x, rot, shift=0):
    """uint8 [B, IMG, IMG] -> float [B, 1, IMG, IMG], ink = 1, rotated by `rot` quarter turns."""
    x = 1 - x.float() / 255
    x = torch.rot90(x, rot, dims=(-2, -1))
    if shift:
        dx, dy = torch.randint(-shift, shift + 1, (2,)).tolist()
        x = torch.roll(x, (dx, dy), dims=(-2, -1))
    return x[:, None]


@torch.no_grad()
def _embed_all(net, imgs, drawings, device, batch=512):
    """imgs [C, E, IMG, IMG] -> [4*C, len(drawings), dim] (class = rot_index * C + char)."""
    C = imgs.shape[0]
    flat = imgs[:, drawings].reshape(-1, *imgs.shape[-2:])
    out = []
    for r in range(len(ROTATIONS)):
        z = torch.cat([net.embed(_prep(flat[i:i + batch].to(device), r)).cpu() for i in range(0, len(flat), batch)])
        out.append(z.view(C, len(drawings), -1))
    return torch.cat(out)


def nearest_centroid_accuracy(emb, n_proto=5):
    """How separable are the classes? Centroids from the first `n_proto`
    drawings of every class, test on the rest (all classes at once)."""
    emb = emb.float()
    C, E, D = emb.shape
    mu, test = emb[:, :n_proto].mean(1), emb[:, n_proto:].reshape(-1, D)
    truth = torch.arange(C).repeat_interleave(E - n_proto)
    pred = torch.cat([torch.cdist(test[i:i + 4096], mu).argmin(1) for i in range(0, len(test), 4096)])
    return (pred == truth).float().mean().item()


def build(path=DEFAULT_PATH, root=RAW_ROOT, dim=64, steps=3000, width=64, shift=4, device=None, batch=256, seed=0):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    imgs, names, dirs = _load_raw(root)
    C, E, H, W = imgs.shape
    n_rot = len(ROTATIONS)
    print(f"loaded {C} characters x {E} drawings of {H}x{W} px  ({time.time() - t0:.0f}s); embedder on {device}")
    imgs = _resize(imgs)

    torch.manual_seed(seed)
    net = Encoder(n_rot * C, dim, width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    train_imgs = imgs[:, :N_EMBED_TRAIN].to(device)              # [C, 10, IMG, IMG]
    for step in range(1, steps + 1):
        chars = torch.randint(0, C, (batch,), device=device)
        draw = torch.randint(0, N_EMBED_TRAIN, (batch,), device=device)
        rot = int(torch.randint(0, n_rot, (1,)))                 # one rotation per batch (cheap)
        x = _prep(train_imgs[chars, draw], rot, shift=shift)
        loss = F.cross_entropy(net(x), rot * C + chars)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps:
            print(f"  embedder step {step:5d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    net.eval()

    emb = _embed_all(net, imgs, range(N_EMBED_TRAIN, E), device)          # unseen drawings 10-19
    emb = emb / emb.norm(dim=-1, keepdim=True)                             # items on the unit sphere
    acc_unseen = nearest_centroid_accuracy(emb)
    acc_seen = nearest_centroid_accuracy(_embed_all(net, imgs, range(0, N_EMBED_TRAIN), device))
    print(f"nearest-centroid accuracy over all {n_rot * C} classes: {acc_unseen:.2f} on unseen drawings "
          f"(task data), {acc_seen:.2f} on the embedder's training drawings")

    out = {
        "emb": emb.half(),
        "names": [f"{n} rot{deg}" for deg in ROTATIONS for n in names],
        "dirs": dirs * n_rot,
        "rotation": [deg for deg in ROTATIONS for _ in names],
        "drawings": list(range(N_EMBED_TRAIN, E)),
        "dim": dim, "nearest_centroid_accuracy": acc_unseen,
        "source": f"{IMG}px 4-block CNN (width {width}) trained on drawings 0-{N_EMBED_TRAIN - 1}, {steps} steps, shift {shift}, seed {seed}",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(out, path)
    print(f"saved {path}: {tuple(emb.shape)} float16  ({time.time() - t0:.0f}s)")
    return out


def _ensure_raw(root):
    """Download the Omniglot images into `root` if they are not there yet (~40 MB)."""
    if not os.path.isdir(os.path.join(root, "omniglot-py", "images_background")):
        from torchvision.datasets import Omniglot
        print(f"downloading the Omniglot images (~40 MB) into {root} ...", flush=True)
        for background in (True, False):
            Omniglot(root, background=background, download=True)
    return root


def plot_examples(task, n_classes=4, n_drawings=6, root=RAW_ROOT, save=None):
    """Show a few drawings of a few training classes. Downloads the raw
    Omniglot images into `root` on first use (the repo only ships embeddings)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    meta, root = task.meta, _ensure_raw(_resolve(root))
    if meta is None:
        raise ValueError("plot_examples needs the Omniglot task (dataset='omniglot')")
    fig, axes = plt.subplots(n_classes, n_drawings, figsize=(1.3 * n_drawings, 1.5 * n_classes))
    for i, c in enumerate(task.train_classes[:n_classes].tolist()):
        d = os.path.join(root, meta["dirs"][c])
        files = sorted(os.listdir(d))[meta["drawings"][0]:][:n_drawings]
        for j, f in enumerate(files):
            img = Image.open(os.path.join(d, f)).rotate(meta["rotation"][c])
            axes[i, j].imshow(img, cmap="gray"); axes[i, j].axis("off")
        axes[i, 0].set_title(f"class {c}: {meta['names'][c]}  ->  label {c % task.cfg.n_labels}", fontsize=8, loc="left")
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
