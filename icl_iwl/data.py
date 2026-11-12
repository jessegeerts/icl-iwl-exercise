"""The task: `n_context` (item, label) pairs, then a query item.

Items are either Omniglot character embeddings (dataset="omniglot", the
default; see omniglot.py) or abstract Gaussian clusters (dataset="gaussian").
Either way a *class* has a fixed label (class % n_labels) and several
*items* (drawings, or noisy samples) that belong to it."""
import math

import torch

from .config import DEVICE

MIN_NOVEL = 64   # classes held out of training for the ICL test (Omniglot)


class Task:
    """`batch(kind)` returns `(ctx_items, ctx_labels, query_item, target)`:

    - kind="train": training distribution. A fraction `p_bursty` of sequences
      are *bursty* (query class appears `n_bursty` times in context, another
      class as often, plus random pairs); the rest have i.i.d. context classes.
    - kind="icl":   novel classes with randomly reassigned labels; the answer
      IS in the context. Only in-context learning can solve this.
    - kind="iwl":   trained classes, but no context pair shares the query's
      label. Only in-weights (memorised) knowledge can solve this.
    """

    def __init__(self, cfg, device=DEVICE, embeddings=None):
        self.cfg, self.device = cfg, device
        g = torch.Generator().manual_seed(cfg.seed)
        K = cfg.n_classes
        if cfg.dataset == "gaussian":
            D = cfg.item_dim
            self.train = (torch.randn(K, D, generator=g) / math.sqrt(D)).to(device)   # class means
            self.novel = (torch.randn(K, D, generator=g) / math.sqrt(D)).to(device)
            self.meta, self.train_classes = None, torch.arange(K)
        else:
            if embeddings is None:
                from .omniglot import load_omniglot
                embeddings = load_omniglot(cfg.omniglot_path)
            emb = embeddings["emb"] if isinstance(embeddings, dict) else embeddings   # [C, E, D]
            self.meta = embeddings if isinstance(embeddings, dict) else None
            C = emb.shape[0]
            if K + MIN_NOVEL > C:
                raise ValueError(f"n_classes={K} too large: {C} Omniglot classes, {MIN_NOVEL} must stay novel")
            perm = torch.randperm(C, generator=g)
            self.train_classes = perm[:K]
            self.train = emb[perm[:K]].float().to(device)        # [K, E, D] drawings of each training class
            self.novel = emb[perm[K:]].float().to(device)        # held-out classes for the ICL test
        self.item_dim = self.train.shape[-1]
        self.n_novel = self.novel.shape[0]
        ranks = torch.arange(1, K + 1, dtype=torch.float)
        self.class_p = (ranks ** -cfg.zipf).to(device)

    def _classes(self, *shape):
        n = math.prod(shape)
        return torch.multinomial(self.class_p, n, replacement=True).view(*shape)

    def _items(self, pool, classes):
        """One item per entry of `classes`: a random drawing (Omniglot) or the
        class mean (Gaussian), plus isotropic noise of size `cfg.noise`."""
        if pool.dim() == 3:   # [K, n_drawings, D]
            which = torch.randint(0, pool.shape[1], classes.shape, device=self.device)
            x = pool[classes, which]
        else:                 # [K, D]
            x = pool[classes]
        x = x + self.cfg.noise * torch.randn(*classes.shape, self.item_dim, device=self.device) / math.sqrt(self.item_dim)
        return x / math.sqrt(1 + self.cfg.noise ** 2)

    def _shuffle(self, ctx):
        return torch.gather(ctx, 1, torch.rand(ctx.shape, device=self.device).argsort(1))

    def batch(self, kind, B=None):
        """kind: "train", "icl" (novel classes, answer is in context) or
        "iwl" (trained classes, answer is NOT in context)."""
        c, L, N = self.cfg, self.cfg.n_labels, self.cfg.n_context
        B = B or c.batch_size

        if kind == "icl":   # 2 novel classes x 4 examples, labels reassigned at random
            n = self.n_novel
            q = torch.randint(0, n, (B,), device=self.device)
            other = (q + 1 + torch.randint(0, n - 1, (B,), device=self.device)) % n
            ctx = self._shuffle(torch.stack([q] * (N // 2) + [other] * (N - N // 2), 1))
            lab_q = torch.randint(0, L, (B,), device=self.device)
            lab_o = (lab_q + torch.randint(1, L, (B,), device=self.device)) % L
            ctx_lab = torch.where(ctx == q[:, None], lab_q[:, None], lab_o[:, None])
            pool, target = self.novel, lab_q
        else:
            K = c.n_classes
            q = self._classes(B)
            other = (q + 1 + torch.randint(0, K - 1, (B,), device=self.device)) % K
            other = torch.where(other % L == q % L, (other + 1) % K, other)   # different label from q
            iid = self._classes(B, N)
            if kind == "iwl":   # no class in context shares the query's label
                ctx = torch.where(iid % L == q[:, None] % L, (iid + 1) % K, iid)
            else:               # bursty: B x query class, B x one other class, rest random
                Bn = c.n_bursty
                bursty = self._shuffle(torch.cat([torch.stack([q] * Bn + [other] * Bn, 1), iid[:, : N - 2 * Bn]], 1))
                is_bursty = torch.rand(B, device=self.device) < c.p_bursty
                ctx = torch.where(is_bursty[:, None], bursty, iid)
            ctx_lab = ctx % L
            pool, target = self.train, q % L
        return self._items(pool, ctx), ctx_lab, self._items(pool, q), target
