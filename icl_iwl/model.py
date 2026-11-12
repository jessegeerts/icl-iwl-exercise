"""A small causal transformer (or LSTM). This is the file to edit for the
architecture experiment: token format, attention rule, LayerNorm, embeddings…"""
import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        self.pattern = scores.masked_fill(~causal, -1e9).softmax(-1)   # kept for plotting
        return self.out((self.pattern @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.d_model), Attention(cfg)
        self.use_mlp = cfg.use_mlp
        if cfg.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        return x


class Model(nn.Module):
    """Input tokens: item_1, label_1, ..., item_N, label_N, query_item.
    The prediction is read out from the last (query) position."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        T, d = 2 * cfg.n_context + 1, cfg.d_model
        self.embed_item = nn.Linear(cfg.item_dim, d)
        self.embed_label = nn.Embedding(cfg.n_labels, d)
        if cfg.pos_enc == "learned":
            self.pos = nn.Parameter(0.02 * torch.randn(T, d))
        elif cfg.pos_enc == "sinusoidal":
            t = torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2) / d)
            self.register_buffer("pos", torch.cat([t.sin(), t.cos()], 1))
        else:
            self.register_buffer("pos", torch.zeros(T, d))
        if cfg.model == "lstm":
            self.core = nn.LSTM(d, d, num_layers=cfg.n_layers, batch_first=True)
        else:
            self.core = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.readout = nn.Linear(d, cfg.n_labels)

    def forward(self, ctx_items, ctx_labels, query):
        B, N, _ = ctx_items.shape
        pairs = torch.stack([self.embed_item(ctx_items), self.embed_label(ctx_labels)], 2).view(B, 2 * N, -1)
        x = torch.cat([pairs, self.embed_item(query)[:, None]], 1) + self.pos
        x = self.core(x)
        x = x[0] if isinstance(x, tuple) else x
        return self.readout(self.ln_out(x[:, -1]))
