"""Progress measures for the induction-head circuit (after Reddy 2023, Fig. 3-4),
computed on a batch of ICL test sequences:

- ila1:  layer-1 attention from each label token to the item just before it
         ("previous-token" head). Positions: item, label, item, label, ..., query.
- tila2: last-layer attention from the query to the context labels that carry
         the correct answer (the induction head's output).
- cla:   probability the model assigns to *any* label present in the context
         (Reddy's CLA): rises during the slow phase, before ILA1 / TILA2 do.
"""
import math

import torch


@torch.no_grad()
def induction_measures(model, task, n=500):
    nan = {"ila1": math.nan, "tila2": math.nan, "cla": math.nan}
    if not isinstance(model.core, torch.nn.Sequential):      # LSTM: no attention patterns
        return nan
    was_training = model.training
    model.eval()
    ctx, labels, query, target = task.batch("icl", n)
    logits = model(ctx, labels, query)
    model.train(was_training)
    blocks = list(model.core)
    N = labels.shape[1]
    first = blocks[0].attn.pattern.mean(1)                    # [n, T, T], heads averaged
    label_pos = torch.arange(1, 2 * N, 2, device=first.device)
    ila1 = first[:, label_pos, label_pos - 1].mean().item()
    last = blocks[-1].attn.pattern.mean(1)[:, -1, :-1].view(n, N, 2)   # query -> (item, label) of each pair
    correct = labels == target[:, None]
    tila2 = (last[..., 1] * correct).sum(1).mean().item()
    probs = logits.softmax(-1)
    in_ctx = torch.zeros_like(probs).scatter_(1, labels, 1.0)
    cla = (probs * in_ctx).sum(1).mean().item()
    return {"ila1": ila1, "tila2": tila2, "cla": cla}
