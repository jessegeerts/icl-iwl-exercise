"""Section 2: baseline. A 2-layer transformer on fully bursty data with 4096
classes, then a look at what the query attends to."""
from icl_iwl import train, plot_runs, plot_attention

model, task, log = train(n_classes=4096, p_bursty=1.0, steps=4000)
plot_runs({"baseline": [log]}, save="results/baseline.png")
plot_attention(model, task, save="results/baseline_attention.png")
