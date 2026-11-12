"""Section 3: the data decides the strategy. Same model, different training
data. Before running: which corner of this grid do you expect to give ICL,
and which IWL?"""
from icl_iwl import train, plot_grid

Ks, ps = [1024, 4096], [0.0, 0.5, 1.0]
grid = {}
for K in Ks:
    for p in ps:
        log = grid[K, p] = train(verbose=False, n_classes=K, p_bursty=p, steps=4000)[2]
        print(f"n_classes={K:5d} p_bursty={p:.1f}   final ICL {log['icl'][-1]:.2f}   final IWL {log['iwl'][-1]:.2f}")
plot_grid(grid, Ks, ps, save="results/data_grid.png")
