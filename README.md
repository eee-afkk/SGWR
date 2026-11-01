# SGWR

Core implementation of **SGWR**, the structured Gromov–Wasserstein readout of the accompanying manuscript (under review).
SGWR improves the scalability of the semi-relaxed fused Gromov–Wasserstein (srFGW) transport on scene graphs and uses it to read
out the labels of a cross-scene classifier. The scene graph of the target image is transported onto a class graph whose class
masses are free, so no class proportion of the target has to be given or estimated. Pixels that need no label enter the graph as
zero-mass relay nodes. A matrix-free solver handles scenes with several hundred thousand pixels.

Author of the code: Zheng-Yang Ren.

## What is here

`sgwr.py` holds the readout. It needs PyTorch only, runs on CPU or GPU, involves no training and uses no target label.

| Function | Step of the method |
|---|---|
| `standardize`, `knn_indices`, `scene_graph` | Scene graph: k-NN graph (cosine similarity) of the feature vectors of all pixels of the target image, symmetrized and normalized |
| `StructureOperator` | Structure operator H = P h_alpha(S) P^T, applied implicitly by L sparse diffusion steps. The pixels that are not evaluated are zero-mass relay nodes, and the N x N matrix H is never formed |
| `class_graph` | Closed-form class graph B*(Q) = S(Q) / (m m^T) with the floor eps_B (estimated class graph E, learned class graph G) |
| `source_class_graph`, `scale_to_target`, `pool` | Source prior: class graph of the labeled source samples, scaled to the total connection of the target and pooled geometrically with B*, log B = (1 - w) log B* + w log B^s |
| `smooth_objective`, `objective`, `gradient` | Objective Phi(Q, B) of the readout and its gradient; identity structure X with the mass weight mu |
| `stationarity_gap` | Stationarity gap mean_i KL(Q_i \|\| softmax(-G_i)) |
| `solve_mirror_descent` | Mirror-descent sweeps with the backtracking test (the solver of the reported results) |
| `solve_conditional_gradient` | Conditional-gradient sweeps with backtracking on the step (one product with H per sweep) |
| `sgwr` | The complete readout |

## Use

```python
import torch
from sgwr import sgwr, standardize

# posterior: N x K class posteriors of a trained cross-scene classifier on the N evaluated target pixels
# features:  M x d feature vectors of ALL M pixels of the target image, the N evaluated pixels first (same order as posterior);
#            in the paper g_v = [standardize(EMAP), standardize(spectrum)]
out = sgwr(posterior, features, k=8, alpha=0.99, steps=50, variant='E', lam=32.0, temperature=1.0, sweeps=30)
labels = out['labels']   # label of every evaluated pixel
masses = out['masses']   # class masses chosen by the readout (free, not prescribed)
B = out['B']             # class graph
gap = out['gap']         # stationarity gap after the sweeps (nats per pixel)

# with the source prior (variants E and G)
# source_features: N_s x d network features of the labeled source samples (the layer before the classifier)
# source_labels:   their N_s class indices
out = sgwr(posterior, features, k=8, alpha=0.99, steps=50, variant='E', lam=32.0, temperature=1.0, sweeps=30,
           source_features=source_features, source_labels=source_labels, w=0.75)
```

Frozen configurations of the paper (30 mirror-descent sweeps, L = 50 diffusion steps, eps_B = 1e-3):

| Task | Class graph (`variant`) | `k` | `alpha` | `lam` | `mu` | `temperature` | `w` |
|---|---|---|---|---|---|---|---|
| Houston 2013 → 2018 | estimated, `'E'` | 8 | 0.99 | 32 | – | 1 | 0.75 |
| Pavia University → Pavia Center | learned, `'G'` | 16 | 0.99 | 16 | – | 1 | 0.75 |
| Pavia Center → Pavia University | identity, `'X'` | 8 | 0.99 | 128 | 16 | 1 | – |
| Shanghai → Hangzhou | learned, `'G'` | 16 | 0.99 | 2 | – | 1 | 0 |
| Hangzhou → Shanghai | no readout (the network output is kept) | | | | | | |
| Zhangye F1 → F2 | learned, `'G'` | 16 | 0 | 32 | – | 2 | 0 |
| Shared by the meter-resolution tasks | learned, `'G'` | 8 | 0.99 | 16 | – | 1 | 0 |

## Status

This repository currently holds the core readout. The complete code will be released here when the paper is published. It will
include the descriptor computation, the class-balanced base network and its training, the readout-guided retraining, the
evaluation scripts, the per-run results and the scripts that generate the tables of the paper.
