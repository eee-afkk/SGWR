"""Structured Gromov-Wasserstein readout (SGWR): reference implementation.

A cross-scene classifier gives a posterior p_i over K classes for each of the N evaluated pixels of the target image.  SGWR
replaces the pixel-wise arg max by a semi-relaxed fused Gromov-Wasserstein transport of the scene graph onto a class graph whose
class masses are free:

  scene graph   k-NN graph (cosine similarity of the feature vectors g_v) of ALL M pixels of the target image.  The N evaluated
                pixels come first and carry the mass 1/N; the other M - N pixels are zero-mass relay nodes that only relay the
                diffusion.  W = (A + A^T) / 2 for the 0/1 k-NN matrix A, normalized affinity S = D^-1/2 W D^-1/2.
  operator      H = P h_alpha(S) P^T,  h_alpha(S) = (1 - alpha) sum_{l<L} (alpha S)^l + (alpha S)^L   (alpha = 0: S itself),
                applied implicitly: HQ costs L sparse products on the M nodes; H (N x N) is never formed.
  objective     Phi(Q, B) = mean_i KL(Q_i || p~_i) + lam/2 (m'Bm - <S(Q), log B>),   m = Q'1/N,  S(Q) = Q'HQ/N,
                p~_i = softmax(log p_i / T)   (T: temperature).
  class graph   closed form B*(Q) = S(Q) / (m m'), floored at eps_B.  Variants: 'E' estimates B once from p~ and keeps it;
                'G' re-estimates it after every sweep; 'X' uses the identity structure,
                Phi_X(Q) = mean_i KL(Q_i || p~_i) - lam/2 tr S(Q) + mu/2 |m|^2   (mu = 0: iterated smoothing).
  source prior  optional, for 'E' and 'G': log B = (1 - w) log B* + w log B^s, floored at eps_B; once at the input posterior in 'E',
                after every sweep in 'G' (w = 0: the two variants).  B^s = c S_s / (m_s m_s'), S_s = Y_s' H_s Y_s / N_s, is the class
                graph of the labeled source samples (H_s: diffusion operator of the k-NN graph of their network features, with the k
                and alpha of the scene graph); c makes m'B^s m equal to the total connection sum_kl S(p~)_kl of the target.
  solver        mirror-descent sweeps  Q+ = softmax((log Q - eta G) / (1 + eta)),  G = N grad of the smooth part of Phi,
                with a backtracking test on the step eta <= 1 (the solver of the reported results), or conditional-gradient sweeps
                Q+ = Q + gamma (Q^ - Q),  Q^ = softmax(-G),  gamma in {1, 1/2, ...} with  Phi(Q+) <= Phi(Q) - gamma/2 gap(Q).
  output        labels argmax_k Q_ik, class masses m, class graph B and the stationarity gap mean_i KL(Q_i || Q^_i).

PyTorch only.  No target label is used anywhere.
"""
import torch
import torch.nn.functional as F

__all__ = ['standardize', 'knn_indices', 'scene_graph', 'StructureOperator', 'class_graph', 'source_class_graph', 'scale_to_target',
           'pool', 'smooth_objective', 'gradient', 'objective', 'stationarity_gap', 'solve_mirror_descent',
           'solve_conditional_gradient', 'sgwr']

EPS_B = 1e-3


@torch.no_grad()
def standardize(x):
    """column-wise standardization (zero mean, unit variance, population std) of a pixels x features matrix, computed in float64
    and returned in float32.  In the experiments the two parts of g_v (EMAP and spectrum) are standardized separately and
    concatenated."""
    x = torch.as_tensor(x).double()
    return ((x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-6)).float()


@torch.no_grad()
def knn_indices(feat, k=8, kmax=32, chunk=2.5e7):
    """indices (M x k) of the k nearest neighbors of every row of `feat` by cosine similarity, the row itself excluded.
    The neighbors are ranked once with topk(kmax) and the first k are kept (as in the experiments, where k in {8, 16})."""
    z = F.normalize(feat.float(), dim=1)
    n = z.shape[0]
    step = max(64, int(chunk // n))
    out = torch.empty(n, max(k, kmax), dtype=torch.int64, device=z.device)
    for a in range(0, n, step):
        sim = z[a:a + step] @ z.t()
        sim[torch.arange(sim.shape[0], device=z.device), torch.arange(a, a + sim.shape[0], device=z.device)] = -3.0
        out[a:a + step] = sim.topk(max(k, kmax), dim=1).indices
    return out[:, :k].contiguous()


@torch.no_grad()
def scene_graph(idx):
    """symmetrized 0/1 k-NN affinity W = (A + A^T) / 2 (edge weights 0.5 or 1) as a sparse M x M tensor, and D^-1/2"""
    n, k = idx.shape
    rows = torch.arange(n, device=idx.device).repeat_interleave(k)
    W = torch.sparse_coo_tensor(torch.stack([rows, idx.reshape(-1)]), torch.ones(n * k, device=idx.device), (n, n))
    W = ((W + W.t()) * 0.5).coalesce()
    return W, torch.sparse.sum(W, 1).to_dense().clamp_min(1e-12) ** -0.5


class StructureOperator:
    """the structure operator H = P h_alpha(S) P^T on a graph whose first N nodes are the evaluated pixels.

    H(Q) pads Q (N x K) with zero rows for the M - N relay nodes and runs the recursion F <- alpha S F + (1 - alpha) Q for L steps
    from F = Q, which applies the truncated diffusion kernel h_alpha(S); alpha = 0 applies S itself.  H is symmetric, entry-wise
    non-negative and has spectral norm at most 1."""

    def __init__(self, W, dinv, n_evaluated, alpha=0.99, steps=50):
        self.W, self.d, self.N, self.M = W, dinv, int(n_evaluated), len(dinv)
        self.alpha, self.steps = float(alpha), int(steps)

    def S(self, X):
        return self.d[:, None] * torch.sparse.mm(self.W, self.d[:, None] * X)

    def pad(self, Q):
        if self.M == self.N:
            return Q
        Z = torch.zeros(self.M, Q.shape[1], device=Q.device, dtype=Q.dtype)
        Z[:self.N] = Q
        return Z

    @torch.no_grad()
    def __call__(self, Q):
        Qa = self.pad(Q)
        if self.alpha == 0.0:
            return self.S(Qa)[:self.N]
        Fd = Qa.clone()
        for _ in range(self.steps):
            Fd = self.alpha * self.S(Fd) + (1 - self.alpha) * Qa
        return Fd[:self.N]


def class_graph(Q, n, eps_B=EPS_B):
    """closed-form class graph B*(Q) = S(Q) / (m m'), S(Q) = Q'HQ/N symmetrized, floored at eps_B (n = HQ)"""
    N = len(Q)
    m = Q.double().mean(0).clamp_min(1e-12)
    S = (Q.double().t() @ n.double()) / N
    S = 0.5 * (S + S.t())
    return (S / torch.outer(m, m)).clamp_min(eps_B)


@torch.no_grad()
def source_class_graph(features, labels, n_classes, k=8, alpha=0.99, steps=50, eps_B=EPS_B):
    """class graph of the labeled source samples, before the scaling c: S_s / (m_s m_s'), S_s = Y_s' H_s Y_s / N_s, floored at eps_B.

    features: N_s x d network features of the source samples (the layer before the classifier);  labels: their N_s class indices.
    k, alpha and steps are those of the scene graph."""
    g = torch.as_tensor(features).float()
    Y = torch.eye(n_classes, device=g.device)[torch.as_tensor(labels, device=g.device).long()]
    W, dinv = scene_graph(knn_indices(g, k))
    return class_graph(Y, StructureOperator(W, dinv, len(g), alpha, steps)(Y), eps_B)


def scale_to_target(Bs, Q, n, eps_B=EPS_B):
    """the source class graph with the scaling c: m'(c Bs)m equals the total connection sum_kl S(Q)_kl of the target at the input
    posterior Q (n = HQ), floored at eps_B"""
    Qd = Q.double()
    m = Qd.mean(0)
    total = float((Qd.t() @ n.double()).sum() / len(Q))
    return (Bs * (total / float(m @ Bs @ m))).clamp_min(eps_B)


def pool(B, Bs, w, eps_B=EPS_B):
    """the source prior: log B = (1 - w) log B + w log Bs, floored at eps_B (Bs = None or w = 0: B itself)"""
    if Bs is None or w == 0.0:
        return B
    return torch.exp((1 - w) * torch.log(B) + w * torch.log(Bs)).clamp_min(eps_B)


def smooth_objective(Q, n, B, logp, variant, lam, mu=0.0):
    """Phi without the entropy term: -mean_i <Q_i, log p~_i> + structure and mass terms (n = HQ; B = None for 'X')"""
    N = len(Q)
    Qd = Q.double()
    m = Qd.mean(0)
    feat = -(Qd * logp.double()).sum() / N
    if variant == 'X':
        return feat - 0.5 * lam * (Qd * n.double()).sum() / N + 0.5 * mu * (m * m).sum()
    S = (Qd.t() @ n.double()) / N
    return feat + 0.5 * lam * (m @ B @ m - (S * torch.log(B)).sum())


def gradient(Q, n, B, logp, variant, lam, mu=0.0):
    """N times the gradient of the smooth part of Phi with respect to Q:
    E, G:  G_ik = -log p~_ik + lam ((Bm)_k - sum_l n_il log B_kl);   X:  G_ik = -log p~_ik - lam n_ik + mu m_k"""
    m = Q.double().mean(0)
    if variant == 'X':
        return -logp - lam * n + (mu * m).float()[None, :]
    return -logp + lam * ((B @ m).float()[None, :] - n @ torch.log(B).t().float())


def objective(Q, n, B, logp, variant, lam, mu=0.0):
    """Phi(Q, B) up to a constant: the smooth part plus the entropy mean_i sum_k Q_ik log Q_ik"""
    Qd = Q.double()
    return float(smooth_objective(Q, n, B, logp, variant, lam, mu) + (Qd * torch.log(Qd.clamp_min(1e-300))).sum() / len(Q))


def stationarity_gap(Q, G):
    """gap(Q, B) = mean_i KL(Q_i || softmax(-G_i)), in nats per pixel; zero exactly at the fixed points of the readout map"""
    Qd = Q.double()
    return float((Qd * (torch.log(Qd.clamp_min(1e-300)) - torch.log_softmax(-G.double(), 1))).sum() / len(Q))


@torch.no_grad()
def solve_mirror_descent(logp, H, variant='E', lam=32.0, mu=0.0, sweeps=30, eps_B=EPS_B, Bs=None, w=0.0):
    """mirror-descent sweeps with the backtracking test (the solver of the reported results).

    logp: N x K tempered log-posterior log p~;  H: callable Q -> HQ.  Each sweep tries eta = min(1, 2 eta_prev) and halves it until
    f(Q+) <= f(Q) + <G, Q+ - Q>/N + KL(Q+ || Q)/(N eta)  (f: smooth part of Phi);  below eta = 1e-3 the step is taken as computed.
    Variant 'G' updates B in closed form after every sweep.  Bs: output of source_class_graph, pooled with the weight w (E, G).
    Returns Q, B (None for 'X') and n = HQ."""
    N = len(logp)
    Q = torch.softmax(logp, 1)
    n = H(Q)
    Bs = scale_to_target(Bs, Q, n, eps_B) if Bs is not None and w > 0 and variant != 'X' else None
    B = pool(class_graph(Q, n, eps_B), Bs, w, eps_B) if variant in ('E', 'G') else None
    eta = 1.0
    h = smooth_objective(Q, n, B, logp, variant, lam, mu)
    for _ in range(sweeps):
        g = gradient(Q, n, B, logp, variant, lam, mu)
        logQ = torch.log(Q.clamp_min(1e-30))
        eta = min(1.0, 2 * eta)
        while True:
            Qn = torch.softmax((logQ - eta * g) / (1 + eta), 1)
            nn_ = H(Qn)
            hn = smooth_objective(Qn, nn_, B, logp, variant, lam, mu)
            kl = (Qn.double() * (torch.log(Qn.clamp_min(1e-30)).double() - logQ.double())).sum() / N
            lin = (g.double() * (Qn - Q).double()).sum() / N
            ok = float(hn) <= float(h + lin + kl / eta) + 1e-9
            if ok or eta < 1e-3:
                break
            eta *= 0.5
        Q, n, h = Qn, nn_, hn
        if variant == 'G':
            B = pool(class_graph(Q, n, eps_B), Bs, w, eps_B)
            h = smooth_objective(Q, n, B, logp, variant, lam, mu)
    return Q, B, n


@torch.no_grad()
def solve_conditional_gradient(logp, H, variant='E', lam=32.0, mu=0.0, sweeps=30, eps_B=EPS_B, sigma=0.5, Bs=None, w=0.0):
    """conditional-gradient sweeps with backtracking on the segment Q + gamma (Q^ - Q), Q^ = softmax(-G).

    The first trial gamma = 1 is the fixed-point step and costs the one product with H of the sweep; further trials evaluate Phi on
    the segment from K x K statistics and the entropy of the mixture, without a product with H.  A trial is accepted when
    Phi(Q+) <= Phi(Q) - sigma gamma gap(Q); below gamma = 1e-4 the iterate is kept if Phi would increase.  Bs and w as in
    solve_mirror_descent (the reported results with the source prior use the mirror-descent sweeps)."""
    N = len(logp)
    lp = logp.double()
    Q = torch.softmax(logp, 1)
    n = H(Q)
    Bs = scale_to_target(Bs, Q, n, eps_B) if Bs is not None and w > 0 and variant != 'X' else None
    B = pool(class_graph(Q, n, eps_B), Bs, w, eps_B) if variant in ('E', 'G') else None
    phi = objective(Q, n, B, logp, variant, lam, mu)
    for _ in range(sweeps):
        g = gradient(Q, n, B, logp, variant, lam, mu)
        gp = stationarity_gap(Q, g)
        Qh = torch.softmax(-g, 1)
        nh = H(Qh)
        gam = 1.0
        val = objective(Qh, nh, B, logp, variant, lam, mu)
        if val <= phi - sigma * gp + 1e-9:
            Q, n = Qh, nh
        else:
            Qd, nd = Q.double(), n.double()
            dQ, dn = Qh.double() - Qd, nh.double() - nd
            f0, f1 = -(Qd * lp).sum() / N, -(dQ * lp).sum() / N
            m0, m1 = Qd.mean(0), dQ.mean(0)
            S0, S1, S2 = (Qd.t() @ nd) / N, (Qd.t() @ dn + dQ.t() @ nd) / N, (dQ.t() @ dn) / N
            while True:
                gam *= 0.5
                m = m0 + gam * m1
                S = S0 + gam * S1 + gam * gam * S2
                U = Qd + gam * dQ
                if variant == 'X':
                    f = f0 + gam * f1 - 0.5 * lam * S.diagonal().sum() + 0.5 * mu * (m * m).sum()
                else:
                    f = f0 + gam * f1 + 0.5 * lam * (m @ B @ m - (S * torch.log(B)).sum())
                val = float(f + (U * torch.log(U.clamp_min(1e-300))).sum() / N)
                if val <= phi - sigma * gam * gp + 1e-9 or gam < 1e-4:
                    break
            if val <= phi:
                Q, n = U.float(), (nd + gam * dn).float()
            val = objective(Q, n, B, logp, variant, lam, mu)
        phi = val
        if variant == 'G':
            B = pool(class_graph(Q, n, eps_B), Bs, w, eps_B)
            phi = objective(Q, n, B, logp, variant, lam, mu)
    return Q, B, n


@torch.no_grad()
def sgwr(posterior, features, k=8, alpha=0.99, steps=50, variant='E', lam=32.0, mu=0.0, temperature=1.0, sweeps=30,
         solver='md', eps_B=EPS_B, source_features=None, source_labels=None, w=0.0):
    """the complete readout.

    posterior: N x K class posteriors of the evaluated pixels (rows sum to one).
    features:  M x d feature vectors of ALL pixels of the target image, the N evaluated pixels first (same order as `posterior`);
               the other M - N rows are the relay nodes.  Pass features of the evaluated pixels only (M = N) for a graph without
               relay nodes.
    variant:   'E' (estimated class graph), 'G' (learned class graph) or 'X' (identity structure; lam and mu).
    solver:    'md' (mirror-descent sweeps, used for the reported results) or 'cg' (conditional-gradient sweeps).
    source_features, source_labels, w:  source prior of 'E' and 'G' with the weight w > 0, see source_class_graph.
    Returns a dict with the labels, the assignment Q, the class masses m, the class graph B (None for 'X') and the final gap."""
    P = torch.as_tensor(posterior).float()
    G_ = torch.as_tensor(features).float().to(P.device)
    N = len(P)
    W, dinv = scene_graph(knn_indices(G_, k))
    H = StructureOperator(W, dinv, N, alpha, steps)
    logp = torch.log(P.clamp_min(1e-12)) / temperature
    Bs = None
    if w > 0 and variant != 'X':
        Fs = torch.as_tensor(source_features).float().to(P.device)
        Bs = source_class_graph(Fs, source_labels, P.shape[1], k, alpha, steps, eps_B)
    solve = solve_mirror_descent if solver == 'md' else solve_conditional_gradient
    Q, B, n = solve(logp, H, variant, lam, mu, sweeps, eps_B, Bs=Bs, w=w)
    gap = stationarity_gap(Q, gradient(Q, n, B, logp, variant, lam, mu))
    return {'labels': Q.argmax(1), 'Q': Q, 'masses': Q.mean(0), 'B': B, 'gap': gap}
