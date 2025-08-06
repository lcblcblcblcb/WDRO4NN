import math
import torch
import cvxpy as cp
import numpy as np
import pandas as pd
from dnn_dro import *
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Any
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

# ──────────────────────────────────────────────────────────────────────
# For Checking Inequalities
# ──────────────────────────────────────────────────────────────────────

# build random 2-layer ReLU network
def init_experiment(seed, n=4, n1=3, K=3, device="cpu"):
    g = g = make_rng(seed, device=device)
    W1 = gaussian((n1, n), rng=g, device=device)
    W2 = gaussian((K , n1), rng=g, device=device)
    x  = gaussian((n,), rng=g, device=device)

    # Jacobian norm at x
    D  = torch.diag((W1 @ x > 0).float())
    Jn = spec_norm(W2 @ D @ W1)

    # Global Lipschitz constant L = √2 * max_D ‖W2 D W1‖
    masks = torch.tensor([[ (i >> k) & 1 for k in range(n1) ]
                          for i in range(2**n1)],
                         dtype=torch.float32, device=device)
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))
    L = sqrt2 * torch.stack([spec_norm(W2 @ torch.diag(m) @ W1)
                             for m in masks]).max()

    return dict(rng=g, W1=W1, W2=W2, x=x, Jn=Jn, L=L, sqrt2=sqrt2)

# Cross-entropy loss for one vector
def ce_loss(vec, W1, W2, target_cls=1):
    logits = (W2 @ relu(W1 @ vec)).unsqueeze(0)
    target = torch.tensor([target_cls], device=vec.device)
    return cross_entropy(logits, target)

# Run Monte Carlo verification
def run_mc(seed, *, n=4, n1=3, K=3, trials=400, noise=0.7, device="cpu"):
    ctx = init_experiment(seed, n, n1, K, device)
    g, W1, W2, x, Jn, L, sqrt2 = ctx.values()
    base_loss = ce_loss(x, W1, W2)

    r1 = r2 = r3 = 0.0
    for _ in range(trials):
        x_p = x + noise * gaussian((n,), rng=g, device=device)
        d   = torch.norm(x_p - x)
        if d == 0: continue

        diff   = (ce_loss(x_p, W1, W2) - base_loss).abs()
        bound1 = sqrt2 * Jn * d
        bound2 = L * d

        r1 = max(r1, (diff / bound1).item())
        r2 = max(r2, (bound1 / bound2).item())
        r3 = max(r3, (diff / bound2).item())

    return dict(seed=seed, J_norm=Jn.item(), L=L.item(),
                max_diff_over_grad=r1,
                max_grad_over_L=r2,
                max_diff_over_L=r3)

# Run pathwise verification  
def run_path(seed, *, n=4, n1=3, K=3, n_pairs=200, grid_pts=300, device="cpu"):
    ctx = init_experiment(seed, n, n1, K, device)
    W1, W2, x, Jn, L, sqrt2 = ctx["W1"], ctx["W2"], ctx["x"], ctx["Jn"], ctx["L"], ctx["sqrt2"]
    base_loss = ce_loss(x, W1, W2)

    recs = []
    for _ in range(n_pairs):
        x_p = x + torch.randn_like(x)
        diff = (ce_loss(x_p, W1, W2) - base_loss).abs()
        dist = torch.norm(x_p - x)

        # Single-cell bound
        b_single = sqrt2 * Jn * dist
        ratio_single = (diff / b_single).item()

        # Pathwise bound
        ts = torch.linspace(0, 1, grid_pts+1, device=device)
        last_x, last_mask = x, (W1 @ x > 0)
        seg_norms, seg_lens = [], []

        for t in ts[1:]:
            xx = x + t * (x_p - x)
            mask = (W1 @ xx > 0)
            if not torch.equal(mask, last_mask):
                seg_norms.append(spec_norm(W2 @ torch.diag(last_mask.float()) @ W1))
                seg_lens.append(torch.norm(xx - last_x))
                last_x, last_mask = xx, mask

        seg_norms.append(spec_norm(W2 @ torch.diag(last_mask.float()) @ W1))
        seg_lens.append(torch.norm(x_p - last_x))

        Jmax = torch.stack(seg_norms).max()
        path_bound = sqrt2 * Jmax * sum(seg_lens)
        ratio_path = (diff / path_bound).item()
        ratio_L    = (diff / (L * dist)).item()

        recs.append(dict(diff=diff.item(),
                         dist=dist.item(),
                         ratio_single=ratio_single,
                         ratio_path=ratio_path,
                         ratio_L=ratio_L,
                         n_segs=len(seg_lens)))
    return pd.DataFrame(recs)

# Compare norms of W2, W1, and their product
def compare_norms(seed: int, n: int = 4, n1: int = 3, K: int = 3, device="cpu"):
    # use shared initializer
    ctx = init_experiment(seed, n=n, n1=n1, K=K, device=device)
    W1, W2 = ctx["W1"], ctx["W2"]

    # individual layer norms
    norm_W1 = spec_norm(W1)
    norm_W2 = spec_norm(W2)
    prod_norm = norm_W1 * norm_W2               # ‖W₂‖₂·‖W₁‖₂

    # full linear map
    full_norm = spec_norm(W2 @ W1)              # ‖W₂W₁‖₂

    # enumerate all binary masks (2^n1)
    masks = torch.tensor([[ (i >> k) & 1 for k in range(n1) ]
                          for i in range(2**n1)],
                         dtype=torch.float32, device=device)
    mask_norms = torch.stack([spec_norm(W2 @ torch.diag(m) @ W1) for m in masks])
    max_mask = mask_norms.max()
    min_mask = mask_norms.min()

    return {
        "seed": seed,
        "‖W₂‖₂·‖W₁‖₂": prod_norm.item(),
        "‖W₂W₁‖₂": full_norm.item(),
        "max‖W₂DW₁‖₂": max_mask.item(),
        "min‖W₂DW₁‖₂": min_mask.item(),
        "ratio (max / prod)": (max_mask / prod_norm).item(),
        "ratio (max / full)": (max_mask / full_norm).item() if full_norm > 0 else float("nan")
    }
    
# ──────────────────────────────────────────────────────────────────────
# For Comparing with Rui Gao
# ──────────────────────────────────────────────────────────────────────

# Compute theoretical, GCK, and empirical LHS
def compare_bounds(
    seeds=range(10),
    Ns=torch.logspace(1, 4, steps=4, base=10).to(torch.int).tolist(),
    *,
    n=4, n1=3, K=3,
    delta0=0.3, t=1.0,
    constant_radius=False,
    device="cpu"
):
    rows = []
    for seed in seeds:
        # fixed network per seed
        ctx = init_experiment(seed, n=n, n1=n1, K=K, device=device)
        W1, W2, L_net = ctx["W1"], ctx["W2"], ctx["L"]

        for N in Ns:
            # radius scaling
            delta_N = delta0 / math.sqrt(N)
            effective_delta = delta0 if constant_radius else delta_N

            # sample data
            g = make_rng(seed + 1000 + N, device=device)
            X = gaussian((N, n), rng=g, device=device)
            Y_idx = torch.randint(0, K, (N,), generator=g, device=device)
            Y = torch.eye(K, device=device)[Y_idx]

            grad_sq, worst_diffs = [], []
            for x, y in zip(X, Y):
                a1  = W1 @ x
                D   = torch.diag((a1 > 0).float())
                z   = W2 @ relu(a1)
                p   = torch.softmax(z, dim=0)
                gvec = (p - y) @ W2 @ D @ W1
                gn   = torch.norm(gvec)
                grad_sq.append(gn.item()**2)

                # one-step PGD move
                if gn > 0:
                    x_adv = x + effective_delta * gvec / gn
                    z_adv = W2 @ relu(W1 @ x_adv)
                    p_adv = torch.softmax(z_adv, dim=0)
                    diff  = -torch.log((p_adv * y).sum()) + torch.log((p * y).sum())
                    worst_diffs.append(diff.item())

            grad_emp = math.sqrt(sum(grad_sq) / len(grad_sq))
            lhs_emp  = max(worst_diffs) if worst_diffs else 0.0

            # theoretical bound
            ours = effective_delta * L_net.item()

            # GCK bound
            gck = (delta_N * grad_emp + 0.125 * delta_N**2
                   + math.sqrt(3) * n1 * math.sqrt(n / N)
                   + delta_N * math.sqrt(t / (2 * N)))

            # rescale GCK to δ0-scale if constant radius
            if constant_radius:
                gck *= math.sqrt(N)

            rows.append(dict(seed=seed, N=N, ours=ours, gck=gck, lhs_emp=lhs_emp))
    return pd.DataFrame(rows).groupby("N").mean().reset_index()

# ──────────────────────────────────────────────────────────────────────
# For Checking Topological Properties
# ──────────────────────────────────────────────────────────────────────

# Check whether 𝓓(i) = 𝓓 for a 1-hidden-layer ReLU network f(x)=W2·ReLU(W1x).
def check_mask_topology(
    n_in=5, n_hidden=6, n_out=1,
    max_random_search=200_000,
    patience=20_000,
    num_samples_segment=120,
    tail_fraction=0.20,
    seed=42,
    device="cpu"
):
    """
    Checks whether 𝓓(i) = 𝓓 for a 1-hidden-layer ReLU network f(x)=W2·ReLU(W1x).
    Returns (bool, D_global, D_i).
    """
    g = make_rng(seed, device=device)

    # random weights
    W1 = torch.randn((n_hidden, n_in), generator=g, device=device)
    W2 = torch.randn((n_out, n_hidden), generator=g, device=device)

    # helpers
    def relu_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.diag((W1 @ x > 0).float())

    def mask_to_int(D: torch.Tensor) -> int:
        bits = (torch.diag(D) > 0).int().cpu().numpy()
        return int("".join(str(b) for b in bits), 2)

    # 1) Monte Carlo: find witnesses for all feasible masks
    witness = {}
    tries_since_new = 0
    for _ in range(max_random_search):
        x = torch.randn((n_in,), generator=g, device=device) * 3
        key = mask_to_int(relu_mask(x))
        if key not in witness:
            witness[key] = x
            tries_since_new = 0
        else:
            tries_since_new += 1
        if tries_since_new > patience:
            break
    D_global = set(witness.keys())
    total_masks = 2 ** n_hidden
    print(f"\n[Step 1] Found {len(D_global)} distinct feasible masks out of {total_masks} possible.")

    # 2) Build 𝓓(i): walk straight segments from x^(i) to each x^D
    x_i = torch.randn((n_in,), generator=g, device=device)
    D_i = set()
    for _, xD in witness.items():
        segment = x_i + torch.linspace(0.0, 1.0, num_samples_segment, device=device).unsqueeze(1) * (xD - x_i)
        tail = segment[int((1 - tail_fraction) * num_samples_segment):]
        for z in tail:
            D_i.add(mask_to_int(relu_mask(z)))

    # 3) Compare & pretty print
    fmt = lambda k: format(k, f"0{n_hidden}b")
    def format_set(S): return "{ " + ", ".join(sorted(fmt(k) for k in S)) + " }"

    print("\n[Step 2] Feasible masks 𝓓:")    ; print(format_set(D_global))
    print("\n[Step 3] Segment-tail masks 𝓓(i):"); print(format_set(D_i))
    equal = D_global == D_i
    print(f"\nResult: 𝓓(i) {'equals' if equal else 'DOES NOT equal'} 𝓓.")
    # return equal, D_global, D_i
    return

# helper to find u with CVXPY
def find_feasible_u(J, w, W1, D_star,
                    n_sols=5, eps=1e-5, tol=1e-6, solver=cp.SCS):
    """
    Return up to n_sols vectors u with
        J u = w,   ||u||₂ = 1,   and   u ∈ Ω(D_star)  (strict ε-margin).
    """
    n          = J.shape[1]
    D          = D_star.astype(float)                 # (n1,n1) diagonal
    I_minus_D  = np.eye(D.shape[0]) - D
    sols, rng  = [], np.random.default_rng()

    for _ in range(n_sols):
        u = cp.Variable(n)
        t = cp.Variable()                             # radial slack (≤ 1)
        jitter = rng.standard_normal(n)               # random affine obj

        prob = cp.Problem(cp.Maximize(t + 1e-3 * jitter @ u), [
            J @ u == w,
            cp.norm(u, 2) <= t,
            t <= 1,
            D         @ W1 @ u >=  eps,               # active neurons
            I_minus_D @ W1 @ u <= -eps                # inactive neurons
        ])
        prob.solve(solver=solver, eps=tol, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            break

        u_val = u.value / np.linalg.norm(u.value)     # force ||u||₂ = 1
        # de-duplicate (up to sign)
        if all(min(np.linalg.norm(u_val - v),
                   np.linalg.norm(u_val + v)) > 1e-3 for v in sols):
            sols.append(u_val)

    return sols

# Satisfy (D2) for a 1-hidden-layer ReLU network
def build_D2_satisfying(n=2, n1=2, K=2, seed=42,
                        tol=1e-4, device="cpu", max_tries=5000,
                        n_sols_per_mask=5):
    """
    Sample (W1, W2) until (D2) holds and at least one unit vector u
    satisfying all constraints is found via CVXPY.
    Returns:
        W1, W2, D_star, J_star, radius, w, u_list, c, j_plus
    """
    # -----------------------  torch RNG helpers
    g = torch.Generator(device).manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if device.startswith("cuda") else None
    np_rng = np.random.default_rng(seed)
    e      = torch.eye(K, device=device)

    # -----------------------  random label
    c = np_rng.integers(0, K)

    # -----------------------  all 2^n1 binary masks
    masks = ((torch.arange(2 ** n1, device=device).unsqueeze(1)
              >> torch.arange(n1, device=device)) & 1).float()

    # -----------------------  try / repeat
    for _ in range(max_tries):
        W1 = torch.randn(n1, n,  generator=g, device=device)
        W2 = torch.randn(K,  n1, generator=g, device=device)

        # Jacobians for every mask
        W1_masked = masks.unsqueeze(2) * W1
        J_all     = torch.einsum("kn,bnj->bkj", W2, W1_masked)  # (2^n1,K,n)
        sigmas    = torch.linalg.svdvals(J_all)[:, 0]           # spectral norm
        best_idx  = torch.argmax(sigmas)
        radius    = sigmas[best_idx].item()
        J_star_t  = J_all[best_idx]
        D_star    = torch.diag(masks[best_idx]).cpu().numpy()

        # full-rank check
        if torch.linalg.matrix_rank(J_star_t) < min(n, K):
            continue

        # ---------- convert to NumPy once for CVXPY
        J_star_np = J_star_t.cpu().numpy()
        W1_np     = W1.cpu().numpy()

        # ---------- loop over j_+ ≠ c
        for j_plus in (j for j in range(K) if j != c):
            w_t = (radius / math.sqrt(2)) * (e[j_plus] - e[c])
            w_np = w_t.cpu().numpy()

            # ---- find *all* feasible u inside Ω
            u_list = find_feasible_u(J_star_np, w_np, W1_np, D_star,
                                     n_sols=n_sols_per_mask)
            if u_list:                      # success!
                return (W1_np, W2.cpu().numpy(), D_star, J_star_np,
                        radius, w_np, u_list, c, j_plus)

    raise RuntimeError("Failed to satisfy (D2) within max_tries.")

# Pretty print the results of build_D2_satisfying
def pretty_print_D2(W1, W2, D_star, J_star, radius, w, u, c, j_plus):
    np.set_printoptions(precision=4, suppress=True)
    def banner(text): print(f"\n=== {text} ===")
    def mat(name, M): print(f"{name} =\n{M}\n")

    # Network ingredients
    banner("NETWORK INGREDIENTS")
    mat("W1", W1)
    mat("W2", W2)
    print("D* (bitstring) =", "".join(str(int(x)) for x in np.diag(D_star)))
    mat("D*", D_star)
    mat("J*", J_star)
    print(f"‖J*‖₂ = {radius:.4f}")

    # Strict ReLU cell
    banner("STRICT ReLU CELL 𝒞*")
    for k in range(W1.shape[0]):
        coef = "D*W1" if D_star[k, k] == 1 else "(I−D*)W1"
        sign = ">" if D_star[k, k] == 1 else "<"
        a, b = W1[k].round(4)
        print(f"  {coef}[{k}] · x {sign} 0   (⇒ {a:+.4f}·x₁ {b:+.4f}·x₂ {sign} 0)")

    # Recession cone
    banner("RECESSION CONE rec(𝒞*)")
    for k in range(W1.shape[0]):
        coef = "D*W1" if D_star[k, k] == 1 else "(I−D*)W1"
        cmp = "≥" if D_star[k, k] == 1 else "≤"
        a, b = W1[k].round(4)
        print(f"  {coef}[{k}] · u {cmp} 0   (⇒ {a:+.4f}·u₁ {b:+.4f}·u₂ {cmp} 0)")
    print("Ω = int(rec(𝒞*))  (all inequalities strict)")

    # Image cone
    banner("IMAGE CONE V AND SLICE V*")
    print("V = J* Ω  (linear image of the interior cone)")
    print(f"Sphere radius = ‖J*‖₂ = {radius:.4f}")
    print("V* = V ∩ { z : ‖z‖₂ = ‖J*‖₂ }  (open arc on that circle)")

    # Proof vector
    banner("PROOF VECTOR w & UNIT u")
    print(f"c = {c}, j+ = {j_plus}")
    print("w =", w.round(4))
    print(f"‖w‖₂ = {np.linalg.norm(w):.4f} (matches ‖J*‖₂)")
    print("u =", u.round(4))
    print(f"‖u‖₂ = {np.linalg.norm(u):.4f} (must be 1)")
    
# Assumption A2: compute ψ(Z^{(i)} + T u), slope, and ε(T)
def compute_a2_data(W1, W2, J_star, Z_i, u, radius, y_true=0, T_max=1000, steps=200, device="cpu"):
    """
    Compute ψ(Z^{(i)} + T u), slope, and ε(T) for Assumption A2.
    """
    W1_t = torch.tensor(W1, dtype=torch.float64, device=device)
    W2_t = torch.tensor(W2, dtype=torch.float64, device=device)
    y = torch.tensor([y_true], device=device)
    
    def ce_loss(x_vec):
        logits = (W2_t @ relu(W1_t @ x_vec)).unsqueeze(0)
        return cross_entropy(logits, y)

    psi_Z_i = ce_loss(torch.tensor(Z_i, dtype=torch.float64, device=device)).item()
    L = np.sqrt(2) * radius

    T_vals = np.linspace(1e-3, T_max, num=steps)
    psi_vals, slopes, epsilons, d_vals = [], [], [], []

    for T in T_vals:
        Z_eps = Z_i + T * u
        psi_T = ce_loss(torch.tensor(Z_eps, dtype=torch.float64, device=device)).item()
        d_val = np.linalg.norm(Z_eps - Z_i)
        slope = (psi_T - psi_Z_i) / d_val
        psi_vals.append(psi_T)
        slopes.append(slope)
        epsilons.append(L - slope)
        d_vals.append(d_val)

    return dict(
        d_vals=d_vals,
        psi_vals=psi_vals,
        slopes=slopes,
        epsilons=epsilons,
        psi_Z_i=psi_Z_i,
        L=L
    )
    
# Plot Assumption A2: ReLU cell, cone slice, slope, and ε(T)
def plot_assumption_a2(W1, W2, D_star, J_star, Z_i, u, radius, a2_data, w=None, T_max=2):
    d_vals = a2_data["d_vals"]
    psi_vals = a2_data["psi_vals"]
    slopes = a2_data["slopes"]
    epsilons = a2_data["epsilons"]
    psi_Z_i = a2_data["psi_Z_i"]
    L = a2_data["L"]

    # ReLU cell mask for shading
    grid = np.linspace(-8, 8, 1000)
    xx, yy = np.meshgrid(grid, grid)
    cond = np.ones_like(xx, dtype=bool)
    for k in range(2):
        lin = W1[k, 0] * xx + W1[k, 1] * yy
        cond &= lin > 0 if D_star[k, k] == 1 else lin < 0

    # Ray endpoint
    ray_end = Z_i + T_max * u

    # Cone slice
    n = J_star.shape[0]
    Jinv = np.linalg.inv(J_star)
    A = np.vstack([(1 if D_star[k, k] else -1) * (W1[k] @ Jinv) for k in range(n)])
    theta = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    circle_pts = np.c_[np.cos(theta), np.sin(theta)] * radius
    mask = (A @ circle_pts.T > 0).all(axis=0)
    idx_true = np.where(mask)[0]
    splits = np.where(np.diff(idx_true) != 1)[0] + 1
    segments = np.split(idx_true, splits)
    if segments[0][0] == 0 and segments[-1][-1] == len(theta) - 1:
        segments[0] = np.concatenate([segments[-1], segments[0]])
        segments.pop()
    arc_pts = circle_pts[max(segments, key=len)]

    # === FIGURE ===
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.4, wspace=0.35)

    # Row 1: ReLU cell
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.contourf(xx, yy, cond.astype(int), levels=[0.5, 1.5], alpha=0.2)
    x_line = np.linspace(-8, 8, 900)
    for k in range(W1.shape[0]):
        a, b = W1[k]
        style = '-' if D_star[k, k] else '--'
        if abs(b) > 1e-12:
            ax0.plot(x_line, (-a / b) * x_line, style, lw=1.2)
        else:
            ax0.axvline(0, ls=style, lw=1.2)
    ax0.plot(*Z_i, 'ro'); ax0.text(Z_i[0]+.1, Z_i[1], r'$Z^{(i)}$', color='r')
    ax0.arrow(Z_i[0], Z_i[1], *(ray_end - Z_i), width=0.015,
              head_width=.1, head_length=.15, length_includes_head=True)
    ax0.arrow(0, 0, *u, ls=':', alpha=.5,
              head_width=.07, head_length=.1, length_includes_head=True)
    ax0.set_xlim([-3, 3]); ax0.set_ylim([-3, 3])
    ax0.set_aspect('equal'); ax0.grid(alpha=.3)
    ax0.set_xlabel('$x_1$'); ax0.set_ylabel('$x_2$')
    ax0.set_title('Strict cell $\\mathcal{C}^*$')

    # Row 1: Cone slice
    ax1 = fig.add_subplot(gs[0, 1])
    cone_poly = np.vstack([[0, 0], arc_pts])
    ax1.add_patch(Polygon(cone_poly, closed=True, facecolor="#d62728", alpha=0.18))
    ax1.plot(arc_pts[:, 0], arc_pts[:, 1], color="#d62728", lw=2)
    ax1.add_patch(plt.Circle((0, 0), radius, edgecolor="gray", linestyle="--", facecolor="none"))
    if w is not None:
        ax1.arrow(0, 0, w[0], w[1], color='green', width=0.005,
                  head_width=0.1, head_length=0.16, length_includes_head=True)
        ax1.text(w[0]*1.1, w[1]*1.1, r"$w$", color="green", fontsize=12)
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3)
    ax1.set_title(r"Filled cone slice $V^*$")

    # Row 2: Slope
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(d_vals, slopes); ax2.axhline(y=L, color='red', linestyle='--')
    ax2.set_title('Slope vs $T$'); ax2.grid(alpha=0.3)

    # Row 2: Psi
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(d_vals, psi_vals, color='purple'); ax3.axhline(y=psi_Z_i, color='gray', linestyle='--')
    ax3.set_title(r'$\psi$ vs $T$'); ax3.grid(alpha=0.3)

    # Row 2: Epsilon
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(d_vals, epsilons, color='orange')
    ax4.set_title(r'$\varepsilon(T)$'); ax4.grid(alpha=0.3)

    plt.show()

# Estimate the probability of satisfying (D2) for a range of parameters
def estimate_d2_probability(param_name, param_values, fixed_params, trials=500):
    results = []
    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        success = 0
        for seed in range(trials):
            try:
                build_D2_satisfying(
                    n=params['n'], n1=params['n1'], K=params['K'],
                    seed=seed, max_tries=1
                )
                success += 1
            except RuntimeError:
                continue
        prob = success / trials
        results.append((val, prob, success, trials))
        print(f"{param_name}={val}, prob={prob*100}%, success={success}, trials={trials}")
    return pd.DataFrame(results, columns=[param_name, 'probability', 'success', 'trials'])