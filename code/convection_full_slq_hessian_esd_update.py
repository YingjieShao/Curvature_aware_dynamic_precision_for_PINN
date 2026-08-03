# Full-model Hessian spectral-density and condition-number diagnostics for train_dynamic_precision_fixed_beta_50_curvature.
# This implements the same diagnostic family used in Rathore et al.:
# Hessian-vector products + stochastic Lanczos quadrature (SLQ).
# Condition number is max |lambda| / min |lambda| from the estimated Hessian eigenvalues.
#

import sys
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from collections import deque
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(str(Path("../PINN_FP64-main/")))

from models import QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only, PINNMamba
from util import make_time_sequence
step_size = 1e-4
num_step=5
seq_diff = int(1e-2/step_size)
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def flatten_params(model):
    """Utility to flatten model parameters into a single 1D tensor."""
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def relative_improvement(history):
    """Calculates the relative improvement of the loss over the history window."""
    if len(history) < 2:
        return float("inf")
    old = history[0]
    new = history[-1]
    return (old - new) / (abs(old) + 1e-12)


def init_model(model_name='PINN', hidden_dim=512, num_layer=4, dtype=torch.float64):
    """Initialize model with support for different architectures.

    Args:
        model_name: 'PINN', 'PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba', 'KAN', 'QRes', 'ProPINN'
        hidden_dim: Hidden dimension for PINN (default 512)
        num_layer: Number of layers for PINN (default 4)
        dtype: torch.float32 or torch.float64
    """

    if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        model = PINNsFormer.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'KAN':
        model = KAN.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0,
                          noise_scale_base=0.25, device=device).to(dtype).to(device)
    elif model_name == 'QRes':
        model = QRes.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'PINNMamba':
        model = PINNMamba.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(dtype).to(device)
        model.apply(init_weights)
    else:  # Default to PINN
        model = PINN(hidden_dim=hidden_dim, num_layer=num_layer).to(dtype).to(device)
        model.apply(init_weights)
    return model


def preprocess_data_for_model(model_name, res, b_left, b_right, b_upper, b_lower, num_step=5, step_size=1e-4):
    """Apply model-specific data preprocessing.

    PINNsFormer and PINNMamba require temporal sequences: (N, 2) -> (N, num_step, 2)
    Other models use standard shape: (N, 2)
    """
    if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
        res = make_time_sequence(res, num_step=num_step, step=step_size)
        b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
        b_right = make_time_sequence(b_right, num_step=num_step, step=step_size) if b_right is not None else None
        b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
        b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    return res, b_left, b_right, b_upper, b_lower


def get_data(x_range, y_range, x_num, y_num):
    # Same as fp64 paper
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x, t)

    data = np.concatenate(
        (np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)),
        axis=-1
    )

    b_left = data[0, :, :]
    b_right = data[-1, :, :]
    b_upper = data[:, -1, :]
    b_lower = data[:, 0, :]
    res = data.reshape(-1, 2)

    return res, b_left, b_right, b_upper, b_lower
def get_convection_data(dtype, x_num=101, t_num=101):
    res, b_left, b_right, b_upper, b_lower = get_data(
        [0, 2 * np.pi], [0, 1], x_num, t_num
    )

    res = torch.tensor(res, dtype=dtype, requires_grad=True, device=device)
    b_left = torch.tensor(b_left, dtype=dtype, requires_grad=True, device=device)
    b_right = torch.tensor(b_right, dtype=dtype, requires_grad=True, device=device)
    b_upper = torch.tensor(b_upper, dtype=dtype, requires_grad=True, device=device)
    b_lower = torch.tensor(b_lower, dtype=dtype, requires_grad=True, device=device)

    x_res, t_res = res[:, 0:1], res[:, 1:2]
    x_left, t_left = b_left[:, 0:1], b_left[:, 1:2]
    x_right, t_right = b_right[:, 0:1], b_right[:, 1:2]
    x_upper, t_upper = b_upper[:, 0:1], b_upper[:, 1:2]
    x_lower, t_lower = b_lower[:, 0:1], b_lower[:, 1:2]

    return (
        x_res, t_res,
        x_left, t_left,
        x_right, t_right,
        x_upper, t_upper,
        x_lower, t_lower
    )
class PINN(nn.Module):
    def __init__(self,hidden_dim=512, num_layer=4, in_dim=2,  out_dim=1):
        super(PINN, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        return self.linear(src)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def plot_convection_results(model, beta, run_name, strategy, model_name='PINN'):
    print("Generating results plot...")
    nx, nt = 101, 101
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.linspace(0, 1, nt)
    T, X = np.meshgrid(t, x, indexing="ij")
    u_exact = np.sin(X - beta * T)

    dtype = next(model.parameters()).dtype
    x_flat = X.flatten()[:, None]
    t_flat = T.flatten()[:, None]

    # model.eval()
    with torch.no_grad():
        ev_data = np.concatenate((x_flat, t_flat), axis=-1)
        if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
            ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
            x_t = torch.tensor(ev_data[..., 0:1], dtype=dtype, device=device)
            t_t = torch.tensor(ev_data[..., 1:2], dtype=dtype, device=device)
        else:
            x_t = torch.tensor(x_flat, dtype=dtype, device=device)
            t_t = torch.tensor(t_flat, dtype=dtype, device=device)

        u_pred = model(x_t, t_t)
        if u_pred.dim() == 3: u_pred = u_pred[:, -1, :]
        u_pred = u_pred.cpu().numpy().reshape(T.shape)

    error = np.abs(u_exact - u_pred)
    rRMSE = np.sqrt(np.sum((u_exact - u_pred) ** 2) / np.sum(u_exact ** 2))
    rMAE = np.sum(np.abs(u_exact - u_pred)) / np.sum(np.abs(u_exact))

    wandb.run.summary["rRMSE"] = rRMSE
    wandb.run.summary["rMAE"] = rMAE

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [0, 2 * np.pi, 1, 0]

    im0 = axes[0].imshow(u_exact, extent=extent, aspect="auto", cmap='viridis', origin="upper")
    axes[0].set_title(f"Exact Solution (beta={beta:.2f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, extent=extent, aspect="auto", cmap='viridis', origin="upper")
    axes[1].set_title(f"Prediction (rRMSE={rRMSE:.4e})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=extent, aspect="auto", cmap='coolwarm', origin="upper", vmin=-0.15, vmax=0.15,)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    filename = f"result_{run_name}_{strategy}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    wandb.log({"Result Plot": wandb.Image(filename)})
    plt.close(fig)
    print(f"Plot saved to {filename}")


def _trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def _num_parameters(params):
    return int(sum(p.numel() for p in params))


def _flatten_grads_or_zeros(grads, params):
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def _hessian_vector_product(loss_fn, params, vector):
    """
    Full-model Hessian-vector product v -> H v using Pearlmutter/autodiff.
    This is to avoid form the dense Hessian.
    """
    loss = loss_fn()
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    flat_grad = _flatten_grads_or_zeros(grads, params)
    grad_dot_vector = torch.dot(flat_grad, vector)
    hvp = torch.autograd.grad(
        grad_dot_vector,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return _flatten_grads_or_zeros(hvp, params).detach()


def _lanczos_tridiagonal_hessian(
        loss_fn,
        params,
        num_lanczos=30,
        reorthogonalize=True,
        eps=1e-12,
):
    """
    One stochastic Lanczos run for the full PINN-loss Hessian.
    Returns the tridiagonal matrix T whose eigenvalues/weights define one SLQ estimate.
    """
    n_params = _num_parameters(params)
    dtype = params[0].dtype
    dev = params[0].device

    q = torch.randint(0, 2, (n_params,), dtype=dtype, device=dev)
    q[q == 0] = -1
    q = q / (torch.linalg.norm(q) + eps)
    q_prev = torch.zeros_like(q)

    alphas = []
    betas = []
    basis = []
    beta_prev = torch.tensor(0.0, dtype=dtype, device=dev)

    for j in range(num_lanczos):
        if reorthogonalize:
            basis.append(q.detach().clone())

        z = _hessian_vector_product(loss_fn, params, q)
        if j > 0:
            z = z - beta_prev * q_prev

        alpha = torch.dot(q, z)
        z = z - alpha * q

        if reorthogonalize:
            # Full reorthogonalization improves numerical stability of Ritz values.
            for qi in basis:
                z = z - torch.dot(z, qi) * qi

        beta = torch.linalg.norm(z)
        alphas.append(alpha.detach())

        if j < num_lanczos - 1:
            if beta.item() < eps:
                break
            betas.append(beta.detach())
            q_prev = q
            q = z / (beta + eps)
            beta_prev = beta.detach()

    m = len(alphas)
    T = torch.zeros((m, m), dtype=dtype, device=dev)
    for i, a in enumerate(alphas):
        T[i, i] = a
    for i, b in enumerate(betas[:max(0, m - 1)]):
        T[i, i + 1] = b
        T[i + 1, i] = b
    return T.detach()


def estimate_hessian_spectral_density_slq(
        model,
        loss_fn,
        num_probes=2,
        num_lanczos=30,
        num_grid=400,
        sigma=None,
        lanczos_eps=1e-12,
        reorthogonalize=True,
):
    """
    Full-model estimated Hessian spectral density using HVP + SLQ.

    This follows the same diagnostic used in Rathore et al.:
    estimate the Hessian eigenvalue distribution using Hessian-vector products
    and stochastic Lanczos quadrature, without forming the dense Hessian.

    The condition number is computed from the estimated Hessian eigenvalues as

        kappa = max_i |lambda_i| / min_i |lambda_i|,

    matching the definition in the paper.
    """
    params = _trainable_parameters(model)
    n_params = _num_parameters(params)

    all_nodes = []
    all_weights = []

    for probe_idx in range(num_probes):
        T = _lanczos_tridiagonal_hessian(
            loss_fn=loss_fn,
            params=params,
            num_lanczos=num_lanczos,
            reorthogonalize=reorthogonalize,
            eps=lanczos_eps,
        )
        eigvals, eigvecs = torch.linalg.eigh(T)
        nodes = eigvals.detach().cpu().numpy().astype(np.float64)
        weights = (eigvecs[0, :] ** 2).detach().cpu().numpy().astype(np.float64)
        weights = weights / (np.sum(weights) + 1e-300)
        all_nodes.append(nodes)
        all_weights.append(weights)

    nodes = np.concatenate(all_nodes)
    weights = np.concatenate(all_weights)
    weights = weights / (np.sum(weights) + 1e-300)

    finite = np.isfinite(nodes) & np.isfinite(weights)
    nodes = nodes[finite]
    weights = weights[finite]
    weights = weights / (np.sum(weights) + 1e-300)

    if nodes.size == 0:
        raise RuntimeError("SLQ produced no finite Ritz values.")

    eig_min = float(np.min(nodes))
    eig_max = float(np.max(nodes))
    if eig_min == eig_max:
        eig_min -= 1.0
        eig_max += 1.0

    grid = np.linspace(eig_min, eig_max, num_grid)
    if sigma is None:
        sigma_used = 0.01 * max(eig_max - eig_min, 1e-12)
    else:
        sigma_used = float(sigma)
    sigma_used = max(sigma_used, 1e-16)

    density = np.zeros_like(grid, dtype=np.float64)
    norm_const = 1.0 / (sigma_used * np.sqrt(2.0 * np.pi))
    for lam, w in zip(nodes, weights):
        density += w * norm_const * np.exp(-0.5 * ((grid - lam) / sigma_used) ** 2)

    abs_nodes = np.abs(nodes)
    lambda_abs_max = float(np.max(abs_nodes))
    lambda_abs_min = float(np.min(abs_nodes))

    if lambda_abs_min > 0.0:
        condition_number = float(lambda_abs_max / lambda_abs_min)
        log10_condition_number = float(np.log10(condition_number))
    else:
        condition_number = float("inf")
        log10_condition_number = float("inf")

    return {
        "n_params": n_params,
        "num_probes": int(num_probes),
        "num_lanczos": int(num_lanczos),
        "num_grid": int(num_grid),
        "sigma": float(sigma_used),
        "nodes": nodes,
        "weights": weights,
        "grid": grid,
        "density": density,
        "lambda_min_est": eig_min,
        "lambda_max_est": eig_max,
        "lambda_abs_min_est": lambda_abs_min,
        "lambda_abs_max_est": lambda_abs_max,
        "condition_number": condition_number,
        "log10_condition_number": log10_condition_number,
    }


def plot_slq_hessian_condition_history(slq_history, run_name, strategy):
    if len(slq_history) == 0:
        return None

    steps = np.asarray([h["step"] for h in slq_history], dtype=np.float64)
    log_condition = np.asarray([h["log10_condition_number"] for h in slq_history], dtype=np.float64)
    lambda_abs_max = np.asarray([h["lambda_abs_max_est"] for h in slq_history], dtype=np.float64)
    lambda_abs_min = np.asarray([h["lambda_abs_min_est"] for h in slq_history], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(steps, log_condition, marker="o")
    axes[0].set_ylabel(r"$\log_{10}\kappa(H)$")
    axes[0].set_title("Estimated full-model Hessian condition number by SLQ/HVP")

    axes[1].semilogy(steps, lambda_abs_max + 1e-300, marker="o", label=r"$\max |\lambda|$")
    axes[1].semilogy(steps, lambda_abs_min + 1e-300, marker="o", label=r"$\min |\lambda|$")
    axes[1].set_ylabel("Eigenvalue magnitude")
    axes[1].legend()

    precision_fp64 = np.asarray([1 if h["precision_state"] == "fp64" else 0 for h in slq_history], dtype=np.float64)
    axes[2].step(steps, precision_fp64, where="post")
    axes[2].set_ylabel("FP64 state")
    axes[2].set_xlabel("Training step")
    axes[2].set_yticks([0, 1])

    fig.tight_layout()
    out_path = f"{run_name}_{strategy}_full_hessian_slq_condition_history.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_slq_hessian_spectral_density(slq_history, run_name, strategy, max_curves=6):
    if len(slq_history) == 0:
        return None

    if len(slq_history) <= max_curves:
        chosen = slq_history
    else:
        idx = np.linspace(0, len(slq_history) - 1, max_curves).astype(int)
        chosen = [slq_history[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    for h in chosen:
        ax.semilogy(h["grid"], h["density"] + 1e-300, linewidth=1.5, label=f"step {h['step']} ({h['precision_state']})")

    ax.set_xlabel("Estimated Hessian eigenvalue")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Full-model Hessian spectral density estimated by SLQ/HVP")
    ax.legend()
    fig.tight_layout()

    out_path = f"{run_name}_{strategy}_full_hessian_slq_spectral_density.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def train_dynamic_precision_fixed_beta_50_curvature(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=5000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = 'PINN',
        slq_enabled=True,
        slq_interval=500,
        slq_num_probes=2,
        slq_num_lanczos=30,
        slq_num_grid=400,
        slq_sigma=None,
):
    set_seed(seed)
    run_name = f"convection_dynamic_precision_curvature_fixedbeta_{BETA_MAX}_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Curvature Run (Convection): {run_name}")
    print("=" * 60)

    MIN_SWITCH_STEP = 0
    MIN_DWELL_STEPS = 10
    COND_WINDOW = 10
    TRIGGER_PATIENCE = 10
    STUCK_PATIENCE = 1
    DIAGNOSTIC_INTERVAL = 10

    CURV_EPS = 1e-30
    LOG_PROXY_LOW = 2.5
    LOG_PROXY_HIGH = 2.5
    LOG_SLOPE_FLAT = 0.02
    LOG_SLOPE_UP = 0.03
    EMA_BETA = 0.9

    if benchmark is not False:
        strategy = f"{benchmark}_curvature"
        wandb.init(
            project="pinn_convection_dynamic_precision_plotforillcondition_analysis{}".format(model_name),
            name=run_name,
            config={"strategy": strategy, "seed": seed, "beta": BETA_MAX, "min_switch_step": 'NA',
                    "min_dwell_steps": 'NA', "cond_window": 'NA',
                    "trigger_patience": 'NA',
                    "stuck_patience": 'NA',
                    "slq_enabled": slq_enabled,
                    "slq_interval": slq_interval,
                    "slq_num_probes": slq_num_probes,
                    "slq_num_lanczos": slq_num_lanczos,
                    "slq_num_grid": slq_num_grid},
            reinit=True
        )
    else:
        strategy = "dynamic"

        wandb.init(
            project="pinn_convection_dynamic_precision_plotforillcondition_analysis{}".format(model_name),
            name=run_name,
            config={"strategy": strategy, "seed": seed, "beta": BETA_MAX, "min_switch_step": MIN_SWITCH_STEP,
                    "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW, "trigger_patience": TRIGGER_PATIENCE,
                    "stuck_patience": STUCK_PATIENCE, "LOG_PROXY_LOW": LOG_PROXY_LOW,
                    "slq_enabled": slq_enabled,
                    "slq_interval": slq_interval,
                    "slq_num_probes": slq_num_probes,
                    "slq_num_lanczos": slq_num_lanczos,
                    "slq_num_grid": slq_num_grid},
            reinit=True
        )

    beta_used = BETA_MAX

    if benchmark is False:
        precision_state, current_dtype = "fp64", torch.float64
    elif benchmark == "switching":
        precision_state, current_dtype = "fp32", torch.float32
    elif benchmark == "fp64":
        precision_state, current_dtype = "fp64", torch.float64
    elif benchmark == "fp32":
        precision_state, current_dtype = "fp32", torch.float32
    else:
        raise ValueError(f"Unsupported benchmark value: {benchmark}")

    last_switch_step = 0
    has_switched = False

    model = init_model(model_name, hidden_dim=512, num_layer=4, dtype=current_dtype)
    print(model)

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data([0, 2 * np.pi], [0, 1], 101, 101)
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = preprocess_data_for_model(
        model_name, res_np, b_left_np, b_right_np, b_upper_np, b_lower_np
    )

    data_cache = {}
    for dt in [torch.float32, torch.float64]:
        data_cache[dt] = {
            "res": torch.tensor(res_np, dtype=dt, device=device),
            "b_left": torch.tensor(b_left_np, dtype=dt, device=device),
            "b_right": torch.tensor(b_right_np, dtype=dt, device=device),
            "b_upper": torch.tensor(b_upper_np, dtype=dt, device=device),
            "b_lower": torch.tensor(b_lower_np, dtype=dt, device=device),
        }

    def _cast_obj(obj, dtype, device):
        if torch.is_tensor(obj):
            if obj.is_floating_point(): return obj.to(device=device, dtype=dtype)
            return obj.to(device=device)
        elif isinstance(obj, dict):
            return {k: _cast_obj(v, dtype, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_cast_obj(v, dtype, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(_cast_obj(v, dtype, device) for v in obj)
        else:
            return obj

    def rebuild_data_and_optimizer(dtype, model, old_optimizer=None):
        res = data_cache[dtype]["res"].detach().clone().requires_grad_(True)
        b_left = data_cache[dtype]["b_left"].detach().clone().requires_grad_(True)
        b_right = data_cache[dtype]["b_right"].detach().clone().requires_grad_(True)
        b_upper = data_cache[dtype]["b_upper"].detach().clone().requires_grad_(True)
        b_lower = data_cache[dtype]["b_lower"].detach().clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                                      tolerance_change=1e-10)
        if old_optimizer is not None and isinstance(old_optimizer, torch.optim.LBFGS):
            print(f"Optimizer type 'LBFGS' is the same. Attempting to transfer state.")
            try:
                old_state_dict = old_optimizer.state_dict()
                old_state_dict["state"] = _cast_obj(old_state_dict["state"], dtype, device)
                optimizer.load_state_dict(old_state_dict)
                print("Successfully transferred optimizer state.")
            except Exception as e:
                print(f"Could not transfer optimizer state: {e}. Re-initializing optimizer from scratch.")
        else:
            if old_optimizer is not None:
                print(f"Optimizer types differ ({type(old_optimizer).__name__} -> LBFGS). Re-initializing optimizer.")
            else:
                print("No old optimizer provided. Initializing optimizer from scratch.")

        x_res, t_res = res[..., 0:1], res[..., 1:2]
        x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
        x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
        x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
        x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

        return optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower

    optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower = \
        rebuild_data_and_optimizer(current_dtype, model)

    def compute_losses():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        if rescale_derivative:
            S = 1024.0
            u_scaled = pred_res * S
            u_t_scaled = torch.autograd.grad(u_scaled, t_res, grad_outputs=torch.ones_like(u_scaled), retain_graph=True,
                                             create_graph=True)[0]
            u_x_scaled = torch.autograd.grad(u_scaled, x_res, grad_outputs=torch.ones_like(u_scaled), retain_graph=True,
                                             create_graph=True)[0]
            u_t, u_x = u_t_scaled / S, u_x_scaled / S
        else:
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]
            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

        loss_res = torch.mean((u_t + beta_used * u_x) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[..., 0] - torch.sin(x_left[..., 0])) ** 2)
        loss = loss_res + loss_bc + loss_ic
        return loss, loss_res, loss_ic, loss_bc

    def get_quasi_newton_curvature_proxy():
        if not optimizer.state: return None
        state = optimizer.state[next(iter(optimizer.state))]
        if "old_dirs" not in state or "old_stps" not in state: return None
        directional_curvatures = []
        for y, s in zip(state["old_dirs"], state["old_stps"]):
            y, s = y.flatten(), s.flatten()
            ys, ss = torch.dot(s, y).item(), torch.dot(s, s).item()
            if np.isfinite(ys) and np.isfinite(ss) and ss > CURV_EPS and ys > CURV_EPS:
                directional_curvatures.append(ys / ss)
        if len(directional_curvatures) < 2: return None
        directional_curvatures = np.asarray(directional_curvatures, dtype=np.float64)
        k_min = max(np.min(directional_curvatures), 1e-12)
        k_max = max(np.max(directional_curvatures), k_min)
        return float(k_max / k_min)

    cond_history = deque(maxlen=COND_WINDOW)
    stats = {}
    fp32_tiny_update_count = 0
    switch_event = 0
    smoothed_log_proxy = prev_smoothed_log_proxy = None
    proxy_slope = 0.0
    curvature_proxy = None
    best_loss = float('inf')
    stagnant_steps = 0
    start_time = time.time()

    slq_history = []
    max_rel_update = 0.0
    rRMSE = None
    rMAE = None

    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        params_before = [p.detach().clone() for p in model.parameters()]

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, loss_res, loss_ic, loss_bc = compute_losses()
            if torch.isnan(loss) or torch.isinf(loss): raise EOFError("Invalid loss value")
            loss.backward()
            stats["loss"] = float(loss.detach())
            stats["loss_res"] = float(loss_res.detach())
            stats["loss_ic"] = float(loss_ic.detach())
            stats["loss_bc"] = float(loss_bc.detach())
            return loss

        optimizer.step(closure)

        max_rel_update = 0.0
        for p_old, p_new in zip(params_before, model.parameters()):
            delta = (p_new.detach() - p_old)
            rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
            max_rel_update = max(max_rel_update, rel)

        if track_tiny_update:
            if max_rel_update < 1e-8:
                fp32_tiny_update_count += 1
            else:
                fp32_tiny_update_count = 0
        else:
            fp32_tiny_update_count = 0

        if do_diagnostic:
            curvature_proxy = get_quasi_newton_curvature_proxy()
            if curvature_proxy is not None and np.isfinite(curvature_proxy) and curvature_proxy > 0.0:
                cond_history.append(curvature_proxy)
                log_proxy = float(np.log10(curvature_proxy + 1e-12))
                if smoothed_log_proxy is None:
                    smoothed_log_proxy = prev_smoothed_log_proxy = log_proxy
                    proxy_slope = 0.0
                else:
                    prev_smoothed_log_proxy = smoothed_log_proxy
                    smoothed_log_proxy = EMA_BETA * smoothed_log_proxy + (1.0 - EMA_BETA) * log_proxy
                    proxy_slope = smoothed_log_proxy - prev_smoothed_log_proxy

        can_switch = (step - last_switch_step) >= MIN_DWELL_STEPS

        if benchmark is False:
            has_full_proxy_context = (
                    len(cond_history) >= 2 and smoothed_log_proxy is not None
                    and np.isfinite(smoothed_log_proxy) and np.isfinite(proxy_slope)
            )
            trigger_to_fp32 = (
                    do_diagnostic and precision_state == "fp64" and step >= MIN_SWITCH_STEP
                    and can_switch and has_full_proxy_context
                    and smoothed_log_proxy < LOG_PROXY_LOW and abs(proxy_slope) < LOG_SLOPE_FLAT
                    and fp32_tiny_update_count < STUCK_PATIENCE
            )
            trigger_to_fp64 = (
                    precision_state == "fp32" and (
                    fp32_tiny_update_count >= STUCK_PATIENCE or (
                    do_diagnostic and np.isfinite(curvature_proxy if curvature_proxy is not None else 1.0)
                    and has_full_proxy_context and (
                                (smoothed_log_proxy > LOG_PROXY_HIGH) or (proxy_slope > LOG_SLOPE_UP))
            )
            )
            )
            if trigger_to_fp32:
                precision_state, current_dtype, switch_event = "fp32", torch.float32, 1
            elif trigger_to_fp64:
                precision_state, current_dtype, switch_event = "fp64", torch.float64, 1

        elif benchmark == "switching":
            if not has_switched and precision_state == "fp32":
                trigger_to_fp64 = (fp32_tiny_update_count >= STUCK_PATIENCE)
                if trigger_to_fp64:
                    precision_state, current_dtype, switch_event, has_switched = "fp64", torch.float64, 1, True

        if switch_event == 1:
            last_switch_step = step
            print(
                f"\nStep {step}: SWITCH → {precision_state} (log_proxy={smoothed_log_proxy if smoothed_log_proxy is not None else float('nan'):.4f}, slope={proxy_slope:.4e}, tiny={fp32_tiny_update_count})\n")
            wandb.log({"switch_event_triggered": 1, "target_precision_fp64": 1 if precision_state == "fp64" else 0})

            model = model.to(dtype=current_dtype)
            old_optimizer = optimizer
            optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower = \
                rebuild_data_and_optimizer(dtype=current_dtype, model=model, old_optimizer=old_optimizer)

            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            with torch.no_grad():
                x_eval = x_res[:, -1, :] if x_res.dim() == 3 else x_res
                t_eval = t_res[:, -1, :] if t_res.dim() == 3 else t_res
                u_exact = torch.sin(x_eval - beta_used * t_eval)
                u_pred = model(x_res, t_res)
                if u_pred.dim() == 3: u_pred = u_pred[:, -1, :]
                rRMSE = torch.sqrt(torch.sum((u_exact - u_pred) ** 2) / torch.sum(u_exact ** 2)).item()
                rMAE = (torch.sum(torch.abs(u_exact - u_pred)) / torch.sum(torch.abs(u_exact))).item()

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state} | "
                f"max_rel_update {max_rel_update:.3e}"
            )

            wandb.log({
                "loss": stats["loss"], "loss_res": stats["loss_res"], "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "curvature_proxy": curvature_proxy if curvature_proxy is not None else 0.0,
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope, "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count, "max_rel_update": max_rel_update,
                "rRMSE": rRMSE, "rMAE": rMAE,
            })

        if slq_enabled and (step % slq_interval == 0):
            print(
                f"Computing full-model Hessian spectral density by SLQ/HVP at step {step} "
                f"(probes={slq_num_probes}, lanczos={slq_num_lanczos}, dtype={current_dtype}) ..."
            )
            optimizer.zero_grad(set_to_none=True)
            slq_stats = estimate_hessian_spectral_density_slq(
                model=model,
                loss_fn=lambda: compute_losses()[0],
                num_probes=slq_num_probes,
                num_lanczos=slq_num_lanczos,
                num_grid=slq_num_grid,
                sigma=slq_sigma,
                reorthogonalize=True,
            )
            slq_stats.update({
                "step": step,
                "strategy": strategy,
                "precision_state": precision_state,
                "current_dtype": str(current_dtype),
                "loss": stats.get("loss", np.nan),
                "loss_res": stats.get("loss_res", np.nan),
                "loss_ic": stats.get("loss_ic", np.nan),
                "loss_bc": stats.get("loss_bc", np.nan),
                "rRMSE": rRMSE,
                "rMAE": rMAE,
                "curvature_proxy": curvature_proxy,
                "smoothed_log_proxy": smoothed_log_proxy,
                "proxy_slope": proxy_slope,
                "tiny_updates_count": fp32_tiny_update_count,
                "max_rel_update": max_rel_update,
            })
            slq_history.append(slq_stats)

            wandb.log({
                "slq_hessian_log10_condition_number": slq_stats["log10_condition_number"],
                "slq_hessian_condition_number": slq_stats["condition_number"],
                "slq_hessian_lambda_min_est": slq_stats["lambda_min_est"],
                "slq_hessian_lambda_max_est": slq_stats["lambda_max_est"],
                "slq_hessian_lambda_abs_min_est": slq_stats["lambda_abs_min_est"],
                "slq_hessian_lambda_abs_max_est": slq_stats["lambda_abs_max_est"],
                "slq_hessian_n_params": slq_stats["n_params"],
                "slq_hessian_num_probes": slq_stats["num_probes"],
                "slq_hessian_num_lanczos": slq_stats["num_lanczos"],
            })

        if stats["loss"] < best_loss - 1e-7:
            best_loss = stats["loss"]
            stagnant_steps = 0
        elif step > 100:
            stagnant_steps += 1

        if stagnant_steps > 50:
            print(f"Converged at step {step}: Loss hasn't improved by 1e-7 for more than 50 epochs.")
            break

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")
    wandb.log({"training_time_seconds": duration})

    if len(slq_history) > 0:
        slq_path = f"{run_name}_{strategy}_full_hessian_slq_history.pt"
        torch.save(slq_history, slq_path)
        wandb.save(slq_path)

        cond_plot_path = plot_slq_hessian_condition_history(slq_history, run_name, strategy)
        if cond_plot_path is not None:
            wandb.log({"full_hessian_slq_condition_history": wandb.Image(cond_plot_path)})
            wandb.save(cond_plot_path)

        density_plot_path = plot_slq_hessian_spectral_density(slq_history, run_name, strategy)
        if density_plot_path is not None:
            wandb.log({"full_hessian_slq_spectral_density": wandb.Image(density_plot_path)})
            wandb.save(density_plot_path)

    plot_convection_results(model, beta_used, run_name, strategy, model_name)
    wandb.finish()
    return model
def main():
    import os

    for seed in [0,1,2,3,4]:
        train_dynamic_precision_fixed_beta_50_curvature(
            seed=seed,
            benchmark="fp32",
            slq_enabled=True,
            slq_interval=10,
            slq_num_probes=1,  # PyHessian default n_v
            slq_num_lanczos=100,  # PyHessian default iter
            slq_num_grid=10000,  # PyHessian density default num_bins
        )


if __name__ == "__main__":
    main()