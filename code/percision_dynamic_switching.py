"""
We follow the same model structure as paper 'FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks'
, the model class and data generation are derive from https://github.com/miniHuiHui/PINN_FP64/ for comparison
"""
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
sys.path.append(str(Path("../PINN_FP64-main/"))) #this is the code repository for paper 'FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks'

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


def get_reaction_training_data(x_num=101, t_num=101):
    """
    Generates training data for Reaction-Diffusion:
    u_t = D * u_xx + rho * u * (1 - u)
    Domain: x in [0, 1], t in [0, 1]
    IC: Gaussian pulse
    BC: Dirichlet (u=0 at boundaries)
    """
    res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], x_num, t_num)
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

def get_wave_training_data(x_num=101, t_num=101,model_name='PINN'):
    """
    Generates training data for the Wave equation.
    Domain: x in [0, 1], t in [0, 1]
    """
    res, b_left, b_right, b_upper, b_lower = get_data([0, 1], [0, 1], x_num, t_num)
    if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
        res, b_left, b_right, b_upper, b_lower = preprocess_data_for_model(
            model_name, res, b_left, b_right, b_upper, b_lower
        )
    return res, b_left, b_right, b_upper, b_lower


def get_wave_exact_solution(c=2.0, nx=101, nt=101):
    """
    Provides the exact solution for the 1D wave equation problem.
    u_tt = c^2 * u_xx
    Domain: x in [0, 1], t in [0, 1]
    BC: u(0,t) = u(1,t) = 0
    IC: u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x), u_t(x,0) = 0
    """
    pi = np.pi
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    T_grid, X_grid = np.meshgrid(t, x, indexing="ij")

    u_exact = (np.cos(c * pi * T_grid) * np.sin(pi * X_grid) +
               0.5 * np.cos(c * 3 * pi * T_grid) * np.sin(3 * pi * X_grid))

    return x, t, u_exact


def get_allen_cahn_training_data(x_num=101, t_num=101):
    """
    Generates training data for the Allen-Cahn equation.
    Domain: x in [-1, 1], t in [0, 1]
    """
    res, b_left, b_right, b_upper, b_lower = get_data([-1, 1], [0, 1], x_num, t_num)

    return res, b_left, b_right, b_upper, b_lower


import scipy.io


def get_allen_cahn_exact_solution():
    """
    Loads high-fidelity reference solution for Allen-Cahn equation
    from pre-computed 'allen_cahn.mat' file.
    """
    data = scipy.io.loadmat("allen_cahn.mat")

    t = data["t"].flatten()
    x = data["x"].flatten()
    u_exact = np.real(data["usol"])

    return x, t, u_exact


def irradiance(t, amplitude=1.0, phi=0.75):
    """Normalized sinusoidal driver on t in [0, 1].

    Implements amplitude * sin(2*pi*(t + phi)),
    where phi is a phase shift in normalized time units.
    """
    val = 4 * np.pi * (t + phi)

    if torch.is_tensor(t):
        return amplitude * torch.sin(val)
    else:
        return amplitude * np.sin(val)

def get_irradiance_ode_training_data(n_amp=101, n_time=101):
    """
    Generates training data for Irradiance ODE parameterized by amplitude.
    Domain: amplitude in [0.1, 1.1], t in [0, 1]
    IC: t=0, u=u0 for all amplitudes.
    """
    res, b_left, b_right, b_upper, b_lower = get_data([-1, 1], [0.0, 1.0], n_amp, n_time)
    return res, b_left, b_right, b_upper, b_lower

def solve_irradiance_ode_family(r=3.0, K=10.0, u0=0.05, n_time=101, n_amplitude=101):
    """
    Solve irradiance ODE over a family of amplitude parameters.
    Returns 2D grid: (amplitude, t) -> u(amplitude, t)
    """
    t_vals = np.linspace(0, 1, n_time)
    amplitude_vals = np.linspace(-1, 1, n_amplitude)

    u_exact_2d = np.zeros((n_amplitude, n_time))

    for i, amp in enumerate(amplitude_vals):
        def rhs(t, u):
            s = irradiance(t, amplitude=amp)
            return (r + s) * u * (1 - u / K)

        sol = solve_ivp(rhs, [0, 1], [u0], t_eval=t_vals, method="RK45")
        u_exact_2d[i, :] = sol.y[0]

    return t_vals, amplitude_vals, u_exact_2d

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


def train_dynamic_precision_fixed_rho_curvature(
        seed=1234,
        RHO_MAX=5,
        MAX_STEPS=50000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = 'PINN',
):
    set_seed(seed)
    run_name = f"reaction_1d_dynamic_precision_curvature_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Curvature Run (1D Reaction): {run_name}")
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
    else:
        strategy = "dynamic_precision_lbfgs"

    wandb.init(
        project="pinn_reaction_dynamic_precision_3layerMLP".format(model_name),
        name=run_name,
        config={"strategy": strategy, "seed": seed, "rho_max": RHO_MAX, "min_switch_step": MIN_SWITCH_STEP,
                "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW, "trigger_patience": TRIGGER_PATIENCE,
                "stuck_patience": STUCK_PATIENCE,"LOG_PROXY_LOW": LOG_PROXY_LOW},
        reinit=True,
    )

    rho_used = RHO_MAX

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

    model = init_model(model_name, hidden_dim=648, num_layer=3, dtype=current_dtype)
    print(model)

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_reaction_training_data(x_num=101, t_num=101)
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = preprocess_data_for_model(
        model_name, res_np, b_left_np, b_right_np, b_upper_np, b_lower_np
    )

    data_cache = {}
    for dt in [torch.float32, torch.float64]:
        data_cache[dt] = {
            "res": torch.tensor(res_np, dtype=dt, device=device),
            "b_left": torch.tensor(b_left_np, dtype=dt, device=device),
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
        x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
        x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

        return optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_lower, t_lower

    optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_lower, t_lower = \
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
            u_t = u_t_scaled / S
        else:
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

        loss_res = torch.mean((u_t - rho_used * pred_res * (1.0 - pred_res)) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean(
            (pred_left[..., 0] - torch.exp(-((x_left[..., 0] - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2))) ** 2)
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

    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

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

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = (p_new.detach() - p_old)
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
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
            optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_lower, t_lower = \
                rebuild_data_and_optimizer(dtype=current_dtype, model=model, old_optimizer=old_optimizer)

            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            with torch.no_grad():
                x_eval = x_res[:, -1, :] if x_res.dim() == 3 else x_res
                t_eval = t_res[:, -1, :] if t_res.dim() == 3 else t_res
                h_val = torch.exp(-((x_eval - torch.pi) ** 2) / (2 * (torch.pi / 4) ** 2))
                u_exact = h_val * torch.exp(rho_used * t_eval) / (h_val * torch.exp(rho_used * t_eval) + 1.0 - h_val)
                u_pred = model(x_res, t_res)
                if u_pred.dim() == 3: u_pred = u_pred[:, -1, :]
                rRMSE = torch.sqrt(torch.sum((u_exact - u_pred) ** 2) / torch.sum(u_exact ** 2)).item()
                rMAE = (torch.sum(torch.abs(u_exact - u_pred)) / torch.sum(torch.abs(u_exact))).item()

            # model.train()
            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"], "loss_res": stats["loss_res"], "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"], "rho": rho_used,
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope, "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count, "rRMSE": rRMSE, "rMAE": rMAE,
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
    plot_1d_reaction_results(model, rho_used, run_name, "dynamic_precision", model_name)
    wandb.finish()
    return model



def train_dynamic_precision_fixed_beta_50_curvature(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=5000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = 'PINN',
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
            project="pinn_convection_dynamic_precision_plotMLP".format(model_name),
            name=run_name,
            config={"strategy": strategy, "seed": seed, "beta": BETA_MAX, "min_switch_step": 'NA',
                    "min_dwell_steps": 'NA', "cond_window": 'NA',
                    "trigger_patience": 'NA',
                    "stuck_patience": 'NA'},
            reinit=True
        )
    else:
        strategy = "dynamic"

        wandb.init(
            project="pinn_convection_dynamic_precision_plotMLP".format(model_name),
            name=run_name,
            config={"strategy": strategy, "seed": seed, "beta": BETA_MAX, "min_switch_step": MIN_SWITCH_STEP,
                    "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW, "trigger_patience": TRIGGER_PATIENCE,
                    "stuck_patience": STUCK_PATIENCE,"LOG_PROXY_LOW": LOG_PROXY_LOW},
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

    model = init_model(model_name, hidden_dim=512, num_layer=4,dtype=current_dtype)
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
        # if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
        #     loss_seq = torch.mean((pred_res[101:10201,0:num_step-seq_diff,:]-pred_res[0:10100,seq_diff:num_step,:]) ** 2)
        #     loss = loss_res + 10 * loss_bc + loss_ic + 1000 * loss_seq
        # else:
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

    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

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

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = (p_new.detach() - p_old)
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
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
            # model.train()

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"], "loss_res": stats["loss_res"], "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope, "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count, "rRMSE": rRMSE, "rMAE": rMAE,
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
    plot_convection_results(model, beta_used, run_name, strategy, model_name)
    wandb.finish()
    return model




def train_dynamic_precision_allen_cahn_curvature(
        seed=1234,
        MAX_STEPS=50000,
        benchmark: bool | str = False,
        model_name: str = 'PINN',
):
    set_seed(seed)
    run_name = f"allen_cahn_dynamic_precision_curvature_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Curvature Run (1D Allen-Cahn): {run_name}")
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
    else:
        strategy = "dynamic"

    wandb.init(
        project="pinn_allen_cahn_dynamic_precision_3layer".format(model_name),
        name=run_name,
        config={"strategy": strategy, "seed": seed, "min_switch_step": MIN_SWITCH_STEP,
                "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW, "trigger_patience": TRIGGER_PATIENCE,
                "stuck_patience": STUCK_PATIENCE,"LOG_PROXY_LOW": LOG_PROXY_LOW},
        reinit=True,
    )

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

    model = init_model(model_name, hidden_dim=64, num_layer=3, dtype=current_dtype)
    print(model)

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_allen_cahn_training_data(x_num=101, t_num=101)
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

        ones = torch.ones_like(pred_res)
        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]

        bnd_upper_dx = \
        torch.autograd.grad(pred_upper, x_upper, grad_outputs=torch.ones_like(pred_upper), retain_graph=True,
                            create_graph=True)[0]
        bnd_lower_dx = \
        torch.autograd.grad(pred_lower, x_lower, grad_outputs=torch.ones_like(pred_lower), retain_graph=True,
                            create_graph=True)[0]

        loss_res = torch.mean((u_t - 0.0001 * u_xx - 5 * pred_res + 5 * pred_res ** 3) ** 2)
        loss_ic = torch.mean((pred_left[..., 0] - (x_left[..., 0] ** 2) * torch.cos(math.pi * x_left[..., 0])) ** 2)
        loss_bc_1 = torch.mean((pred_upper - pred_lower) ** 2)
        loss_bc_2 = torch.mean((bnd_upper_dx - bnd_lower_dx) ** 2)
        loss_bc = loss_bc_1 + loss_bc_2

        loss = 10 * loss_res + 100 * loss_ic + loss_bc
        return loss, loss_res, loss_ic, loss_bc

    x_ev, t_ev, u_eval_exact = get_allen_cahn_exact_solution()
    T_ev, X_ev = np.meshgrid(t_ev, x_ev, indexing="ij")
    X_ev_flat = X_ev.flatten()[:, None]
    T_ev_flat = T_ev.flatten()[:, None]

    cond_history = deque(maxlen=COND_WINDOW)
    stats = {}
    fp32_tiny_update_count = 0
    switch_event = 0
    smoothed_log_proxy = prev_smoothed_log_proxy = None
    proxy_slope = 0.0
    curvature_proxy = None
    best_loss = float('inf')
    stagnant_steps = 0

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

    start_time = time.time()
    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

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

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = (p_new.detach() - p_old)
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
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
            # model.eval()
            with torch.no_grad():
                ev_data = np.concatenate((X_ev_flat, T_ev_flat), axis=-1)
                if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
                    ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
                    x_t = torch.tensor(ev_data[..., 0:1], dtype=current_dtype, device=device)
                    t_t = torch.tensor(ev_data[..., 1:2], dtype=current_dtype, device=device)
                else:
                    x_t = torch.tensor(X_ev_flat, dtype=current_dtype, device=device)
                    t_t = torch.tensor(T_ev_flat, dtype=current_dtype, device=device)

                u_pred_ev = model(x_t, t_t)
                if u_pred_ev.dim() == 3: u_pred_ev = u_pred_ev[:, -1, :]
                u_pred_ev = u_pred_ev.cpu().numpy().reshape(T_ev.shape)

            rRMSE = float(np.sqrt(np.sum((u_eval_exact - u_pred_ev) ** 2) / np.sum(u_eval_exact ** 2)))
            rMAE = float(np.sum(np.abs(u_eval_exact - u_pred_ev)) / np.sum(np.abs(u_eval_exact)))
            # model.train()

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"], "loss_res": stats["loss_res"], "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope, "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count, "rRMSE": rRMSE, "rMAE": rMAE,
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
    plot_allen_cahn_results(model, run_name, "dynamic_precision", model_name)
    wandb.finish()
    return model


def train_dynamic_precision_wave_curvature(
        seed=1234,
        MAX_STEPS=50000,
        benchmark: bool | str = False,
        model_name: str = 'PINN',
):
    set_seed(seed)

    C_SQUARED = 4.0
    C = 2.0
    pi = np.pi
    run_name = f"wave_dynamic_precision_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Run (1D Wave Curvature): {run_name}")
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
        strategy = benchmark
    else:
        strategy = "dynamic"

    wandb.init(
        project="pinn_wave_dynamic_precision_3layerMLP".format(model_name),
        name=run_name,
        config={"strategy": strategy, "seed": seed, "c_squared": C_SQUARED, "min_switch_step": MIN_SWITCH_STEP,
                "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW, "trigger_patience": TRIGGER_PATIENCE,
                "stuck_patience": STUCK_PATIENCE,"LOG_PROXY_LOW": LOG_PROXY_LOW},
        reinit=True,
    )

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

    model = init_model(model_name, hidden_dim=648, num_layer=3, dtype=current_dtype)
    print(model)

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_wave_training_data(x_num=101, t_num=101,
                                                                                   model_name=model_name)

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

        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        if old_optimizer is not None and isinstance(old_optimizer, torch.optim.LBFGS):
            try:
                old_state_dict = old_optimizer.state_dict()
                old_state_dict["state"] = _cast_obj(old_state_dict["state"], dtype, device)
                optimizer.load_state_dict(old_state_dict)
            except Exception as e:
                print(f"Could not transfer optimizer state: {e}. Re-initializing optimizer from scratch.")

        x_res, t_res = res[..., 0:1], res[..., 1:2]
        x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
        x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
        x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
        x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

        return optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower

    optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower = \
        rebuild_data_and_optimizer(dtype=current_dtype, model=model)

    def compute_losses():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        ones = torch.ones_like(pred_res)
        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=ones, retain_graph=True, create_graph=True)[0]

        loss_res = torch.mean((u_tt - C_SQUARED * u_xx) ** 2)
        loss_bc = torch.mean(pred_lower ** 2) + torch.mean(pred_upper ** 2)

        ui_t = torch.autograd.grad(pred_left, t_left, grad_outputs=torch.ones_like(pred_left), retain_graph=True,
                                   create_graph=True)[0]
        loss_ic_1 = torch.mean(
            (pred_left[..., 0] - torch.sin(pi * x_left[..., 0]) - 0.5 * torch.sin(3 * pi * x_left[..., 0])) ** 2)
        loss_ic_2 = torch.mean((ui_t) ** 2)

        loss_ic = loss_ic_1 + loss_ic_2
        loss = loss_res + loss_ic + loss_bc
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

    nx_ev, nt_ev = 101, 101
    x_ev, t_ev, u_eval_exact = get_wave_exact_solution(C, nx_ev, nt_ev)
    T_ev, X_ev = np.meshgrid(t_ev, x_ev, indexing="ij")
    X_ev_flat = X_ev.flatten()[:, None]
    T_ev_flat = T_ev.flatten()[:, None]

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

    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

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

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = (p_new.detach() - p_old)
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
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
            has_full_proxy_context = (len(cond_history) >= 2 and smoothed_log_proxy is not None and np.isfinite(
                smoothed_log_proxy) and np.isfinite(proxy_slope))
            trigger_to_fp32 = (
                        do_diagnostic and precision_state == "fp64" and step >= MIN_SWITCH_STEP and can_switch and has_full_proxy_context and smoothed_log_proxy < LOG_PROXY_LOW and abs(
                    proxy_slope) < LOG_SLOPE_FLAT and fp32_tiny_update_count < STUCK_PATIENCE)
            trigger_to_fp64 = (precision_state == "fp32" and (fp32_tiny_update_count >= STUCK_PATIENCE or (
                        do_diagnostic and np.isfinite(
                    curvature_proxy if curvature_proxy is not None else 1.0) and has_full_proxy_context and (
                                    (smoothed_log_proxy > LOG_PROXY_HIGH) or (proxy_slope > LOG_SLOPE_UP)))))
            if trigger_to_fp32:
                precision_state, current_dtype, switch_event = "fp32", torch.float32, 1
            elif trigger_to_fp64:
                precision_state, current_dtype, switch_event = "fp64", torch.float64, 1
        elif benchmark == "switching":
            if not has_switched and precision_state == "fp32":
                trigger_to_fp64 = (fp32_tiny_update_count >= STUCK_PATIENCE)
                if trigger_to_fp64: precision_state, current_dtype, switch_event, has_switched = "fp64", torch.float64, 1, True

        if switch_event == 1:
            last_switch_step = step
            print(
                f"\nStep {step}: SWITCH → {precision_state} (log_proxy={smoothed_log_proxy if smoothed_log_proxy is not None else float('nan'):.4f}, slope={proxy_slope:.4e}, tiny={fp32_tiny_update_count})\n")
            model = model.to(dtype=current_dtype)
            old_optimizer = optimizer
            optimizer, x_res, t_res, x_left, t_left, x_right, t_right, x_upper, t_upper, x_lower, t_lower = rebuild_data_and_optimizer(
                dtype=current_dtype, model=model, old_optimizer=old_optimizer)
            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            # model.eval()
            with torch.no_grad():
                ev_data = np.concatenate((X_ev_flat, T_ev_flat), axis=-1)
                if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
                    ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
                    x_t = torch.tensor(ev_data[..., 0:1], dtype=current_dtype, device=device)
                    t_t = torch.tensor(ev_data[..., 1:2], dtype=current_dtype, device=device)
                else:
                    x_t = torch.tensor(X_ev_flat, dtype=current_dtype, device=device)
                    t_t = torch.tensor(T_ev_flat, dtype=current_dtype, device=device)

                u_pred_ev = model(x_t, t_t)
                if u_pred_ev.dim() == 3: u_pred_ev = u_pred_ev[:, -1, :]
                u_pred_ev = u_pred_ev.cpu().numpy().reshape(T_ev.shape)

            rRMSE = float(np.sqrt(np.sum((u_eval_exact - u_pred_ev) ** 2) / np.sum(u_eval_exact ** 2)))
            rMAE = float(np.sum(np.abs(u_eval_exact - u_pred_ev)) / np.sum(np.abs(u_eval_exact)))
            # model.train()

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"], "loss_res": stats["loss_res"], "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope,
                "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count,
                "rRMSE": rRMSE, "rMAE": rMAE,
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
    plot_wave_results(model, C, run_name, strategy, model_name)
    wandb.finish()
    return model


def train_dynamic_precision_irradiance(
        seed=0,
        MAX_STEPS=20000,
        model_name='PINN',
        benchmark: bool | str = False,
):
    set_seed(seed)

    run_name = f"irradiance_ode_dynamic_precision_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Irradiance Run (2D family): {run_name}")
    print("=" * 60)

    MIN_SWITCH_STEP = 0
    MIN_DWELL_STEPS = 10
    COND_WINDOW = 10
    TRIGGER_PATIENCE = 10
    STUCK_PATIENCE = 3
    DIAGNOSTIC_INTERVAL = 10

    CURV_EPS = 1e-30
    LOG_PROXY_LOW = 2.5
    LOG_PROXY_HIGH = 2.5
    LOG_SLOPE_FLAT = 0.02
    LOG_SLOPE_UP = 0.03
    EMA_BETA = 0.9

    r = 2.0
    K = 1.0
    u0 = 0.05

    if benchmark is not False:
        strategy = benchmark
    else:
        strategy = "dynamic"

    wandb.init(
        project="pinn_irradiance_dynamic_precision",
        name=run_name,
        config={
            "strategy": strategy, "seed": seed, "min_switch_step": MIN_SWITCH_STEP,
            "min_dwell_steps": MIN_DWELL_STEPS, "cond_window": COND_WINDOW,
            "trigger_patience": TRIGGER_PATIENCE, "stuck_patience": STUCK_PATIENCE,
            "LOG_PROXY_LOW": LOG_PROXY_LOW, "LOG_PROXY_HIGH": LOG_PROXY_HIGH,
            "LOG_SLOPE_FLAT": LOG_SLOPE_FLAT, "LOG_SLOPE_UP": LOG_SLOPE_UP,
            "EMA_BETA": EMA_BETA, "r": r, "K": K,
        },
        reinit=True,
    )

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

    # Switched out irradiance_PINN for standard PDE initialization taking 2 inputs
    model = init_model(model_name, hidden_dim=512, num_layer=4, dtype=current_dtype)

    # Build and cache training tensors for both precisions
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_irradiance_ode_training_data(5, 10100)
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
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        if old_optimizer is not None and isinstance(old_optimizer, torch.optim.LBFGS):
            try:
                old_state_dict = old_optimizer.state_dict()
                old_state_dict["state"] = _cast_obj(old_state_dict["state"], dtype, device)
                optimizer.load_state_dict(old_state_dict)
            except Exception as e:
                print(f"Could not transfer optimizer state: {e}. Re-initializing optimizer from scratch.")

        amp_res, t_res = res[..., 0:1], res[..., 1:2]
        amp_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
        amp_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
        amp_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
        amp_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

        return optimizer, amp_res, t_res, amp_left, t_left, amp_right, t_right, amp_upper, t_upper, amp_lower, t_lower

    optimizer, amp_res, t_res, amp_left, t_left, amp_right, t_right, amp_upper, t_upper, amp_lower, t_lower = rebuild_data_and_optimizer(
        dtype=current_dtype, model=model)

    stats = {}

    def compute_losses():
        u = model(amp_res, t_res)

        u_t = torch.autograd.grad(
            u, t_res,
            grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]

        s = irradiance(t_res, amplitude=amp_res)

        residual = u_t - (r + s) * u * (1.0 - u / K)
        loss_res = torch.mean(residual ** 2)

        u_ic_pred = model(amp_left, t_left)
        loss_ic = torch.mean((u_ic_pred - u0) ** 2)

        loss_bc = torch.tensor(0.0, device=u.device, dtype=u.dtype)

        loss = loss_res + loss_ic

        return loss, loss_res, loss_ic, loss_bc

    def get_quasi_newton_curvature_proxy():
        if not optimizer.state:
            return None
        state = optimizer.state[next(iter(optimizer.state))]
        if "old_dirs" not in state or "old_stps" not in state:
            return None

        directional_curvatures = []
        for y, s in zip(state["old_dirs"], state["old_stps"]):
            y, s = y.flatten(), s.flatten()
            ys = torch.dot(s, y).item()
            ss = torch.dot(s, s).item()
            if np.isfinite(ys) and np.isfinite(ss) and ss > CURV_EPS and ys > CURV_EPS:
                directional_curvatures.append(ys / ss)

        if len(directional_curvatures) < 2: return None

        directional_curvatures = np.asarray(directional_curvatures, dtype=np.float64)
        k_min = max(np.min(directional_curvatures), 1e-12)
        k_max = max(np.max(directional_curvatures), k_min)
        return float(k_max / k_min)

    t_ev, amp_ev, u_eval_exact = solve_irradiance_ode_family(r=r, K=K, u0=u0, n_time=1010, n_amplitude=10)
    A_grid, T_grid = np.meshgrid(amp_ev, t_ev, indexing='ij')
    A_ev_flat = A_grid.flatten()[:, None]
    T_ev_flat = T_grid.flatten()[:, None]

    cond_history = deque(maxlen=COND_WINDOW)
    fp32_tiny_update_count = 0
    switch_event = 0
    smoothed_log_proxy = prev_smoothed_log_proxy = None
    proxy_slope = 0.0
    curvature_proxy = None
    best_loss = float('inf')
    stagnant_steps = 0
    start_time = time.time()
    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = (step % DIAGNOSTIC_INTERVAL == 0)
        track_tiny_update = (precision_state == "fp32")

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, loss_res, loss_ic, loss_bc = compute_losses()
            if torch.isnan(loss) or torch.isinf(loss):
                raise EOFError("Invalid loss value")
            loss.backward()

            stats["loss"] = float(loss.detach())
            stats["loss_res"] = float(loss_res.detach())
            stats["loss_ic"] = float(loss_ic.detach())
            stats["loss_bc"] = float(loss_bc.detach())
            return loss

        optimizer.step(closure)

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = (p_new.detach() - p_old)
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
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
                    len(cond_history) >= 2 and smoothed_log_proxy is not None and
                    np.isfinite(smoothed_log_proxy) and np.isfinite(proxy_slope)
            )

            trigger_to_fp32 = (
                    do_diagnostic and precision_state == "fp64" and step >= MIN_SWITCH_STEP and
                    can_switch and has_full_proxy_context and smoothed_log_proxy < LOG_PROXY_LOW and
                    abs(proxy_slope) < LOG_SLOPE_FLAT and fp32_tiny_update_count < STUCK_PATIENCE
            )

            trigger_to_fp64 = (
                    precision_state == "fp32" and (
                    fp32_tiny_update_count >= STUCK_PATIENCE or (
                    do_diagnostic and np.isfinite(curvature_proxy if curvature_proxy is not None else 1.0) and
                    has_full_proxy_context and (
                            (smoothed_log_proxy > LOG_PROXY_HIGH) or (proxy_slope > LOG_SLOPE_UP)
                    )
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
                f"\nStep {step}: SWITCH -> {precision_state} "
                f"(log_proxy={smoothed_log_proxy if smoothed_log_proxy is not None else float('nan'):.4f}, "
                f"slope={proxy_slope:.4e}, tiny={fp32_tiny_update_count})\n"
            )
            wandb.log({"switch_event_triggered": 1, "target_precision_fp64": 1 if precision_state == "fp64" else 0})

            model = model.to(dtype=current_dtype)
            old_optimizer = optimizer
            optimizer, amp_res, t_res, amp_left, t_left, amp_right, t_right, amp_upper, t_upper, amp_lower, t_lower = rebuild_data_and_optimizer(
                dtype=current_dtype, model=model, old_optimizer=old_optimizer
            )

            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            with torch.no_grad():
                ev_data = np.concatenate((A_ev_flat, T_ev_flat), axis=-1)
                if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
                    ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
                    a_t = torch.tensor(ev_data[..., 0:1], dtype=current_dtype, device=device)
                    t_t = torch.tensor(ev_data[..., 1:2], dtype=current_dtype, device=device)
                else:
                    a_t = torch.tensor(A_ev_flat, dtype=current_dtype, device=device)
                    t_t = torch.tensor(T_ev_flat, dtype=current_dtype, device=device)

                u_pred_ev = model(a_t, t_t)
                if u_pred_ev.dim() == 3: u_pred_ev = u_pred_ev[:, -1, :]
                u_pred_ev = u_pred_ev.cpu().numpy().reshape(u_eval_exact.shape)

            rRMSE = float(np.sqrt(np.sum((u_eval_exact - u_pred_ev) ** 2) / np.sum(u_eval_exact ** 2)))
            rMAE = float(np.sum(np.abs(u_eval_exact - u_pred_ev)) / np.sum(np.abs(u_eval_exact)))

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"log_proxy {smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "smoothed_log_proxy": smoothed_log_proxy if smoothed_log_proxy is not None else 0.0,
                "proxy_slope": proxy_slope,
                "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count,
                "rRMSE": rRMSE, "rMAE": rMAE,
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

    plot_irradiance_results(model, run_name, strategy, model_name, r=r, K=K, u0=u0)
    wandb.finish()
    return model

def plot_1d_reaction_results(model, rho, run_name, strategy, model_name='PINN'):
    print("\nGenerating final evaluation metrics and plots...")
    res_test, b_left, b_right, b_upper, b_lower = get_reaction_training_data(x_num=101, t_num=101)
    res_test, _, _, _, _ = preprocess_data_for_model(model_name, res_test, b_left, b_right, b_upper, b_lower)

    dtype = next(model.parameters()).dtype
    res_test_tensor = torch.tensor(res_test, dtype=dtype).to(device)
    x_test, t_test = res_test_tensor[..., 0:1], res_test_tensor[..., 1:2]

    # model.eval()
    with torch.no_grad():
        pred = model(x_test, t_test)
        if pred.dim() == 3: pred = pred[:, -1, :]
        pred = pred[:, 0:1].cpu().numpy().reshape(101, 101)

    def h(x): return np.exp(- (x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))
    def u_ana(x, t): return h(x) * np.exp(rho * t) / (h(x) * np.exp(rho * t) + 1 - h(x))

    x_np = res_test[:, -1, 0] if res_test.ndim == 3 else res_test[:, 0]
    t_np = res_test[:, -1, 1] if res_test.ndim == 3 else res_test[:, 1]
    u = u_ana(x_np, t_np).reshape(101, 101)

    rMAE = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
    rRMSE = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

    print('-' * 40)
    print(f'Relative MAE (rMAE): {rMAE:4f}')
    print(f'Relative L2 Error (rRMSE): {rRMSE:4f}')
    print('-' * 40)

    wandb.run.summary["relative_l1_error"] = rMAE
    wandb.run.summary["rMAE"] = rMAE
    wandb.run.summary["rRMSE"] = rRMSE

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(u, extent=[0, 1, 1, 0], aspect='auto', cmap='viridis')
    axes[0].set_title("Exact u(x,t)")
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred, extent=[0, 1, 1, 0], aspect='auto', cmap='viridis')
    axes[1].set_title("Predicted u(x,t)")
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred - u, extent=[0, 1, 1, 0], aspect='auto', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axes[2].set_title(f"Absolute Error (rRMSE: {rRMSE:.4f})")
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    filename = f"1d_reaction_{run_name}_{strategy}.png"
    plt.savefig(filename, bbox_inches='tight')
    wandb.log({"Result Plots": wandb.Image(filename)})
    plt.close()
    print(f"Plots saved successfully to {filename}.")



def plot_convection_results(model, beta, run_name, strategy, model_name='PINN'):
    print("Generating results plot...")
    nx, nt = 200, 100
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


def plot_wave_results(model, c, run_name, strategy, model_name='PINN'):
    print("Generating Wave equation results plot...")
    nx, nt = 101, 101
    x, t, u_exact = get_wave_exact_solution(c, nx, nt)
    T, X = np.meshgrid(t, x, indexing="ij")

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
    extent = [0, 1, 1, 0]

    im0 = axes[0].imshow(u_exact, extent=extent, aspect='auto', cmap='viridis', origin="upper")
    axes[0].set_title(f"Exact Solution (c={c})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, extent=extent, aspect='auto', cmap='viridis', origin="upper")
    axes[1].set_title(f"Prediction (rRMSE={rRMSE:.4e})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=extent,origin="upper", aspect='auto', cmap='coolwarm', vmin=-0.15, vmax=0.15)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    filename = f"result_wave_{run_name}_{strategy}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    wandb.log({"Result Plot": wandb.Image(filename)})
    plt.close(fig)
    print(f"Plot saved to {filename}")


def plot_allen_cahn_results(model, run_name, strategy, model_name='PINN'):
    print("Generating Allen-Cahn results plot...")
    x, t, u_exact = get_allen_cahn_exact_solution()
    T, X = np.meshgrid(t, x, indexing="ij")

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
    extent = [-1, 1, 1, 0]

    im0 = axes[0].imshow(u_exact, extent=extent, aspect='auto',  cmap='viridis',origin="upper")
    axes[0].set_title("Exact Solution (Numerical)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, extent=extent, aspect='auto', cmap='viridis', origin="upper")
    axes[1].set_title(f"Prediction (rRMSE={rRMSE:.4e})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=extent, aspect='auto', origin="upper",cmap='coolwarm', vmin=-0.15, vmax=0.15, )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    filename = f"result_allen_cahn_{run_name}_{strategy}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    wandb.log({"Result Plot": wandb.Image(filename)})
    plt.close(fig)
    print(f"Plot saved to {filename}")


def plot_irradiance_results(model, run_name, strategy=None, model_name='PINN', r=3.0, K=10.0, u0=0.05):
    """Plot results for the 2D Irradiance family"""
    print("Generating irradiance ODE results plot (2D family sweep)...")
    n_time, n_amp = 101, 5

    # Solve 2D family over amplitude parameter using the provided r, K, u0
    t_vals, amplitude_vals, u_exact_2d = solve_irradiance_ode_family(
        r=r, K=K, u0=u0, n_time=n_time, n_amplitude=n_amp
    )

    dtype = next(model.parameters()).dtype

    # Create evaluation grid matching the other PDEs
    A_grid, T_grid = np.meshgrid(amplitude_vals, t_vals, indexing='ij')
    a_flat = A_grid.flatten()[:, None]
    t_flat = T_grid.flatten()[:, None]

    with torch.no_grad():
        ev_data = np.concatenate((a_flat, t_flat), axis=-1)
        if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
            ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
            a_t = torch.tensor(ev_data[..., 0:1], dtype=dtype, device=device)
            t_t = torch.tensor(ev_data[..., 1:2], dtype=dtype, device=device)
        else:
            a_t = torch.tensor(a_flat, dtype=dtype, device=device)
            t_t = torch.tensor(t_flat, dtype=dtype, device=device)

        u_pred = model(a_t, t_t)
        if u_pred.dim() == 3: u_pred = u_pred[:, -1, :]
        u_pred_2d = u_pred.cpu().numpy().reshape(u_exact_2d.shape)

    error_2d = np.abs(u_exact_2d - u_pred_2d)

    # Compute relative metrics
    valid_mask = np.abs(u_exact_2d) > 1e-10
    if valid_mask.sum() > 0:
        rRMSE = np.sqrt(np.sum((u_exact_2d[valid_mask] - u_pred_2d[valid_mask]) ** 2) /
                        np.sum(u_exact_2d[valid_mask] ** 2))
        rMAE = np.sum(np.abs(u_exact_2d[valid_mask] - u_pred_2d[valid_mask])) / np.sum(np.abs(u_exact_2d[valid_mask]))
    else:
        rRMSE = np.nan
        rMAE = np.nan

    print(f"rRMSE: {rRMSE:.4e}, rMAE: {rMAE:.4e}")

    if wandb.run is not None:
        wandb.run.summary["rRMSE"] = float(rRMSE)
        wandb.run.summary["rMAE"] = float(rMAE)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [0, 1, -1, 1]  # [t_min, t_max, amp_max, amp_min]

    im0 = axes[0].imshow(u_exact_2d, extent=extent, aspect="auto", cmap='viridis', origin="upper")
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Irradiance Amplitude")
    axes[0].set_title("Exact Solution u(amp, t)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred_2d, extent=extent, aspect="auto", cmap='viridis', origin="upper")
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Irradiance Amplitude")
    axes[1].set_title(f"Prediction (rRMSE={rRMSE:.4e})")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error_2d, extent=extent, aspect="auto", cmap='coolwarm', origin="upper", vmin=-0.15, vmax=0.15)
    axes[2].set_xlabel("Time t")
    axes[2].set_ylabel("Irradiance Amplitude")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if strategy is None:
        filename_heatmap = f"irradiance_{run_name}.png"
        filename_lines = f"irradiance_lines_{run_name}.png"
    else:
        filename_heatmap = f"irradiance_{run_name}_{strategy}.png"
        filename_lines = f"irradiance_lines_{run_name}_{strategy}.png"

    plt.savefig(filename_heatmap, dpi=200, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"Result Plot": wandb.Image(filename_heatmap)})
    plt.close(fig)
    print(f"Heatmap plot saved to {filename_heatmap}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    indices_to_plot = np.linspace(0, len(amplitude_vals) - 1, 6, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

    for color, idx in zip(colors, indices_to_plot):
        amp = amplitude_vals[idx]
        ax2.plot(t_vals, u_exact_2d[idx, :], linestyle='--', color=color, alpha=0.5)
        ax2.plot(t_vals, u_pred_2d[idx, :], linestyle='-', color=color, label=f'Amp={amp:.2f}')

    ax2.plot([], [], 'k--', alpha=0.5, label='Exact (Dashed)')

    ax2.set_xlabel("Time t")
    ax2.set_ylabel("u(t)")
    ax2.set_title("u vs t (prediction vs exact) for Selected Amplitudes")
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename_lines, dpi=300, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"Cross Sections Plot": wandb.Image(filename_lines)})
    plt.close(fig2)
    print(f"Cross-sections plot saved to {filename_lines}")


def main():
    import os

    # train_dynamic_precision_allen_cahn_curvature(seed=0, benchmark='fp64')
    # train_dynamic_precision_wave_curvature(seed=5)
    # train_dynamic_precision_allen_cahn_curvature(seed=5)

    for seed in [0, 1, 2, 3, 4]:
        train_dynamic_precision_irradiance(seed=seed,benchmark='fp32')
        train_dynamic_precision_irradiance(seed=seed)
        train_dynamic_precision_irradiance(seed=seed, benchmark='fp64')
    #     train_dynamic_precision_allen_cahn_curvature(seed=seed, benchmark='switching')
    #     train_dynamic_precision_allen_cahn_curvature(seed=seed)
    #     train_dynamic_precision_allen_cahn_curvature(seed=seed, benchmark='fp32')
    #     train_dynamic_precision_allen_cahn_curvature(seed=seed,benchmark='fp64')
        # train_dynamic_precision_allen_cahn_curvature(seed=seed, benchmark="switching")
        # train_dynamic_precision_allen_cahn_curvature(seed=seed, benchmark='fp32', model_name='KAN')
    # for seed in [0]:
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp64',model_name='PINNsFormer')
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed)
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp64')
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp32')
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark="switching")
    # for seed in [0,1,2,3,4]:
        # train_dynamic_precision_fixed_rho_curvature(seed=seed)
        # train_dynamic_precision_fixed_rho_curvature(seed=seed,benchmark='fp64')
        # train_dynamic_precision_fixed_rho_curvature(seed=seed, benchmark='fp32')
        # train_dynamic_precision_fixed_rho_curvature(seed=seed, benchmark="switching")
    #     train_dynamic_precision_fixed_rho_curvature(seed=seed, benchmark='fp32', model_name='KAN')
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed)
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed,benchmark='fp64')
    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp32')
        # train_dynamic_precision_fixed_rho(seed=seed, adam_warmup=False)
        # train_dynamic_precision_fixed_rho(seed=seed,benchmark='fp64',adam_warmup=False)
        #
        # train_dynamic_precision_fixed_rho(seed=seed,benchmark='fp32')

        # train_dynamic_precision_wave_curvature(seed=seed, adam_warmup=False)
        # train_dynamic_precision_wave(seed=seed,adam_warmup=False)
        # train_dynamic_precision_wave(seed=seed,benchmark='fp64',adam_warmup=True)
        # train_dynamic_precision_wave(seed=seed, benchmark='fp64', adam_warmup=False)
        # train_dynamic_precision_wave(seed=seed,benchmark='fp32')
        #
        # train_dynamic_precision_allen_cahn(seed=seed,adam_warmup=True)
        # train_dynamic_precision_allen_cahn(seed=seed, benchmark='fp64',adam_warmup=True)
        # train_dynamic_precision_allen_cahn(seed=seed, benchmark='fp64', adam_warmup=False)
        # train_dynamic_precision_allen_cahn(seed=seed, adam_warmup=False)
        # train_dynamic_precision_allen_cahn(seed=seed,benchmark='fp32')
        # train_dynamic_precision_fixed_beta_50(seed=seed, adam_warmup=False)
    # for seed in [1, 2]:    #
    #
        # train_dynamic_precision_fixed_beta_50(seed=seed, benchmark='fp64', adam_warmup=False)
        # train_dynamic_precision_fixed_beta_50(seed=seed,benchmark='fp32')

    #     train_dynamic_precision_fixed_beta_50_curvature(seed=seed)
    #     train_dynamic_precision_burgers_curvature(seed=seed)
        # train_dynamic_precision_diffusion_reaction(seed=seed)
        # train_dynamic_precision_diffusion_reaction(seed,benchmark='fp64')
        # train_dynamic_precision_diffusion_reaction(seed, benchmark='fp32')
        #
        # train_dynamic_precision_burgers(seed=seed)
        # train_dynamic_precision_burgers(seed=seed,benchmark='fp64')
        # train_dynamic_precision_burgers(seed=seed, benchmark='fp32')

    # for seed in [0,1,2,3,4]:
    #     # train_dynamic_precision_allen_cahn(seed=seed, benchmark="switching", adam_warmup=False)
    #     # train_dynamic_precision_fixed_rho(seed=seed, benchmark="switching", adam_warmup=False)
    #     # train_dynamic_precision_wave(seed=seed, benchmark="switching", adam_warmup=False)
    #     # train_dynamic_precision_fixed_beta_50(seed=seed, benchmark="switching", adam_warmup=False)
    #     train_dynamic_precision_wave_curvature(seed=seed, benchmark='fp64')
    #     train_dynamic_precision_wave_curvature(seed=seed, benchmark='fp32')

        # train_dynamic_precision_wave_curvature(seed=seed,benchmark='fp32')
        # train_dynamic_precision_wave_curvature(seed=seed)
        # train_dynamic_precision_wave_curvature(seed=seed, benchmark='fp64')
        # train_dynamic_precision_wave_curvature(seed=seed, benchmark="switching")
if __name__ == "__main__":
    main()
