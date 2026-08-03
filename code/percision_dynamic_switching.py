"""
We follow the same model structure as paper 'FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks'
, the model class and data generation are derive from https://github.com/miniHuiHui/PINN_FP64/ for comparison.
The get_ns2d_c_data() function is implement based on PINNacle paper and also use the reference data from PINNacle repository: https://github.com/i207m/pinnacle 

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
import deepxde as dde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(str(Path("../PINN_FP64-main/"))) #this is the code repository for paper 'FP64 is All You Need: Rethinking Failure Modes in Physics-Informed Neural Networks'

from models import QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only, PINNMamba #from FP64 repository
from util import make_time_sequence #from FP64 repository
from ssbroyden2_torch import DenseSSBroyden2
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


def init_model(model_name='PINN', hidden_dim=512, num_layer=4, dtype=torch.float64, in_dim=2, out_dim=1):
    """Initialize model with support for different architectures.

    Args:
        model_name: 'PINN', 'PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba', 'KAN', 'QRes', 'ProPINN'
        hidden_dim: Hidden dimension for PINN (default 512)
        num_layer: Number of layers for PINN (default 4)
        dtype: torch.float32 or torch.float64
        in_dim: Number of input coordinates (default 2)
        out_dim: Number of model outputs (default 1)
    """

    if model_name == 'PINNsFormer':
        model = PINNsFormer.Model(in_dim=in_dim, hidden_dim=32, out_dim=out_dim, num_layer=1).to(dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'PINNsFormer_Enc_Only':
        model = PINNsFormer_Enc_Only.Model(in_dim=in_dim, hidden_dim=32, out_dim=out_dim, num_layer=1).to(dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'KAN':
        model = KAN.Model(width=[in_dim, 5, 5, out_dim], grid=5, k=3, grid_eps=1.0,
                          noise_scale_base=0.25, device=device).to(dtype).to(device)
    elif model_name == 'QRes':
        model = QRes.Model(in_dim=in_dim, hidden_dim=256, out_dim=out_dim, num_layer=4).to(dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'PINNMamba':
        model = PINNMamba.Model(in_dim=in_dim, hidden_dim=32, out_dim=out_dim, num_layer=1).to(dtype).to(device)
        model.apply(init_weights)
    else:  # Default to PINN
        model = PINN(hidden_dim=hidden_dim, num_layer=num_layer, in_dim=in_dim, out_dim=out_dim).to(dtype).to(device)
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

def _format_lid_amplitude_tag(lid_amplitude):
    """Return the PINNacle lid-driven reference-data tag, e.g. a4, a8, a16."""
    value = float(lid_amplitude)
    if value.is_integer():
        return f"a{int(value)}"
    return "a" + str(value).replace(".", "p")

def get_ns2d_c_data(
        datapath=None,
        domain_points=8192,
        boundary_points=2048,
        test_points=8192,
        bbox=(0.0, 1.0, 0.0, 1.0),
        lid_amplitude=4.0,
        nu=1.0 / 100.0,
        train_distribution="Hammersley",
):
    """Load and sample the PINNacle NS2d-C lid-driven-cavity benchmark.

    The benchmark is the steady incompressible 2D Navier--Stokes system on
    [0, 1] x [0, 1] with outputs (u, v, p), viscosity nu = 0.01 by default,
    a polynomial top-lid velocity u = a*x*(1-x), no-slip on the remaining
    walls, and the pressure gauge p(0, 0) = 0.  The argument lid_amplitude
    is the PINNacle ``a`` parameter.

    PINNacle defaults used here:
        * 8192 interior collocation points
        * 2048 boundary points
        * Hammersley low-discrepancy sampling
        * all valid rows in ref/lid_driven_a{a}.dat for evaluation

    Returns a dictionary containing fixed NumPy point arrays and the full
    reference solution. The same sampled points are reused by FP32, FP64,
    and dynamic-precision runs for a given function call.
    """
    import deepxde as dde
    if domain_points <= 0 or boundary_points <= 0 or test_points <= 0:
        raise ValueError("domain_points, boundary_points, and test_points must be positive.")

    lid_tag = _format_lid_amplitude_tag(lid_amplitude)
    reference_filename = f"lid_driven_{lid_tag}.dat"

    if datapath is None:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "ref" / reference_filename,
            Path("ref") / reference_filename,
            Path(reference_filename),
        ]
        datapath = next((path for path in candidates if path.exists()), candidates[0])
    else:
        datapath = Path(datapath)

    if not datapath.exists():
        raise FileNotFoundError(
            f"PINNacle reference data were not found at {datapath}. "
            f"Expected {reference_filename} for lid_amplitude/a={lid_amplitude}. "
            "Place the file in a ref/ directory next to this script or pass datapath explicitly."
        )

    ref_data = np.loadtxt(datapath, comments="%").astype(np.float32)
    if ref_data.ndim != 2 or ref_data.shape[1] < 5:
        raise ValueError(
            f"Expected PINNacle {reference_filename} columns [x, y, u, v, p], "
            f"but received shape {ref_data.shape}."
        )

    valid_rows = ~np.isnan(ref_data[:, :5]).any(axis=1)
    ref_valid = ref_data[valid_rows, :5]

    geom = dde.geometry.Rectangle(
        xmin=[bbox[0], bbox[2]],
        xmax=[bbox[1], bbox[3]],
    )

    domain_np = geom.random_points(
        domain_points,
        random=train_distribution,
    ).astype(np.float32)
    boundary_np = geom.random_boundary_points(
        boundary_points,
        random=train_distribution,
    ).astype(np.float32)

    top_mask = np.isclose(boundary_np[:, 1], bbox[3])
    top_np = boundary_np[top_mask]
    other_np = boundary_np[~top_mask]
    if top_np.shape[0] == 0 or other_np.shape[0] == 0:
        raise RuntimeError(
            "Boundary sampling did not produce both top-wall and non-top-wall points."
        )

    pressure_anchor_np = np.asarray([[bbox[0], bbox[2]]], dtype=np.float32)

    return {
        "geom": geom,
        "res": domain_np,
        "boundary": boundary_np,
        "boundary_top": top_np,
        "boundary_other": other_np,
        "pressure_anchor": pressure_anchor_np,
        "eval_x": ref_valid[:, :2],
        "eval_uvp": ref_valid[:, 2:5],
        "ref_data": ref_valid,
        "bbox": tuple(float(value) for value in bbox),
        "lid_amplitude": float(lid_amplitude),
        "lid_tag": lid_tag,
        "nu": float(nu),
        "domain_points": int(domain_points),
        "boundary_points": int(boundary_points),
        "test_points": int(test_points),
        "train_distribution": train_distribution,
        "datapath": str(datapath),
    }


def get_heat10d_training_data(d=10, Nbc=100, Nin=100, Nt0=1000, T=1.0):
    """
    Paper-style random collocation sampling for the heat equation.

    Domain: Omega = [-1, 1]^d, t in [0, T]
    res: Nin random points in Omega x [0, T]
    b_left: Nt0 random points in Omega at t = 0
    b_upper: Nbc random points on each boundary face of partial Omega x [0, T]
    """
    x_res = np.random.uniform(-1.0, 1.0, size=(Nin, d))
    t_res = np.random.uniform(0.0, T, size=(Nin, 1))
    res = np.concatenate((x_res, t_res), axis=-1)

    x_ic = np.random.uniform(-1.0, 1.0, size=(Nt0, d))
    t_ic = np.zeros((Nt0, 1))
    b_left = np.concatenate((x_ic, t_ic), axis=-1)

    bcs = []
    for j in range(d):
        for value in [-1.0, 1.0]:
            x_bc = np.random.uniform(-1.0, 1.0, size=(Nbc, d))
            x_bc[:, j] = value
            t_bc = np.random.uniform(0.0, T, size=(Nbc, 1))
            bcs.append(np.concatenate((x_bc, t_bc), axis=-1))
    b_upper = np.concatenate(bcs, axis=0)

    return res, b_left, None, b_upper, None


def get_heat10d_test_data(d=10, Nbc_v=100, Nin_v=7000, T=1.0):
    """
    Paper-style test sampling for time-dependent problems.

    Test points are sampled from Omega x {T}: Nin_v random interior points and
    Nbc_v random points on each boundary face.
    """
    x_in = np.random.uniform(-1.0, 1.0, size=(Nin_v, d))
    t_in = np.full((Nin_v, 1), T)
    test_in = np.concatenate((x_in, t_in), axis=-1)

    bcs = []
    for j in range(d):
        for value in [-1.0, 1.0]:
            x_bc = np.random.uniform(-1.0, 1.0, size=(Nbc_v, d))
            x_bc[:, j] = value
            t_bc = np.full((Nbc_v, 1), T)
            bcs.append(np.concatenate((x_bc, t_bc), axis=-1))
    test_bc = np.concatenate(bcs, axis=0)

    test = np.concatenate((test_in, test_bc), axis=0)
    return test


def heat10d_exact_np(x, t, d=10):
    return np.cos(np.sum(x, axis=-1, keepdims=True) / d) * np.exp(-t)


def heat10d_exact_torch(x, t, d=10):
    return torch.cos(torch.sum(x, dim=-1, keepdim=True) / d) * torch.exp(-t)


def heat10d_forcing_torch(x, t, d=10):
    return (1.0 / d - 1.0) * torch.cos(torch.sum(x, dim=-1, keepdim=True) / d) * torch.exp(-t)


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

    def forward(self, *inputs):
        src = torch.cat(inputs, dim=-1)
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


def train_dynamic_precision_fixed_beta_50_curvature_ssbroyden2(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=5000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = "PINN",
        hidden_dim: int = 164,
        num_layer: int = 3,
        ssbroyden_inner_iter: int = 20,
        max_inverse_hessian_gib: float = 56,
):
    """
    Dynamic FP32--FP64 convection training using SSBroyden2.
    Dense SSBroyden2 stores an O(n^2) inverse Hessian so we use a smaller model (hidden_dim=164, num_layer=3).

    """

    if model_name != "PINN":
        raise ValueError(
            "Only implement for model_name='PINN'. "
        )


    if max_inverse_hessian_gib <= 0:
        raise ValueError("max_inverse_hessian_gib must be positive.")

    set_seed(seed)

    run_name = (
        f"convection_ssbroyden2_dynamic_precision_beta_{BETA_MAX}_"
        f"h{hidden_dim}_layers{num_layer}_seed{seed}"
    )
    print("\n" + "=" * 70)
    print(f"Starting SSBroyden2 Dynamic-Precision Run: {run_name}")
    print("=" * 70)

    MIN_SWITCH_STEP = 0
    MIN_DWELL_STEPS = 10
    COND_WINDOW = 10
    STUCK_PATIENCE = 1
    DIAGNOSTIC_INTERVAL = 10

    CURV_EPS = 1e-30
    LOG_PROXY_LOW = 2.5
    LOG_PROXY_HIGH = 2.5
    LOG_SLOPE_FLAT = 0.02
    LOG_SLOPE_UP = 0.03
    EMA_BETA = 0.9

    if benchmark is False:
        strategy = "dynamic"
        precision_state, current_dtype = "fp64", torch.float64
    elif benchmark == "switching":
        strategy = "switching_ssbroyden2"
        precision_state, current_dtype = "fp32", torch.float32
    elif benchmark == "fp64":
        strategy = "fp64"
        precision_state, current_dtype = "fp64", torch.float64
    elif benchmark == "fp32":
        strategy = "fp32"
        precision_state, current_dtype = "fp32", torch.float32
    else:
        raise ValueError(f"Unsupported benchmark value: {benchmark}")

    beta_used = BETA_MAX
    last_switch_step = 0
    has_switched = False

    model = init_model(
        model_name,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        dtype=current_dtype,
    )
    print(model)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    inverse_hessian_gib = (
        parameter_count
        * parameter_count
        * torch.tensor([], dtype=current_dtype).element_size()
        / 1024**3
    )

    print(f"Trainable parameters: {parameter_count:,}")
    print(
        "Dense inverse-Hessian state: "
        f"{inverse_hessian_gib:.4f} GiB in {current_dtype}"
    )

    max_inverse_hessian_bytes = int(max_inverse_hessian_gib * 1024**3)

    optimizer = DenseSSBroyden2(
        model.parameters(),
        lr=1.0,
        max_iter=ssbroyden_inner_iter,
        max_eval=max(25, ssbroyden_inner_iter * 5 // 4),
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=COND_WINDOW,
        line_search_fn="strong_wolfe",
        max_inverse_hessian_bytes=max_inverse_hessian_bytes,
    )

    wandb.init(
        project="pinn_convection_dynamic_precision_SSBroyden2",
        name=run_name,
        config={
            "strategy": strategy,
            "optimizer": "SSBroyden2",
            "seed": seed,
            "beta": BETA_MAX,
            "hidden_dim": hidden_dim,
            "num_linear_layers": num_layer,
            "num_hidden_layers": num_layer - 1,
            "parameter_count": parameter_count,
            "inverse_hessian_gib_initial_dtype": inverse_hessian_gib,
            "ssbroyden_inner_iter": ssbroyden_inner_iter,
            "min_switch_step": MIN_SWITCH_STEP,
            "min_dwell_steps": MIN_DWELL_STEPS,
            "cond_window": COND_WINDOW,
            "stuck_patience": STUCK_PATIENCE,
            "LOG_PROXY_LOW": LOG_PROXY_LOW,
            "LOG_PROXY_HIGH": LOG_PROXY_HIGH,
        },
        reinit=True,
    )

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data(
        [0, 2 * np.pi],
        [0, 1],
        101,
        101,
    )
    (
        res_np,
        b_left_np,
        b_right_np,
        b_upper_np,
        b_lower_np,
    ) = preprocess_data_for_model(
        model_name,
        res_np,
        b_left_np,
        b_right_np,
        b_upper_np,
        b_lower_np,
    )

    data_cache = {}
    for dtype in (torch.float32, torch.float64):
        data_cache[dtype] = {
            "res": torch.tensor(res_np, dtype=dtype, device=device),
            "b_left": torch.tensor(b_left_np, dtype=dtype, device=device),
            "b_right": torch.tensor(b_right_np, dtype=dtype, device=device),
            "b_upper": torch.tensor(b_upper_np, dtype=dtype, device=device),
            "b_lower": torch.tensor(b_lower_np, dtype=dtype, device=device),
        }

    def rebuild_data(dtype):
        res = data_cache[dtype]["res"].detach().clone().requires_grad_(True)
        b_left = (
            data_cache[dtype]["b_left"].detach().clone().requires_grad_(True)
        )
        b_right = (
            data_cache[dtype]["b_right"].detach().clone().requires_grad_(True)
        )
        b_upper = (
            data_cache[dtype]["b_upper"].detach().clone().requires_grad_(True)
        )
        b_lower = (
            data_cache[dtype]["b_lower"].detach().clone().requires_grad_(True)
        )

        x_res, t_res = res[..., 0:1], res[..., 1:2]
        x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
        x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
        x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
        x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

        return (
            x_res,
            t_res,
            x_left,
            t_left,
            x_right,
            t_right,
            x_upper,
            t_upper,
            x_lower,
            t_lower,
        )

    (
        x_res,
        t_res,
        x_left,
        t_left,
        x_right,
        t_right,
        x_upper,
        t_upper,
        x_lower,
        t_lower,
    ) = rebuild_data(current_dtype)

    def compute_losses():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        if rescale_derivative:
            scale = 1024.0
            u_scaled = pred_res * scale
            u_t_scaled = torch.autograd.grad(
                u_scaled,
                t_res,
                grad_outputs=torch.ones_like(u_scaled),
                retain_graph=True,
                create_graph=True,
            )[0]
            u_x_scaled = torch.autograd.grad(
                u_scaled,
                x_res,
                grad_outputs=torch.ones_like(u_scaled),
                retain_graph=True,
                create_graph=True,
            )[0]
            u_t = u_t_scaled / scale
            u_x = u_x_scaled / scale
        else:
            u_t = torch.autograd.grad(
                pred_res,
                t_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=True,
            )[0]
            u_x = torch.autograd.grad(
                pred_res,
                x_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=True,
            )[0]

        loss_res = torch.mean((u_t + beta_used * u_x) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean(
            (pred_left[..., 0] - torch.sin(x_left[..., 0])) ** 2
        )
        loss = loss_res + loss_bc + loss_ic
        return loss, loss_res, loss_ic, loss_bc

    def get_quasi_newton_curvature_proxy():
        if not optimizer.state:
            return None

        state = optimizer.state[next(iter(optimizer.state))]
        old_dirs = state.get("old_dirs", [])
        old_stps = state.get("old_stps", [])

        directional_curvatures = []
        for y, s in zip(old_dirs, old_stps):
            y = y.reshape(-1)
            s = s.reshape(-1)
            y_s = torch.dot(s, y).item()
            s_s = torch.dot(s, s).item()

            if (
                np.isfinite(y_s)
                and np.isfinite(s_s)
                and s_s > CURV_EPS
                and y_s > CURV_EPS
            ):
                directional_curvatures.append(y_s / s_s)

        if len(directional_curvatures) < 2:
            return None

        directional_curvatures = np.asarray(
            directional_curvatures,
            dtype=np.float64,
        )
        k_min = max(float(np.min(directional_curvatures)), 1e-12)
        k_max = max(float(np.max(directional_curvatures)), k_min)
        return k_max / k_min

    def max_parameter_relative_update(parameters_before):
        max_relative_update = 0.0
        for old_parameter, new_parameter in zip(
            parameters_before,
            model.parameters(),
        ):
            delta = new_parameter.detach() - old_parameter
            relative_update = (
                delta.abs().max().item()
                / (old_parameter.abs().max().item() + 1e-16)
            )
            max_relative_update = max(
                max_relative_update,
                relative_update,
            )
        return max_relative_update

    cond_history = deque(maxlen=COND_WINDOW)
    stats = {}
    fp32_tiny_update_count = 0
    switch_event = 0

    smoothed_log_proxy = None
    previous_smoothed_log_proxy = None
    proxy_slope = 0.0
    curvature_proxy = None

    best_loss = float("inf")
    stagnant_steps = 0
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        do_diagnostic = step % DIAGNOSTIC_INTERVAL == 0
        track_tiny_update = precision_state == "fp32"

        if track_tiny_update:
            parameters_before = [
                parameter.detach().clone()
                for parameter in model.parameters()
            ]
        else:
            parameters_before = None

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, loss_res, loss_ic, loss_bc = compute_losses()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite SSBroyden2 loss at outer step {step}."
                )

            loss.backward()

            stats["loss"] = float(loss.detach())
            stats["loss_res"] = float(loss_res.detach())
            stats["loss_ic"] = float(loss_ic.detach())
            stats["loss_bc"] = float(loss_bc.detach())
            return loss

        try:
            optimizer.step(closure)
        except (RuntimeError, FloatingPointError) as exc:
            print(
                f"SSBroyden2 failed at outer step {step}: "
                f"{type(exc).__name__}: {exc}"
            )
            wandb.log({
                "optimizer_failure": 1,
                "failure_step": step,
            })
            break

        if track_tiny_update:
            relative_update = max_parameter_relative_update(parameters_before)
            if relative_update < 1e-8:
                fp32_tiny_update_count += 1
            else:
                fp32_tiny_update_count = 0
        else:
            fp32_tiny_update_count = 0

        if do_diagnostic:
            curvature_proxy = get_quasi_newton_curvature_proxy()

            if (
                curvature_proxy is not None
                and np.isfinite(curvature_proxy)
                and curvature_proxy > 0.0
            ):
                cond_history.append(curvature_proxy)
                log_proxy = float(np.log10(curvature_proxy + 1e-12))

                if smoothed_log_proxy is None:
                    smoothed_log_proxy = log_proxy
                    previous_smoothed_log_proxy = log_proxy
                    proxy_slope = 0.0
                else:
                    previous_smoothed_log_proxy = smoothed_log_proxy
                    smoothed_log_proxy = (
                        EMA_BETA * smoothed_log_proxy
                        + (1.0 - EMA_BETA) * log_proxy
                    )
                    proxy_slope = (
                        smoothed_log_proxy
                        - previous_smoothed_log_proxy
                    )

        can_switch = (
            step - last_switch_step
        ) >= MIN_DWELL_STEPS

        if benchmark is False:
            has_full_proxy_context = (
                len(cond_history) >= 2
                and smoothed_log_proxy is not None
                and np.isfinite(smoothed_log_proxy)
                and np.isfinite(proxy_slope)
            )

            trigger_to_fp32 = (
                do_diagnostic
                and precision_state == "fp64"
                and step >= MIN_SWITCH_STEP
                and can_switch
                and has_full_proxy_context
                and smoothed_log_proxy < LOG_PROXY_LOW
                and abs(proxy_slope) < LOG_SLOPE_FLAT
                and fp32_tiny_update_count < STUCK_PATIENCE
            )

            trigger_to_fp64 = (
                precision_state == "fp32"
                and (
                    fp32_tiny_update_count >= STUCK_PATIENCE
                    or (
                        do_diagnostic
                        and has_full_proxy_context
                        and (
                            smoothed_log_proxy > LOG_PROXY_HIGH
                            or proxy_slope > LOG_SLOPE_UP
                        )
                    )
                )
            )

            if trigger_to_fp32:
                precision_state = "fp32"
                current_dtype = torch.float32
                switch_event = 1
            elif trigger_to_fp64:
                precision_state = "fp64"
                current_dtype = torch.float64
                switch_event = 1

        elif benchmark == "switching":
            if not has_switched and precision_state == "fp32":
                if fp32_tiny_update_count >= STUCK_PATIENCE:
                    precision_state = "fp64"
                    current_dtype = torch.float64
                    switch_event = 1
                    has_switched = True

        if switch_event == 1:
            last_switch_step = step

            print(
                f"\nStep {step}: SWITCH -> {precision_state} "
                f"(log_proxy="
                f"{smoothed_log_proxy if smoothed_log_proxy is not None else float('nan'):.4f}, "
                f"slope={proxy_slope:.4e}, "
                f"tiny={fp32_tiny_update_count})\n"
            )

            wandb.log({
                "switch_event_triggered": 1,
                "target_precision_fp64": (
                    1 if precision_state == "fp64" else 0
                ),
            })

            model = model.to(dtype=current_dtype)
            optimizer.cast_state(
                dtype=current_dtype,
                device=torch.device(device),
            )

            (
                x_res,
                t_res,
                x_left,
                t_left,
                x_right,
                t_right,
                x_upper,
                t_upper,
                x_lower,
                t_lower,
            ) = rebuild_data(current_dtype)

            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            with torch.no_grad():
                x_eval = (
                    x_res[:, -1, :]
                    if x_res.dim() == 3
                    else x_res
                )
                t_eval = (
                    t_res[:, -1, :]
                    if t_res.dim() == 3
                    else t_res
                )

                u_exact = torch.sin(x_eval - beta_used * t_eval)
                u_pred = model(x_res, t_res)

                if u_pred.dim() == 3:
                    u_pred = u_pred[:, -1, :]

                rRMSE = torch.sqrt(
                    torch.sum((u_exact - u_pred) ** 2)
                    / torch.sum(u_exact ** 2)
                ).item()
                rMAE = (
                    torch.sum(torch.abs(u_exact - u_pred))
                    / torch.sum(torch.abs(u_exact))
                ).item()

            optimizer_state = optimizer.state[
                next(iter(optimizer.state))
            ]

            print(
                f"Step {step} | "
                f"Loss {stats['loss']:.3e} | "
                f"rRMSE {rRMSE:.3e} | "
                f"rMAE {rMAE:.3e} | "
                f"log_proxy "
                f"{smoothed_log_proxy if smoothed_log_proxy is not None else 0.0:.4f} | "
                f"slope {proxy_slope:.4e} | "
                f"tiny {fp32_tiny_update_count} | "
                f"state {precision_state}"
            )

            wandb.log({
                "loss": stats["loss"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "smoothed_log_proxy": (
                    smoothed_log_proxy
                    if smoothed_log_proxy is not None
                    else 0.0
                ),
                "proxy_slope": proxy_slope,
                "precision_fp64": (
                    1 if current_dtype == torch.float64 else 0
                ),
                "tiny_updates_count": fp32_tiny_update_count,
                "rRMSE": rRMSE,
                "rMAE": rMAE,
                "ssbroyden2_updates": optimizer_state.get(
                    "ssbroyden2_updates", 0
                ),
                "bfgs_fallback_updates": optimizer_state.get(
                    "bfgs_fallback_updates", 0
                ),
                "skipped_updates": optimizer_state.get(
                    "skipped_updates", 0
                ),
            })

        if stats["loss"] < best_loss - 1e-7:
            best_loss = stats["loss"]
            stagnant_steps = 0
        elif step > 100:
            stagnant_steps += 1

        if stagnant_steps > 50:
            print(
                f"Converged at step {step}: loss has not improved by "
                "1e-7 for more than 50 outer optimizer steps."
            )
            break

    duration = time.time() - start_time

    print(f"Training finished in {duration:.2f} seconds.")
    wandb.log({
        "training_time_seconds": duration,
        "final_precision_fp64": (
            1 if current_dtype == torch.float64 else 0
        ),
    })

    plot_convection_results(
        model,
        beta_used,
        run_name,
        strategy,
        model_name,
    )

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


def train_dynamic_precision_ns2d_c_curvature(
        seed=1234,
        MAX_STEPS=5000,
        benchmark: bool | str = False,
        model_name: str = 'PINN',
        datapath=None,
        domain_points: int = 8192,
        boundary_points: int = 2048,
        a: float = 4.0,
):
    """Train PINNacle NS2d-C with the convection controller unchanged.

    Benchmark-specific changes are limited to the PINNacle NS2d-C data,
    PDE/BC definitions, three-output network, and standard PINNacle reference
    evaluation. The FP32/FP64 state machine, L-BFGS settings, curvature proxy,
    logging cadence, tiny-update trigger, and early-stopping rule follow
    train_dynamic_precision_fixed_beta_50_curvature().
    """
    set_seed(seed)
    try:
        import deepxde as dde
        dde.config.set_random_seed(seed)
    except Exception:
        pass

    lid_tag = _format_lid_amplitude_tag(a)
    run_name = f"ns2d_c_{lid_tag}_dynamic_precision_curvature_seed{seed}"
    print("\n" + "=" * 72)
    print(f"Starting Dynamic-Precision Curvature Run (PINNacle NS2d-C, {lid_tag}): {run_name}")
    print("=" * 72)

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
            project=f"pinn_ns2dc_dynamic_precision_{model_name}",
            name=run_name,
            config={
                "strategy": strategy,
                "seed": seed,
                "benchmark": "PINNacle_NS2d-C",
                "lid_amplitude_a": float(a),
                "reference_file": f"lid_driven_{lid_tag}.dat" if datapath is None else str(datapath),
                "domain_points": domain_points,
                "boundary_points": boundary_points,
                "test_points_config": 8192,
                "evaluation": "PINNacle full-reference L2RE/L1RE",
                "min_switch_step": "NA",
                "min_dwell_steps": "NA",
                "cond_window": "NA",
                "trigger_patience": "NA",
                "stuck_patience": "NA",
            },
            reinit=True,
        )
    else:
        strategy = "dynamic"
        wandb.init(
            project=f"pinn_ns2dc_dynamic_precision_{model_name}",
            name=run_name,
            config={
                "strategy": strategy,
                "seed": seed,
                "benchmark": "PINNacle_NS2d-C",
                "lid_amplitude_a": float(a),
                "reference_file": f"lid_driven_{lid_tag}.dat" if datapath is None else str(datapath),
                "domain_points": domain_points,
                "boundary_points": boundary_points,
                "test_points_config": 8192,
                "evaluation": "PINNacle full-reference L2RE/L1RE",
                "min_switch_step": MIN_SWITCH_STEP,
                "min_dwell_steps": MIN_DWELL_STEPS,
                "cond_window": COND_WINDOW,
                "trigger_patience": TRIGGER_PATIENCE,
                "stuck_patience": STUCK_PATIENCE,
                "LOG_PROXY_LOW": LOG_PROXY_LOW,
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

    # Preserve the convection trainer's default PINN width/depth. Only the
    # required output dimension changes from one scalar to (u, v, p).
    if model_name == 'PINN':
        model = PINN(hidden_dim=512, num_layer=4, in_dim=2, out_dim=3).to(current_dtype).to(device)
        model.apply(init_weights)
    elif model_name == 'KAN':
        model = KAN.Model(
            width=[2, 5, 5, 3], grid=5, k=3, grid_eps=1.0,
            noise_scale_base=0.25, device=device,
        ).to(current_dtype).to(device)
    elif model_name == 'QRes':
        model = QRes.Model(in_dim=2, hidden_dim=256, out_dim=3, num_layer=4).to(current_dtype).to(device)
        model.apply(init_weights)
    else:
        raise ValueError(
            "The steady point-cloud NS2d-C implementation supports "
            "model_name='PINN', 'KAN', or 'QRes'."
        )
    print(model)

    ns_data = get_ns2d_c_data(
        datapath=datapath,
        domain_points=domain_points,
        boundary_points=boundary_points,
        lid_amplitude=a,
    )
    nu_used = ns_data["nu"]
    lid_amplitude = ns_data["lid_amplitude"]

    data_cache = {}
    for dt in [torch.float32, torch.float64]:
        data_cache[dt] = {
            "res": torch.tensor(ns_data["res"], dtype=dt, device=device),
            "boundary_top": torch.tensor(ns_data["boundary_top"], dtype=dt, device=device),
            "boundary_other": torch.tensor(ns_data["boundary_other"], dtype=dt, device=device),
            "pressure_anchor": torch.tensor(ns_data["pressure_anchor"], dtype=dt, device=device),
            "eval_x": torch.tensor(ns_data["eval_x"], dtype=dt, device=device),
            "eval_uvp": torch.tensor(ns_data["eval_uvp"], dtype=dt, device=device),
        }

    def _cast_obj(obj, dtype, target_device):
        if torch.is_tensor(obj):
            if obj.is_floating_point():
                return obj.to(device=target_device, dtype=dtype)
            return obj.to(device=target_device)
        elif isinstance(obj, dict):
            return {k: _cast_obj(v, dtype, target_device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_cast_obj(v, dtype, target_device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(_cast_obj(v, dtype, target_device) for v in obj)
        return obj

    def rebuild_data_and_optimizer(dtype, model, old_optimizer=None):
        cache = data_cache[dtype]
        x_res_new = cache["res"].detach().clone().requires_grad_(True)
        x_top_new = cache["boundary_top"].detach().clone().requires_grad_(True)
        x_other_new = cache["boundary_other"].detach().clone().requires_grad_(True)
        x_anchor_new = cache["pressure_anchor"].detach().clone().requires_grad_(True)

        optimizer_new = torch.optim.LBFGS(
            model.parameters(),
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
        )
        if old_optimizer is not None and isinstance(old_optimizer, torch.optim.LBFGS):
            print("Optimizer type 'LBFGS' is the same. Attempting to transfer state.")
            try:
                old_state_dict = old_optimizer.state_dict()
                old_state_dict["state"] = _cast_obj(old_state_dict["state"], dtype, device)
                optimizer_new.load_state_dict(old_state_dict)
                print("Successfully transferred optimizer state.")
            except Exception as exc:
                print(
                    f"Could not transfer optimizer state: {exc}. "
                    "Re-initializing optimizer from scratch."
                )
        else:
            if old_optimizer is not None:
                print(
                    f"Optimizer types differ ({type(old_optimizer).__name__} -> LBFGS). "
                    "Re-initializing optimizer."
                )
            else:
                print("No old optimizer provided. Initializing optimizer from scratch.")

        return optimizer_new, x_res_new, x_top_new, x_other_new, x_anchor_new

    optimizer, x_res, x_top, x_other, x_pressure_anchor = rebuild_data_and_optimizer(
        current_dtype, model
    )

    def _predict(points):
        if model_name in {'KAN', 'QRes'}:
            output = model(points[:, 0:1], points[:, 1:2])
        else:
            output = model(points[:, 0:1], points[:, 1:2])
        if output.dim() == 3:
            output = output[:, -1, :]
        if output.shape[-1] != 3:
            raise RuntimeError(
                f"NS2d-C requires three outputs (u, v, p), got {tuple(output.shape)}."
            )
        return output

    def compute_losses():
        pred_res = _predict(x_res)
        u = pred_res[:, 0:1]
        v = pred_res[:, 1:2]
        p = pred_res[:, 2:3]

        grad_u = torch.autograd.grad(
            u, x_res, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True,
        )[0]
        grad_v = torch.autograd.grad(
            v, x_res, grad_outputs=torch.ones_like(v),
            retain_graph=True, create_graph=True,
        )[0]
        grad_p = torch.autograd.grad(
            p, x_res, grad_outputs=torch.ones_like(p),
            retain_graph=True, create_graph=True,
        )[0]

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, x_res, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True,
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y, x_res, grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True,
        )[0][:, 1:2]
        v_xx = torch.autograd.grad(
            v_x, x_res, grad_outputs=torch.ones_like(v_x),
            retain_graph=True, create_graph=True,
        )[0][:, 0:1]
        v_yy = torch.autograd.grad(
            v_y, x_res, grad_outputs=torch.ones_like(v_y),
            retain_graph=True, create_graph=True,
        )[0][:, 1:2]

        momentum_x = u * u_x + v * u_y + p_x - nu_used * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - nu_used * (v_xx + v_yy)
        continuity = u_x + v_y

        loss_momentum_x = torch.mean(momentum_x.square())
        loss_momentum_y = torch.mean(momentum_y.square())
        loss_continuity = torch.mean(continuity.square())
        loss_res = loss_momentum_x + loss_momentum_y + loss_continuity

        pred_top = _predict(x_top)
        pred_other = _predict(x_other)
        pred_pressure_anchor = _predict(x_pressure_anchor)

        top_u_target = lid_amplitude * x_top[:, 0:1] * (1.0 - x_top[:, 0:1])
        loss_bc_u_top = torch.mean((pred_top[:, 0:1] - top_u_target).square())
        loss_bc_v_top = torch.mean(pred_top[:, 1:2].square())
        loss_bc_u_other = torch.mean(pred_other[:, 0:1].square())
        loss_bc_v_other = torch.mean(pred_other[:, 1:2].square())
        loss_bc_p_anchor = torch.mean(pred_pressure_anchor[:, 2:3].square())
        loss_bc = (
            loss_bc_u_top + loss_bc_v_top
            + loss_bc_u_other + loss_bc_v_other
            + loss_bc_p_anchor
        )

        # Steady benchmark: retained only for the same logging interface.
        loss_ic = torch.zeros((), dtype=loss_res.dtype, device=loss_res.device)
        loss = loss_res + loss_bc
        components = {
            "loss_momentum_x": loss_momentum_x,
            "loss_momentum_y": loss_momentum_y,
            "loss_continuity": loss_continuity,
            "loss_bc_u_top": loss_bc_u_top,
            "loss_bc_v_top": loss_bc_v_top,
            "loss_bc_u_other": loss_bc_u_other,
            "loss_bc_v_other": loss_bc_v_other,
            "loss_bc_p_anchor": loss_bc_p_anchor,
        }
        return loss, loss_res, loss_ic, loss_bc, components

    def evaluate_reference_solution():
        """PINNacle TesterCallback-compatible errors over all (u, v, p)."""
        cache = data_cache[current_dtype]
        with torch.no_grad():
            exact = cache["eval_uvp"]
            pred = _predict(cache["eval_x"])
            error = pred - exact

            mse = torch.mean(error.square())
            mae = torch.mean(torch.abs(error))
            mxe = torch.max(torch.abs(error))
            crmse = torch.abs(torch.mean(error))
            solution_l2 = torch.sqrt(torch.mean(exact.square()))
            solution_l1 = torch.mean(torch.abs(exact))
            tiny = torch.finfo(exact.dtype).tiny

            # PINNacle-style relative errors. These are also exactly the same
            # definitions used by the other PDE trainers in this script:
            # rRMSE = sqrt(sum((y - y_hat)^2) / sum(y^2)) and
            # rMAE  = sum(|y - y_hat|) / sum(|y|).  The mean-based forms
            # below are algebraically equivalent because they average over
            # the same flattened (u, v, p) tensor.
            l2re = torch.sqrt(mse) / torch.clamp(solution_l2, min=tiny)
            l1re = mae / torch.clamp(solution_l1, min=tiny)
            rrmse = l2re
            rmae = l1re

            component_mse = torch.mean(error.square(), dim=0)
            component_solution_l2 = torch.sqrt(torch.mean(exact.square(), dim=0))
            component_l2re = torch.sqrt(component_mse) / torch.clamp(
                component_solution_l2,
                min=tiny,
            )
            component_l1re = torch.mean(torch.abs(error), dim=0) / torch.clamp(
                torch.mean(torch.abs(exact), dim=0),
                min=tiny,
            )

        return {
            "MSE": float(mse.item()),
            "MAE": float(mae.item()),
            "MXE": float(mxe.item()),
            "CRMSE": float(crmse.item()),
            "L2RE": float(l2re.item()),
            "L1RE": float(l1re.item()),
            "rRMSE": float(rrmse.item()),
            "rMAE": float(rmae.item()),
            "RMAE": float(rmae.item()),
            "u_L2RE": float(component_l2re[0].item()),
            "v_L2RE": float(component_l2re[1].item()),
            "p_L2RE": float(component_l2re[2].item()),
            "u_rRMSE": float(component_l2re[0].item()),
            "v_rRMSE": float(component_l2re[1].item()),
            "p_rRMSE": float(component_l2re[2].item()),
            "u_rMAE": float(component_l1re[0].item()),
            "v_rMAE": float(component_l1re[1].item()),
            "p_rMAE": float(component_l1re[2].item()),
        }

    def get_quasi_newton_curvature_proxy():
        if not optimizer.state:
            return None
        state = optimizer.state[next(iter(optimizer.state))]
        if "old_dirs" not in state or "old_stps" not in state:
            return None
        directional_curvatures = []
        for y_vector, s_vector in zip(state["old_dirs"], state["old_stps"]):
            y_flat, s_flat = y_vector.flatten(), s_vector.flatten()
            ys = torch.dot(s_flat, y_flat).item()
            ss = torch.dot(s_flat, s_flat).item()
            if (
                np.isfinite(ys) and np.isfinite(ss)
                and ss > CURV_EPS and ys > CURV_EPS
            ):
                directional_curvatures.append(ys / ss)
        if len(directional_curvatures) < 2:
            return None
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
    completed_step = 0
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        completed_step = step
        do_diagnostic = step % DIAGNOSTIC_INTERVAL == 0
        track_tiny_update = precision_state == "fp32"

        if track_tiny_update:
            params_before = [p.detach().clone() for p in model.parameters()]
        else:
            params_before = None

        def closure():
            optimizer.zero_grad(set_to_none=True)
            x_res.grad = None
            x_top.grad = None
            x_other.grad = None
            x_pressure_anchor.grad = None

            loss, loss_res, loss_ic, loss_bc, components = compute_losses()
            if torch.isnan(loss) or torch.isinf(loss):
                raise EOFError("Invalid loss value")
            loss.backward()

            stats["loss"] = float(loss.detach())
            stats["loss_res"] = float(loss_res.detach())
            stats["loss_ic"] = float(loss_ic.detach())
            stats["loss_bc"] = float(loss_bc.detach())
            for key, value in components.items():
                stats[key] = float(value.detach())
            return loss

        optimizer.step(closure)

        if track_tiny_update:
            max_rel_update = 0.0
            for p_old, p_new in zip(params_before, model.parameters()):
                delta = p_new.detach() - p_old
                rel = delta.abs().max().item() / (p_old.abs().max().item() + 1e-16)
                max_rel_update = max(max_rel_update, rel)
            if max_rel_update < 1e-8:
                fp32_tiny_update_count += 1
            else:
                fp32_tiny_update_count = 0
        else:
            max_rel_update = 0.0
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
                    smoothed_log_proxy = (
                        EMA_BETA * smoothed_log_proxy
                        + (1.0 - EMA_BETA) * log_proxy
                    )
                    proxy_slope = smoothed_log_proxy - prev_smoothed_log_proxy

        can_switch = (step - last_switch_step) >= MIN_DWELL_STEPS

        if benchmark is False:
            has_full_proxy_context = (
                len(cond_history) >= 2
                and smoothed_log_proxy is not None
                and np.isfinite(smoothed_log_proxy)
                and np.isfinite(proxy_slope)
            )
            trigger_to_fp32 = (
                do_diagnostic
                and precision_state == "fp64"
                and step >= MIN_SWITCH_STEP
                and can_switch
                and has_full_proxy_context
                and smoothed_log_proxy < LOG_PROXY_LOW
                and abs(proxy_slope) < LOG_SLOPE_FLAT
                and fp32_tiny_update_count < STUCK_PATIENCE
            )
            trigger_to_fp64 = (
                precision_state == "fp32"
                and (
                    fp32_tiny_update_count >= STUCK_PATIENCE
                    or (
                        do_diagnostic
                        and np.isfinite(
                            curvature_proxy if curvature_proxy is not None else 1.0
                        )
                        and has_full_proxy_context
                        and (
                            smoothed_log_proxy > LOG_PROXY_HIGH
                            or proxy_slope > LOG_SLOPE_UP
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
                trigger_to_fp64 = fp32_tiny_update_count >= STUCK_PATIENCE
                if trigger_to_fp64:
                    precision_state, current_dtype, switch_event, has_switched = (
                        "fp64", torch.float64, 1, True
                    )

        if switch_event == 1:
            last_switch_step = step
            print(
                f"\nStep {step}: SWITCH -> {precision_state} "
                f"(log_proxy={smoothed_log_proxy if smoothed_log_proxy is not None else float('nan'):.4f}, "
                f"slope={proxy_slope:.4e}, tiny={fp32_tiny_update_count})\n"
            )
            wandb.log({
                "switch_event_triggered": 1,
                "target_precision_fp64": 1 if precision_state == "fp64" else 0,
            })

            model = model.to(dtype=current_dtype)
            old_optimizer = optimizer
            optimizer, x_res, x_top, x_other, x_pressure_anchor = rebuild_data_and_optimizer(
                dtype=current_dtype,
                model=model,
                old_optimizer=old_optimizer,
            )
            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            metrics = evaluate_reference_solution()
            displayed_log_proxy = (
                smoothed_log_proxy if smoothed_log_proxy is not None else 0.0
            )
            print(
                f"Step {step} | Loss {stats['loss']:.3e} | "
                f"rRMSE {metrics['rRMSE']:.3e} | rMAE {metrics['rMAE']:.3e} | "
                f"L2RE {metrics['L2RE']:.3e} | L1RE {metrics['L1RE']:.3e} | "
                f"u/v/p rRMSE {metrics['u_rRMSE']:.3e}/"
                f"{metrics['v_rRMSE']:.3e}/{metrics['p_rRMSE']:.3e} | "
                f"log_proxy {displayed_log_proxy:.4f} | "
                f"slope {proxy_slope:.4e} | tiny {fp32_tiny_update_count} | "
                f"state {precision_state}"
            )

            log_payload = {
                "loss": stats["loss"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "loss_momentum_x": stats["loss_momentum_x"],
                "loss_momentum_y": stats["loss_momentum_y"],
                "loss_continuity": stats["loss_continuity"],
                "loss_bc_u_top": stats["loss_bc_u_top"],
                "loss_bc_v_top": stats["loss_bc_v_top"],
                "loss_bc_u_other": stats["loss_bc_u_other"],
                "loss_bc_v_other": stats["loss_bc_v_other"],
                "loss_bc_p_anchor": stats["loss_bc_p_anchor"],
                "smoothed_log_proxy": displayed_log_proxy,
                "proxy_slope": proxy_slope,
                "precision_fp64": 1 if current_dtype == torch.float64 else 0,
                "tiny_updates_count": fp32_tiny_update_count,
                "max_relative_update": max_rel_update,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "MXE": metrics["MXE"],
                "CRMSE": metrics["CRMSE"],
                "L2RE": metrics["L2RE"],
                "L1RE": metrics["L1RE"],
                "u_L2RE": metrics["u_L2RE"],
                "v_L2RE": metrics["v_L2RE"],
                "p_L2RE": metrics["p_L2RE"],
                # Same names as the other PDE trainers in this script.
                "rRMSE": metrics["rRMSE"],
                "rMAE": metrics["rMAE"],
                "RMAE": metrics["RMAE"],
                "u_rRMSE": metrics["u_rRMSE"],
                "v_rRMSE": metrics["v_rRMSE"],
                "p_rRMSE": metrics["p_rRMSE"],
                "u_rMAE": metrics["u_rMAE"],
                "v_rMAE": metrics["v_rMAE"],
                "p_rMAE": metrics["p_rMAE"],
            }
            wandb.log(log_payload)

        if stats["loss"] < best_loss - 1e-7:
            best_loss = stats["loss"]
            stagnant_steps = 0
        elif step > 100:
            stagnant_steps += 1

        if stagnant_steps > 50:
            print(
                f"Converged at step {step}: Loss hasn't improved by 1e-7 "
                "for more than 50 epochs."
            )
            break

    duration = time.time() - start_time
    final_metrics = evaluate_reference_solution()
    print(f"Training finished in {duration:.2f} seconds.")
    print(
        f"Final metrics | rRMSE {final_metrics['rRMSE']:.4e} | "
        f"rMAE {final_metrics['rMAE']:.4e} | "
        f"L2RE {final_metrics['L2RE']:.4e} | L1RE {final_metrics['L1RE']:.4e}"
    )
    wandb.log({
        "training_time_seconds": duration,
        "completed_step": completed_step,
        "final_L2RE": final_metrics["L2RE"],
        "final_L1RE": final_metrics["L1RE"],
        "final_MXE": final_metrics["MXE"],
        "final_CRMSE": final_metrics["CRMSE"],
        "final_u_L2RE": final_metrics["u_L2RE"],
        "final_v_L2RE": final_metrics["v_L2RE"],
        "final_p_L2RE": final_metrics["p_L2RE"],
        "final_rRMSE": final_metrics["rRMSE"],
        "final_rMAE": final_metrics["rMAE"],
        "final_RMAE": final_metrics["RMAE"],
        "final_u_rRMSE": final_metrics["u_rRMSE"],
        "final_v_rRMSE": final_metrics["v_rRMSE"],
        "final_p_rRMSE": final_metrics["p_rRMSE"],
        "final_u_rMAE": final_metrics["u_rMAE"],
        "final_v_rMAE": final_metrics["v_rMAE"],
        "final_p_rMAE": final_metrics["p_rMAE"],
        "final_precision_fp64": 1 if current_dtype == torch.float64 else 0,
    })

    plot_ns2d_c_results(
        model=model,
        run_name=run_name,
        strategy=strategy,
        model_name=model_name,
        ns_data=ns_data,
    )
    wandb.finish()
    return model

def train_dynamic_precision_heat10d_curvature(
        seed=1234,
        D=10,
        Nbc=100,
        Nin=100,
        Nt0=1000,
        MAX_STEPS=50000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = 'PINN',
):
    set_seed(seed)
    run_name = f"heat{D}d_dynamic_precision_curvature_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Dynamic-Precision Curvature Run ({D}D Heat): {run_name}")
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
        project="PINN_heat_dynamic_precision_MLP".format(model_name),
        name=run_name,
        config={"strategy": strategy, "seed": seed, "dimension": D, "Nbc": Nbc, "Nin": Nin, "Nt0": Nt0,
                "min_switch_step": MIN_SWITCH_STEP,
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

    model = init_model(model_name, hidden_dim=64, num_layer=3, dtype=current_dtype, in_dim=D + 1)
    print(model)

    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_heat10d_training_data(
        d=D, Nbc=Nbc, Nin=Nin, Nt0=Nt0, T=1.0
    )
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = preprocess_data_for_model(
        model_name, res_np, b_left_np, b_right_np, b_upper_np, b_lower_np
    )

    np.random.seed(seed)
    test_np = get_heat10d_test_data(d=D, Nbc_v=100, Nin_v=7000, T=1.0)
    x_test_np = test_np[:, :D]
    t_test_np = test_np[:, D:D + 1]
    u_eval_exact = heat10d_exact_np(x_test_np, t_test_np, d=D)

    data_cache = {}
    for dt in [torch.float32, torch.float64]:
        data_cache[dt] = {
            "res": torch.tensor(res_np, dtype=dt, device=device),
            "b_left": torch.tensor(b_left_np, dtype=dt, device=device),
            "b_upper": torch.tensor(b_upper_np, dtype=dt, device=device),
            "test": torch.tensor(test_np, dtype=dt, device=device),
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
        test = data_cache[dtype]["test"].detach().clone()

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

        x_res, t_res = res[..., :D], res[..., D:D + 1]
        x_left, t_left = b_left[..., :D], b_left[..., D:D + 1]
        x_upper, t_upper = b_upper[..., :D], b_upper[..., D:D + 1]
        x_test, t_test = test[..., :D], test[..., D:D + 1]

        return optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_test, t_test

    optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_test, t_test = \
        rebuild_data_and_optimizer(current_dtype, model)

    def compute_laplacian(u, x):
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        laplace_u = 0.0
        for i in range(D):
            u_x_i = u_x[..., i:i + 1]
            u_xx_i = torch.autograd.grad(u_x_i, x, grad_outputs=torch.ones_like(u_x_i), retain_graph=True,
                                          create_graph=True)[0][..., i:i + 1]
            laplace_u = laplace_u + u_xx_i
        return laplace_u

    def compute_losses():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)

        if rescale_derivative:
            S = 1024.0
            u_scaled = pred_res * S
            u_t_scaled = torch.autograd.grad(u_scaled, t_res, grad_outputs=torch.ones_like(u_scaled), retain_graph=True,
                                             create_graph=True)[0]
            laplace_u_scaled = compute_laplacian(u_scaled, x_res)
            u_t, laplace_u = u_t_scaled / S, laplace_u_scaled / S
        else:
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]
            laplace_u = compute_laplacian(pred_res, x_res)

        f_val = heat10d_forcing_torch(x_res, t_res, d=D)
        loss_res = torch.mean((u_t - laplace_u - f_val) ** 2)
        loss_ic = torch.mean((pred_left - heat10d_exact_torch(x_left, t_left, d=D)) ** 2)
        loss_bc = torch.mean((pred_upper - heat10d_exact_torch(x_upper, t_upper, d=D)) ** 2)
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
            optimizer, x_res, t_res, x_left, t_left, x_upper, t_upper, x_test, t_test = \
                rebuild_data_and_optimizer(dtype=current_dtype, model=model, old_optimizer=old_optimizer)

            fp32_tiny_update_count = 0
            switch_event = 0

        if step % 10 == 0:
            # model.eval()
            with torch.no_grad():
                if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
                    u_pred_ev = model(x_test, t_test)
                else:
                    u_pred_ev = model(x_test, t_test)
                if u_pred_ev.dim() == 3: u_pred_ev = u_pred_ev[:, -1, :]
                u_pred_ev = u_pred_ev.cpu().numpy()

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
    plot_heat10d_results(model, D, run_name, strategy, model_name, seed)
    wandb.finish()
    return model



def train_brdr_fixed_beta_50(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=50000,
        benchmark: bool | str = False,
        rescale_derivative: bool = False,
        model_name: str = 'PINN',
):
    """
    BRDR implementation following Chen, Howard & Stinis (JCP 2025) as closely as
    possible, while keeping this project's convection data and NN architecture.

    Kept from our setup:
        - convection benchmark: x in [0, 2*pi], t in [0, 1], 101 x 101 grid
        - same preprocessing through preprocess_data_for_model
        - same model construction: init_model(model_name, hidden_dim=512, num_layer=4)
        - same residual definitions and evaluation/plotting utilities

    BRDR paper:
        - Adam optimizer, default FP32
        - betaC = betaW = 0.999, eps = 1e-14
        - pointwise IRDR weights with one GLOBAL mean normalization over PDE/BC/IC
        - adaptive scaling factor lambda/s, initialized at 1
        - bias correction for the R^4 exponential moving average
    """
    set_seed(seed)
    run_name = f"convection_BRDR_paper_fixedbeta_{BETA_MAX}_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting paper-style BRDR Run (Convection): {run_name}")
    print("=" * 60)

    # --- BRDR hyperparameters from the paper / official implementation ---
    BETA_C = 0.999
    BETA_W = 0.999
    EPSILON = 1e-14
    ADAM_LR = 1e-3
    LAMBDA0_RES = 1.0
    LAMBDA0_BC = 1.0
    LAMBDA0_IC = 1.0

    # Paper BRDR is FP32 + Adam. Keep benchmark='fp64' only for diagnostic ablation.
    if benchmark in [False, "fp32"]:
        current_dtype = torch.float32
    elif benchmark == "fp64":
        current_dtype = torch.float64
    else:
        raise ValueError("For paper-style BRDR, use benchmark=False/'fp32' or benchmark='fp64'.")

    strategy = "BRDR_paper_global_weights_adaptive_scale"

    wandb.init(
        project="pinn_convection_adaptive_weights".format(model_name),
        name=run_name,
        config={
            "strategy": strategy,
            "seed": seed,
            "beta": BETA_MAX,
            "beta_c": BETA_C,
            "beta_w": BETA_W,
            "epsilon": EPSILON,
            "adam_lr": ADAM_LR,
            "adaptive_scale": True,
            "precision": "fp64" if current_dtype == torch.float64 else "fp32",
            "model_name": model_name,
        },
        reinit=True,
    )

    beta_used = BETA_MAX

    model = init_model(model_name, hidden_dim=512, num_layer=4, dtype=current_dtype)
    print(model)

    # Same convection data/preprocessing as train_dynamic_precision_fixed_beta_50_curvature
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data([0, 2 * np.pi], [0, 1], 101, 101)
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = preprocess_data_for_model(
        model_name, res_np, b_left_np, b_right_np, b_upper_np, b_lower_np
    )

    res = torch.tensor(res_np, dtype=current_dtype, device=device).requires_grad_(True)
    b_left = torch.tensor(b_left_np, dtype=current_dtype, device=device).requires_grad_(True)
    b_right = torch.tensor(b_right_np, dtype=current_dtype, device=device).requires_grad_(True)
    b_upper = torch.tensor(b_upper_np, dtype=current_dtype, device=device).requires_grad_(True)
    b_lower = torch.tensor(b_lower_np, dtype=current_dtype, device=device).requires_grad_(True)

    x_res, t_res = res[..., 0:1], res[..., 1:2]
    x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
    x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
    x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
    x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    # --- BRDR state: weights initialized to 1, scaling factor initialized to 1 ---
    w_res = w_bc = w_ic = None
    exp4_res = exp4_bc = exp4_ic = None
    scale_s = 1.0

    stats = {}
    best_metric_loss = float("inf")
    stagnant_steps = 0
    start_time = time.time()

    def compute_raw_residuals(create_graph=True):
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        if rescale_derivative:
            S = 1024.0
            u_scaled = pred_res * S
            u_t_scaled = torch.autograd.grad(
                u_scaled, t_res,
                grad_outputs=torch.ones_like(u_scaled),
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            u_x_scaled = torch.autograd.grad(
                u_scaled, x_res,
                grad_outputs=torch.ones_like(u_scaled),
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            u_t, u_x = u_t_scaled / S, u_x_scaled / S
        else:
            u_t = torch.autograd.grad(
                pred_res, t_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            u_x = torch.autograd.grad(
                pred_res, x_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=create_graph,
            )[0]

        R_res = u_t + beta_used * u_x
        R_bc = pred_upper - pred_lower
        R_ic = pred_left[..., 0:1] - torch.sin(x_left[..., 0:1])
        return R_res, R_bc, R_ic

    def update_brdr_weights(R_res, R_bc, R_ic, step):
        nonlocal w_res, w_bc, w_ic, exp4_res, exp4_bc, exp4_ic

        with torch.no_grad():
            R2_res = R_res.detach().pow(2)
            R2_bc = R_bc.detach().pow(2)
            R2_ic = R_ic.detach().pow(2)

            if w_res is None:
                w_res = torch.ones_like(R2_res)
                w_bc = torch.ones_like(R2_bc)
                w_ic = torch.ones_like(R2_ic)
                exp4_res = torch.zeros_like(R2_res)
                exp4_bc = torch.zeros_like(R2_bc)
                exp4_ic = torch.zeros_like(R2_ic)

            # Official implementation form: exp += (1-betaC) * (R2*R2 - exp)
            exp4_res.add_((1.0 - BETA_C) * (R2_res.pow(2) - exp4_res))
            exp4_bc.add_((1.0 - BETA_C) * (R2_bc.pow(2) - exp4_bc))
            exp4_ic.add_((1.0 - BETA_C) * (R2_ic.pow(2) - exp4_ic))

            # Bias correction used in the official implementation.
            bias_correction = 1.0 / (1.0 - (BETA_C ** step))

            irdr_res = R2_res / (torch.sqrt(exp4_res * bias_correction) + EPSILON)
            irdr_bc = R2_bc / (torch.sqrt(exp4_bc * bias_correction) + EPSILON)
            irdr_ic = R2_ic / (torch.sqrt(exp4_ic * bias_correction) + EPSILON)

            # One global normalization across all training terms, not per loss group.
            total_irdr = irdr_res.sum() + irdr_bc.sum() + irdr_ic.sum()
            total_count = irdr_res.numel() + irdr_bc.numel() + irdr_ic.numel()
            global_irdr_mean = total_irdr / max(1, total_count)

            w_res.add_((1.0 - BETA_W) * (irdr_res / global_irdr_mean - w_res))
            w_bc.add_((1.0 - BETA_W) * (irdr_bc / global_irdr_mean - w_bc))
            w_ic.add_((1.0 - BETA_W) * (irdr_ic / global_irdr_mean - w_ic))

            return float(global_irdr_mean.detach()), float(w_res.mean()), float(w_bc.mean()), float(w_ic.mean())

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        R_res, R_bc, R_ic = compute_raw_residuals(create_graph=True)
        irdr_mean, w_res_mean, w_bc_mean, w_ic_mean = update_brdr_weights(R_res, R_bc, R_ic, step)

        loss_res = LAMBDA0_RES * torch.mean(w_res * R_res.pow(2))
        loss_bc = LAMBDA0_BC * torch.mean(w_bc * R_bc.pow(2))
        loss_ic = LAMBDA0_IC * torch.mean(w_ic * R_ic.pow(2))
        loss_unscaled = loss_res + loss_bc + loss_ic
        loss = scale_s * loss_unscaled

        if torch.isnan(loss) or torch.isinf(loss):
            raise EOFError("Invalid loss value")

        loss.backward()

        # Adaptive scaling factor from the paper / official implementation.
        with torch.no_grad():
            grad_norm_sq = torch.zeros((), dtype=current_dtype, device=device)
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += p.grad.detach().pow(2).sum()

            old_scale_s = scale_s
            if torch.isfinite(grad_norm_sq) and grad_norm_sq.item() > 0.0:
                # s_max = s_old / lr * 2 * L / ||grad L||^2
                s_max = old_scale_s * (2.0 * float(loss.detach())) / (ADAM_LR * float(grad_norm_sq.detach()))
                scale_s = (1.0 - ADAM_LR) * old_scale_s + ADAM_LR * s_max

                # Correct gradients because scale_s is updated after backpropagation.
                grad_correction = scale_s / old_scale_s
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(grad_correction)

        optimizer.step()

        stats["loss"] = float(loss.detach())
        stats["loss_unscaled"] = float(loss_unscaled.detach())
        stats["loss_res"] = float(loss_res.detach())
        stats["loss_ic"] = float(loss_ic.detach())
        stats["loss_bc"] = float(loss_bc.detach())
        stats["scale_s"] = float(scale_s)
        stats["w_res_mean"] = w_res_mean
        stats["w_bc_mean"] = w_bc_mean
        stats["w_ic_mean"] = w_ic_mean
        stats["irdr_mean"] = irdr_mean

        if step % 10 == 0:
            with torch.no_grad():
                x_eval = x_res[:, -1, :] if x_res.dim() == 3 else x_res
                t_eval = t_res[:, -1, :] if t_res.dim() == 3 else t_res
                u_exact = torch.sin(x_eval - beta_used * t_eval)
                u_pred = model(x_res, t_res)
                if u_pred.dim() == 3:
                    u_pred = u_pred[:, -1, :]
                rRMSE = torch.sqrt(torch.sum((u_exact - u_pred) ** 2) / torch.sum(u_exact ** 2)).item()
                rMAE = (torch.sum(torch.abs(u_exact - u_pred)) / torch.sum(torch.abs(u_exact))).item()

            print(
                f"Step {step} | Loss {stats['loss']:.3e} | Unscaled {stats['loss_unscaled']:.3e} | "
                f"rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | s {scale_s:.3e} | "
                f"w_mean(res/bc/ic) {w_res_mean:.2e}/{w_bc_mean:.2e}/{w_ic_mean:.2e}"
            )

            wandb.log({
                "loss": stats["loss"],
                "loss_unscaled": stats["loss_unscaled"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "scale_s": stats["scale_s"],
                "irdr_mean": stats["irdr_mean"],
                "w_res_mean": stats["w_res_mean"],
                "w_bc_mean": stats["w_bc_mean"],
                "w_ic_mean": stats["w_ic_mean"],
                "rRMSE": rRMSE,
                "rMAE": rMAE,
                "precision_fp64": 1 if current_dtype == torch.float64 else 0,
            })

        # Paper-style training uses a fixed number of Adam steps.
        # Do not early-stop here; use MAX_STEPS to control the training budget.

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")
    wandb.log({"training_time_seconds": duration})

    plot_convection_results(model, beta_used, run_name, strategy, model_name)
    wandb.finish()
    return model

def train_hayford_2024_repo_mixed_precision_fixed_beta_50(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=5000,
        model_name: str = "PINN",
        adam_lr: float = 1e-3,
        adam_eps: float = 1e-4,
):
    """
    Implement Hayford et al. 2024 mixed-precision comparator applied to our convection benchmark.

    """

    set_seed(seed)

    run_name = f"convection_Hayford2024_repoMixedAMP_Adam_fixedbeta_{BETA_MAX}_seed{seed}"
    print("\n" + "=" * 60)
    print(f"Starting Hayford 2024 Repo-Style Mixed Precision Comparator: {run_name}")
    print("=" * 60)

    strategy = "hayford_2024_repo_mixed_float16_adam"
    beta_used = BETA_MAX

    #   model parameters stay float32
    #   training inputs are float16 on CUDA
    #   autocast uses float16 compute
    master_dtype = torch.float32
    train_dtype = torch.float16 if device.type == "cuda" else torch.float32
    eval_dtype = torch.float32

    amp_enabled = device.type == "cuda"
    autocast_device = device.type

    wandb.init(
        project="pinn_convection_dynamic_precision_plotMLP{}".format(model_name),
        name=run_name,
        config={
            "strategy": strategy,
            "seed": seed,
            "beta": BETA_MAX,

            # Same experimental setup as our method.
            "model_name": model_name,
            "hidden_dim": 512,
            "num_layer": 4,
            "data_domain_x": "[0, 2*pi]",
            "data_domain_t": "[0, 1]",
            "data_grid": "101x101",
            "loss": "loss_res + loss_bc + loss_ic",
            "evaluation_sampling": "same_as_dynamic_precision_method",

            # Hayford repo-style mixed precision.
            "backend_equivalent": "PyTorch equivalent of TensorFlow mixed_float16",
            "master_dtype": "float32",
            "train_data_dtype": "float16" if train_dtype == torch.float16 else "float32",
            "compute_dtype": "float16" if amp_enabled else "float32",
            "optimizer": "Adam",
            "adam_lr": adam_lr,
            "adam_eps": adam_eps,
            "grad_scaler": False,
            "loss_scaling": False,
            "rescale_derivative": False,
            "precision_switching": False,
            "fp64": False,

            # Same external stopping rule as our method.
            "early_stopping": True,
            "early_stop_min_delta": 1e-7,
            "early_stop_after_step": 100,
            "early_stop_patience": 50,
        },
        reinit=True,
    )

    # Same model structure as our dynamic-precision method.
    # Parameters remain FP32; do not call model.half().
    model = init_model(
        model_name,
        hidden_dim=512,
        num_layer=4,
        dtype=master_dtype,
    )
    print(model)

    # Same data generation as our dynamic-precision method.
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = get_data(
        [0, 2 * np.pi],
        [0, 1],
        101,
        101,
    )

    # Same preprocessing as our method.
    res_np, b_left_np, b_right_np, b_upper_np, b_lower_np = preprocess_data_for_model(
        model_name,
        res_np,
        b_left_np,
        b_right_np,
        b_upper_np,
        b_lower_np,
    )

    # Training tensors.
    # On CUDA, use FP16 inputs to mirror their DeepXDE default_float("float16")
    # and data-casting behavior in the repository.
    res = torch.tensor(res_np, dtype=train_dtype, device=device).requires_grad_(True)
    b_left = torch.tensor(b_left_np, dtype=train_dtype, device=device).requires_grad_(True)
    b_right = torch.tensor(b_right_np, dtype=train_dtype, device=device).requires_grad_(True)
    b_upper = torch.tensor(b_upper_np, dtype=train_dtype, device=device).requires_grad_(True)
    b_lower = torch.tensor(b_lower_np, dtype=train_dtype, device=device).requires_grad_(True)

    x_res, t_res = res[..., 0:1], res[..., 1:2]
    x_left, t_left = b_left[..., 0:1], b_left[..., 1:2]
    x_right, t_right = b_right[..., 0:1], b_right[..., 1:2]
    x_upper, t_upper = b_upper[..., 0:1], b_upper[..., 1:2]
    x_lower, t_lower = b_lower[..., 0:1], b_lower[..., 1:2]

    # Separate FP32 evaluation tensors.
    # This preserves the same sample locations but avoids making the reported
    # rRMSE/rMAE depend on AMP evaluation precision.
    res_eval = torch.tensor(res_np, dtype=eval_dtype, device=device)
    x_eval_all, t_eval_all = res_eval[..., 0:1], res_eval[..., 1:2]

    # Hayford repo-style optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=adam_lr,
        eps=adam_eps,
    )

    def compute_losses():
        """
        Mixed-precision training path.
            - FP32 model variables
            - FP16 training inputs on CUDA
            - FP16 compute through autocast
        """
        with torch.autocast(
                device_type=autocast_device,
                dtype=torch.float16,
                enabled=amp_enabled,
        ):
            pred_res = model(x_res, t_res)
            pred_left = model(x_left, t_left)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            u_t = torch.autograd.grad(
                pred_res,
                t_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=True,
            )[0]

            u_x = torch.autograd.grad(
                pred_res,
                x_res,
                grad_outputs=torch.ones_like(pred_res),
                retain_graph=True,
                create_graph=True,
            )[0]

            loss_res = torch.mean((u_t + beta_used * u_x) ** 2)

            loss_bc = torch.mean(
                (pred_upper - pred_lower) ** 2
            )

            loss_ic = torch.mean(
                (pred_left[..., 0] - torch.sin(x_left[..., 0])) ** 2
            )

            loss = loss_res + loss_bc + loss_ic

        return loss, loss_res, loss_ic, loss_bc

    stats = {}
    best_loss = float("inf")
    stagnant_steps = 0
    start_time = time.time()

    model.train()

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        loss, loss_res, loss_ic, loss_bc = compute_losses()

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"Invalid loss value at step {step}: {loss.detach()}"
            )

        loss.backward()
        optimizer.step()

        stats["loss"] = float(loss.detach())
        stats["loss_res"] = float(loss_res.detach())
        stats["loss_ic"] = float(loss_ic.detach())
        stats["loss_bc"] = float(loss_bc.detach())

        if step % 10 == 0:
            was_training = model.training
            model.eval()

            with torch.no_grad():
                # Same evaluation sampling logic as our dynamic-precision method.
                x_eval = x_eval_all[:, -1, :] if x_eval_all.dim() == 3 else x_eval_all
                t_eval = t_eval_all[:, -1, :] if t_eval_all.dim() == 3 else t_eval_all

                u_exact = torch.sin(x_eval - beta_used * t_eval)

                u_pred = model(x_eval_all, t_eval_all)

                if u_pred.dim() == 3:
                    u_pred = u_pred[:, -1, :]

                u_pred = u_pred.float()

                rRMSE = torch.sqrt(
                    torch.sum((u_exact - u_pred) ** 2) /
                    torch.sum(u_exact ** 2)
                ).item()

                rMAE = (
                    torch.sum(torch.abs(u_exact - u_pred)) /
                    torch.sum(torch.abs(u_exact))
                ).item()

            if was_training:
                model.train()

            print(
                f"Step {step} | "
                f"Loss {stats['loss']:.3e} | "
                f"rRMSE {rRMSE:.3e} | "
                f"rMAE {rMAE:.3e} | "
                f"state Hayford2024 repo-style mixed_float16 Adam"
            )

            wandb.log({
                "loss": stats["loss"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "precision_fp64": 0,
                "precision_fp32_master": 1,
                "train_data_fp16": 1 if train_dtype == torch.float16 else 0,
                "amp_autocast": 1 if amp_enabled else 0,
                "loss_scaling": 0,
                "rRMSE": rRMSE,
                "rMAE": rMAE,
            }, step=step)

        # Same early stopping rule as our dynamic-precision method.
        if stats["loss"] < best_loss - 1e-7:
            best_loss = stats["loss"]
            stagnant_steps = 0
        elif step > 100:
            stagnant_steps += 1

        if stagnant_steps > 50:
            print(
                f"Converged at step {step}: "
                f"Loss has not improved by 1e-7 for more than 50 epochs."
            )
            break

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")

    wandb.log({
        "training_time_seconds": duration,
        "final_step": step,
        "best_loss": best_loss,
    })

    plot_convection_results(
        model,
        beta_used,
        run_name,
        strategy,
        model_name,
    )

    wandb.finish()

    return model


def train_brdr_adaptive_sampling_fixed_beta_50(
        seed=1234,
        BETA_MAX=50,
        MAX_STEPS=100000,
        model_name: str = "PINN",
):
    """BRDR weighting + residual-based adaptive sampling for convection.

    This implements Algorithm 1 and Eqs. (8)--(16) of Chen, Howard and
    Stinis, "Self-adaptive weighting and sampling for physics-informed
    neural networks", applied to this project's beta=50 convection problem.

    The adaptive training procedure follows the paper:
      * FP32 Adam training;
      * BRDR pointwise weights and adaptive global scale;
      * beta_c = beta_w = 0.999 and epsilon = 1e-14;
      * EMA initialization-bias correction from the referenced BRDR method;
      * residual-proportional sampling after median clipping (gamma=100);
      * unconditional replacement of p_u=0.2 of PDE points every N_s=100
        optimization steps;
      * random removal of old PDE points;
      * inverse-distance (1/r) interpolation of both the BRDR weights and
        the EMA(R^4) state for newly selected points;
      * a fixed MAX_STEPS training budget (no early stopping), as in the
        paper's Algorithm 1.

    learning-rate schedule:
        eta_t = 1e-3 * 0.99 ** (t / 100).

    """
    if model_name != "PINN":
        raise ValueError(
            "This controlled comparator keeps the manuscript's standard "
            "PINN architecture; use model_name='PINN'."
        )

    #BRDR + adaptive-sampling settings.
    BETA_C = 0.999
    BETA_W = 0.999
    EPSILON = 1e-14
    CLIPPING_GAMMA = 100.0
    UPDATE_RATIO = 0.20
    UPDATE_EVERY = 100

    ADAM_LR_INITIAL = 1e-3
    ADAM_LR_DECAY = 0.99
    ADAM_LR_DECAY_STEPS = 100.0

    # The paper does not specify the candidate-pool cardinality.  The
    # released ET-PINN implementation uses a candidate/sample ratio of 1.
    CANDIDATE_RATIO = 1.0

    IDW_CHUNK_SIZE = 256

    set_seed(seed)
    current_dtype = torch.float32
    beta_used = float(BETA_MAX)
    strategy = "chen2026_BRDR_plus_residual_adaptive_sampling"
    run_name = (
        f"convection_BRDR_plus_SA_fixedbeta_{BETA_MAX}_"
        f"fp32_seed{seed}"
    )

    print("\n" + "=" * 60)
    print(f"Starting paper BRDR + adaptive-sampling run: {run_name}")
    print("=" * 60)

    wandb.init(
        project="pinn_convection_adaptive_weights".format(model_name),
        name=run_name,
        config={
            "strategy": strategy,
            "seed": seed,
            "beta": BETA_MAX,
            "precision": "fp32",
            "optimizer": "Adam",
            "model_name": model_name,
            "hidden_dim": 512,
            "num_layer": 4,
            "beta_c": BETA_C,
            "beta_w": BETA_W,
            "epsilon": EPSILON,
            "adaptive_scale": True,
            "adam_lr_initial": ADAM_LR_INITIAL,
            "adam_lr_schedule": "0.001 * 0.99 ** (step / 100)",
            "adam_lr_decay": ADAM_LR_DECAY,
            "adam_lr_decay_steps": ADAM_LR_DECAY_STEPS,
            "update_every": UPDATE_EVERY,
            "update_ratio": UPDATE_RATIO,
            "clipping_gamma": CLIPPING_GAMMA,
            "candidate_ratio": CANDIDATE_RATIO,
            "idw_power": 1.0,
            "fixed_training_steps": MAX_STEPS,
            "early_stopping": False,
        },
        reinit=True,
    )

    model = init_model(
        model_name="PINN",
        hidden_dim=512,
        num_layer=4,
        dtype=current_dtype,
    )
    model.train()
    print(model)

    res_np, b_left_np, _b_right_np, b_upper_np, b_lower_np = get_data(
        [0, 2 * np.pi], [0, 1], 101, 101
    )

    residual_points = torch.tensor(
        res_np, dtype=current_dtype, device=device
    ).requires_grad_(True)
    ic_points = torch.tensor(
        b_left_np, dtype=current_dtype, device=device
    ).requires_grad_(True)
    bc_x2pi_points = torch.tensor(
        b_upper_np, dtype=current_dtype, device=device
    ).requires_grad_(True)
    bc_x0_points = torch.tensor(
        b_lower_np, dtype=current_dtype, device=device
    ).requires_grad_(True)

    coordinate_atol = 10.0 * torch.finfo(current_dtype).eps
    if not torch.allclose(
        ic_points[:, 1],
        torch.zeros_like(ic_points[:, 1]),
        atol=coordinate_atol,
        rtol=0.0,
    ):
        raise RuntimeError("get_data mapping error: b_left is not t=0.")
    if not torch.allclose(
        bc_x2pi_points[:, 0],
        torch.full_like(bc_x2pi_points[:, 0], 2.0 * torch.pi),
        atol=coordinate_atol,
        rtol=0.0,
    ):
        raise RuntimeError("get_data mapping error: b_upper is not x=2*pi.")
    if not torch.allclose(
        bc_x0_points[:, 0],
        torch.zeros_like(bc_x0_points[:, 0]),
        atol=coordinate_atol,
        rtol=0.0,
    ):
        raise RuntimeError("get_data mapping error: b_lower is not x=0.")

    eval_points = torch.tensor(res_np, dtype=current_dtype, device=device)
    x_eval = eval_points[:, 0:1]
    t_eval = eval_points[:, 1:2]
    u_exact_eval = torch.sin(x_eval - beta_used * t_eval)

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR_INITIAL)

    # BRDR initialization: w=s=1 and EMA(R^4)=0.
    w_res = torch.ones(
        (residual_points.shape[0], 1), dtype=current_dtype, device=device
    )
    w_bc = torch.ones(
        (bc_x0_points.shape[0], 1), dtype=current_dtype, device=device
    )
    w_ic = torch.ones(
        (ic_points.shape[0], 1), dtype=current_dtype, device=device
    )
    exp4_res = torch.zeros_like(w_res)
    exp4_bc = torch.zeros_like(w_bc)
    exp4_ic = torch.zeros_like(w_ic)
    scale_s = torch.ones((), dtype=current_dtype, device=device)

    def _learning_rate(step: int) -> float:
        return ADAM_LR_INITIAL * (
            ADAM_LR_DECAY ** (float(step) / ADAM_LR_DECAY_STEPS)
        )

    def _split(points):
        return points[:, 0:1], points[:, 1:2]

    def _convection_residual(points, create_graph: bool):
        x, t = _split(points)
        pred = model(x, t)
        u_t = torch.autograd.grad(
            pred,
            t,
            grad_outputs=torch.ones_like(pred),
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        u_x = torch.autograd.grad(
            pred,
            x,
            grad_outputs=torch.ones_like(pred),
            retain_graph=create_graph,
            create_graph=create_graph,
        )[0]
        return u_t + beta_used * u_x

    def _all_raw_residuals(create_graph: bool):
        R_res = _convection_residual(
            residual_points, create_graph=create_graph
        )

        x_ic, t_ic = _split(ic_points)
        x_bc2pi, t_bc2pi = _split(bc_x2pi_points)
        x_bc0, t_bc0 = _split(bc_x0_points)

        R_ic = model(x_ic, t_ic) - torch.sin(x_ic)
        R_bc = model(x_bc2pi, t_bc2pi) - model(x_bc0, t_bc0)
        return R_res, R_bc, R_ic

    def _update_brdr_states(R_res, R_bc, R_ic, step: int):
        nonlocal w_res, w_bc, w_ic, exp4_res, exp4_bc, exp4_ic

        with torch.no_grad():
            R2_res = R_res.detach().square()
            R2_bc = R_bc.detach().square()
            R2_ic = R_ic.detach().square()

            # Eq. (9): EMA of R^4.
            exp4_res.mul_(BETA_C).add_(
                (1.0 - BETA_C) * R2_res.square()
            )
            exp4_bc.mul_(BETA_C).add_(
                (1.0 - BETA_C) * R2_bc.square()
            )
            exp4_ic.mul_(BETA_C).add_(
                (1.0 - BETA_C) * R2_ic.square()
            )

            # The referenced BRDR algorithm corrects the zero-initialized EMA:
            # IRDR = R^2 / sqrt(EMA(R^4)/(1-beta_c^step) + epsilon).
            ema_denominator = 1.0 - (BETA_C ** step)
            irdr_res = R2_res / torch.sqrt(
                exp4_res / ema_denominator + EPSILON
            )
            irdr_bc = R2_bc / torch.sqrt(
                exp4_bc / ema_denominator + EPSILON
            )
            irdr_ic = R2_ic / torch.sqrt(
                exp4_ic / ema_denominator + EPSILON
            )

            # Eq. (10): one mean over every training residual item.
            total_irdr = irdr_res.sum() + irdr_bc.sum() + irdr_ic.sum()
            total_count = (
                irdr_res.numel() + irdr_bc.numel() + irdr_ic.numel()
            )
            global_irdr_mean = total_irdr / total_count

            # Eq. (11): smoothed pointwise weights.
            w_res.mul_(BETA_W).add_(
                (1.0 - BETA_W) * (irdr_res / global_irdr_mean)
            )
            w_bc.mul_(BETA_W).add_(
                (1.0 - BETA_W) * (irdr_bc / global_irdr_mean)
            )
            w_ic.mul_(BETA_W).add_(
                (1.0 - BETA_W) * (irdr_ic / global_irdr_mean)
            )

            return (
                float(global_irdr_mean.detach()),
                float(w_res.mean().detach()),
                float(w_bc.mean().detach()),
                float(w_ic.mean().detach()),
            )

    def _sampling_probabilities(candidate_squared_residual):
        values = candidate_squared_residual.detach().reshape(-1)
        if not torch.all(torch.isfinite(values)):
            raise FloatingPointError(
                "Non-finite candidate PDE residual during adaptive sampling."
            )

        # Eqs. (14)--(15).
        residual2_median = torch.median(values)
        clipped = torch.maximum(
            residual2_median,
            torch.minimum(
                values, CLIPPING_GAMMA * residual2_median
            ),
        )
        normalizer = clipped.sum()
        if not torch.isfinite(normalizer) or normalizer.item() <= 0.0:
            raise FloatingPointError(
                "Paper-defined sampling probabilities are undefined: "
                "the clipped squared-residual sum is non-positive/non-finite."
            )
        return clipped / normalizer, residual2_median

    def _idw_interpolate(new_points, old_points, old_values):
        """Eq. (16), evaluated in chunks to control peak memory."""
        outputs = []
        old_points = old_points.detach()
        old_values = old_values.detach()

        for start in range(0, new_points.shape[0], IDW_CHUNK_SIZE):
            stop = min(start + IDW_CHUNK_SIZE, new_points.shape[0])
            distance = torch.cdist(
                new_points[start:stop], old_points, p=2
            )
            # EPSILON only resolves the undefined 1/0 case for duplicate
            # coordinates; otherwise this is exactly the paper's 1/r IDW.
            inverse_distance = 1.0 / (distance + EPSILON)
            outputs.append(
                (inverse_distance @ old_values)
                / inverse_distance.sum(dim=1, keepdim=True)
            )

        return torch.cat(outputs, dim=0)

    def _refine_residual_points(step: int):
        nonlocal residual_points, w_res, exp4_res

        diagnostics = {
            "sampling_update_event": 0,
            "num_points_replaced": 0,
            "candidate_residual2_mean": float("nan"),
            "candidate_residual2_max": float("nan"),
            "candidate_residual2_median": float("nan"),
            "sampling_probability_max": float("nan"),
            "sampling_time_seconds_step": 0.0,
        }

        if step % UPDATE_EVERY != 0:
            return diagnostics

        if device.type == "cuda":
            torch.cuda.synchronize()
        sampling_start = time.perf_counter()

        n_residual = residual_points.shape[0]
        n_candidate = int(n_residual * CANDIDATE_RATIO)
        n_replace = int(n_residual * UPDATE_RATIO)
        n_keep = n_residual - n_replace

        if n_candidate < n_replace or n_replace <= 0:
            raise ValueError(
                "The candidate pool must contain at least the requested "
                "number of replacement points."
            )

        # Uniform random candidates in the convection space-time domain.
        candidate_points = torch.cat(
            (
                2.0 * torch.pi * torch.rand(
                    (n_candidate, 1),
                    dtype=current_dtype,
                    device=device,
                ),
                torch.rand(
                    (n_candidate, 1),
                    dtype=current_dtype,
                    device=device,
                ),
            ),
            dim=1,
        ).requires_grad_(True)

        candidate_residual = _convection_residual(
            candidate_points, create_graph=False
        )
        candidate_residual2 = candidate_residual.detach().square().reshape(-1)
        probabilities, residual2_median = _sampling_probabilities(
            candidate_residual2
        )

        # Algorithm 1: uniformly select old points for replacement.
        old_points = residual_points.detach()
        old_w_res = w_res.detach()
        old_exp4_res = exp4_res.detach()
        keep_indices = torch.randperm(n_residual, device=device)[:n_keep]

        retained_points = old_points[keep_indices]
        retained_w_res = old_w_res[keep_indices]
        retained_exp4_res = old_exp4_res[keep_indices]

        # Select p_u |X| new points from the residual-based distribution.
        selected_candidate_indices = torch.multinomial(
            probabilities,
            num_samples=n_replace,
            replacement=False,
        )
        new_points = candidate_points.detach()[selected_candidate_indices]

        # Algorithm 1 and Eq. (16): transfer both pointwise states.
        new_w_res = _idw_interpolate(
            new_points, old_points, old_w_res
        )
        new_exp4_res = _idw_interpolate(
            new_points, old_points, old_exp4_res
        )

        residual_points = torch.cat(
            (retained_points, new_points), dim=0
        ).requires_grad_(True)
        w_res = torch.cat(
            (retained_w_res, new_w_res), dim=0
        ).detach()
        exp4_res = torch.cat(
            (retained_exp4_res, new_exp4_res), dim=0
        ).detach()

        diagnostics.update({
            "sampling_update_event": 1,
            "num_points_replaced": int(n_replace),
            "candidate_residual2_mean": float(
                candidate_residual2.mean().detach()
            ),
            "candidate_residual2_max": float(
                candidate_residual2.max().detach()
            ),
            "candidate_residual2_median": float(
                residual2_median.detach()
            ),
            "sampling_probability_max": float(
                probabilities.max().detach()
            ),
        })

        del candidate_points, candidate_residual, candidate_residual2
        if device.type == "cuda":
            torch.cuda.synchronize()
        diagnostics["sampling_time_seconds_step"] = (
            time.perf_counter() - sampling_start
        )
        return diagnostics

    def _evaluate_fixed_grid():
        model.eval()
        with torch.no_grad():
            u_pred = model(x_eval, t_eval)
            rRMSE = torch.sqrt(
                torch.sum((u_exact_eval - u_pred).square())
                / torch.sum(u_exact_eval.square())
            ).item()
            rMAE = (
                torch.sum(torch.abs(u_exact_eval - u_pred))
                / torch.sum(torch.abs(u_exact_eval))
            ).item()
        model.train()
        return rRMSE, rMAE

    stats = {}
    sampling_updates_total = 0
    points_replaced_total = 0
    sampling_time_total = 0.0
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        # Paper Table 1 Burgers schedule; update before this optimization step.
        current_lr = _learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # Algorithm 1, lines 3--7: update BRDR state at current points.
        R_res, R_bc, R_ic = _all_raw_residuals(create_graph=True)
        (
            irdr_mean,
            w_res_mean,
            w_bc_mean,
            w_ic_mean,
        ) = _update_brdr_states(R_res, R_bc, R_ic, step)

        # Algorithm 1, lines 8--19: unconditional scheduled resampling.
        sampling_diag = _refine_residual_points(step)
        sampling_updates_total += sampling_diag["sampling_update_event"]
        points_replaced_total += sampling_diag["num_points_replaced"]
        sampling_time_total += sampling_diag["sampling_time_seconds_step"]

        if sampling_diag["sampling_update_event"]:
            # Assemble the loss on X_t, not the pre-resampling X_{t-1}.
            R_res, R_bc, R_ic = _all_raw_residuals(create_graph=True)
            w_res_mean = float(w_res.mean().detach())

        # Eq. (6), with IC treated as a boundary-condition component and all
        # user-defined component constants alpha set to 1 (plain BRDR).
        loss_res = torch.mean(w_res * R_res.square())
        loss_bc = torch.mean(w_bc * R_bc.square())
        loss_ic = torch.mean(w_ic * R_ic.square())
        loss_unscaled = loss_res + loss_bc + loss_ic
        loss = scale_s * loss_unscaled

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite BRDR+sampling loss at step {step}: "
                f"{float(loss.detach())}"
            )

        loss.backward()

        # Referenced BRDR Algorithm 1: update s with beta_s = 1 - eta,
        # then correct the already-computed gradients by s_new / s_old.
        with torch.no_grad():
            grad_norm_sq = torch.zeros(
                (), dtype=current_dtype, device=device
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_norm_sq.add_(parameter.grad.square().sum())

            old_scale_s = scale_s.clone()
            scale_s = (
                (1.0 - current_lr) * old_scale_s
                + 2.0 * old_scale_s * loss.detach() / grad_norm_sq
            )

            if not torch.isfinite(scale_s):
                raise FloatingPointError(
                    f"Non-finite BRDR global scale at step {step}."
                )

            gradient_correction = scale_s / old_scale_s
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_correction)

        optimizer.step()

        stats["loss"] = float(loss.detach())
        stats["loss_unscaled"] = float(loss_unscaled.detach())
        stats["loss_res"] = float(loss_res.detach())
        stats["loss_ic"] = float(loss_ic.detach())
        stats["loss_bc"] = float(loss_bc.detach())
        stats["scale_s"] = float(scale_s.detach())
        stats["w_res_mean"] = w_res_mean
        stats["w_bc_mean"] = w_bc_mean
        stats["w_ic_mean"] = w_ic_mean
        stats["irdr_mean"] = irdr_mean

        if sampling_diag["sampling_update_event"]:
            print(
                f"Adaptive sampling at step {step}: replaced "
                f"{sampling_diag['num_points_replaced']} / "
                f"{residual_points.shape[0]} PDE points."
            )

        if step % 10 == 0:
            rRMSE, rMAE = _evaluate_fixed_grid()
            print(
                f"Step {step} | Loss {stats['loss']:.3e} | "
                f"Unscaled {stats['loss_unscaled']:.3e} | "
                f"rRMSE {rRMSE:.3e} | rMAE {rMAE:.3e} | "
                f"s {stats['scale_s']:.3e} | lr {current_lr:.3e} | "
                f"w_mean(res/bc/ic) "
                f"{w_res_mean:.2e}/{w_bc_mean:.2e}/{w_ic_mean:.2e}"
            )

            wandb.log({
                "loss": stats["loss"],
                "loss_unscaled": stats["loss_unscaled"],
                "loss_res": stats["loss_res"],
                "loss_ic": stats["loss_ic"],
                "loss_bc": stats["loss_bc"],
                "scale_s": stats["scale_s"],
                "irdr_mean": stats["irdr_mean"],
                "w_res_mean": stats["w_res_mean"],
                "w_bc_mean": stats["w_bc_mean"],
                "w_ic_mean": stats["w_ic_mean"],
                "rRMSE": rRMSE,
                "rMAE": rMAE,
                "precision_fp64": 0,
                # Additional sampling diagnostics; original keys above are
                # neither removed nor renamed.
                "adam_lr_current": current_lr,
                "sampling_update_event": sampling_diag[
                    "sampling_update_event"
                ],
                "num_points_replaced": sampling_diag[
                    "num_points_replaced"
                ],
                "candidate_residual2_mean": sampling_diag[
                    "candidate_residual2_mean"
                ],
                "candidate_residual2_max": sampling_diag[
                    "candidate_residual2_max"
                ],
                "candidate_residual2_median": sampling_diag[
                    "candidate_residual2_median"
                ],
                "sampling_probability_max": sampling_diag[
                    "sampling_probability_max"
                ],
                "sampling_time_seconds_step": sampling_diag[
                    "sampling_time_seconds_step"
                ],
                "sampling_updates_cumulative": sampling_updates_total,
                "points_replaced_cumulative": points_replaced_total,
                "sampling_time_seconds_cumulative": sampling_time_total,
                "num_residual_points": int(residual_points.shape[0]),
            })

    duration = time.time() - start_time
    final_rRMSE, final_rMAE = _evaluate_fixed_grid()

    print(f"Training finished in {duration:.2f} seconds.")
    print(
        f"Final fixed-grid rRMSE={final_rRMSE:.6e}, "
        f"rMAE={final_rMAE:.6e}; sampling updates={sampling_updates_total}."
    )

    wandb.log({
        "training_time_seconds": duration,
        "final_rRMSE": final_rRMSE,
        "final_rMAE": final_rMAE,
        "sampling_updates_total": sampling_updates_total,
        "points_replaced_total": points_replaced_total,
        "sampling_time_seconds_total": sampling_time_total,
    })

    plot_convection_results(model, beta_used, run_name, strategy, model_name)
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



def plot_ns2d_c_results(
        model,
        run_name,
        strategy,
        model_name='PINN',
        ns_data=None,
        levels=60,
):
    """Plot PINNacle NS2d-C reference, prediction, and absolute error.

    The reported global L2RE/L1RE exactly follow PINNacle's TesterCallback:
    errors and reference magnitudes are averaged jointly over all reference
    points and all three outputs (u, v, p), after removing NaN rows only.
    """
    print("Generating PINNacle NS2d-C result plots...")
    if ns_data is None:
        ns_data = get_ns2d_c_data()

    dtype = next(model.parameters()).dtype
    x_exact = np.asarray(ns_data["eval_x"])
    uvp_exact = np.asarray(ns_data["eval_uvp"])

    with torch.no_grad():
        x_tensor = torch.tensor(x_exact, dtype=dtype, device=device)
        uvp_pred = model(x_tensor[:, 0:1], x_tensor[:, 1:2])
        if uvp_pred.dim() == 3:
            uvp_pred = uvp_pred[:, -1, :]
        uvp_pred = uvp_pred.detach().cpu().numpy()

    valid_rows = ~np.isnan(
        np.concatenate([x_exact, uvp_exact, uvp_pred], axis=1)
    ).any(axis=1)
    x_valid = x_exact[valid_rows]
    exact = uvp_exact[valid_rows]
    pred = uvp_pred[valid_rows]
    error = pred - exact

    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    solution_l2 = np.sqrt(np.mean(exact ** 2))
    solution_l1 = np.mean(np.abs(exact))
    l2re = np.sqrt(mse) / solution_l2
    l1re = mae / solution_l1
    rRMSE = np.sqrt(np.sum(error ** 2) / np.sum(exact ** 2))
    rMAE = np.sum(np.abs(error)) / np.sum(np.abs(exact))

    component_l2re = np.sqrt(np.mean(error ** 2, axis=0)) / np.sqrt(
        np.mean(exact ** 2, axis=0)
    )
    component_rmae = np.sum(np.abs(error), axis=0) / np.sum(np.abs(exact), axis=0)

    print(
        f"rRMSE: {rRMSE:.4e}, rMAE: {rMAE:.4e}; "
        f"PINNacle L2RE: {l2re:.4e}, L1RE: {l1re:.4e}; "
        f"component rRMSE u/v/p: {component_l2re[0]:.4e}/"
        f"{component_l2re[1]:.4e}/{component_l2re[2]:.4e}"
    )

    if wandb.run is not None:
        wandb.run.summary["L2RE"] = float(l2re)
        wandb.run.summary["L1RE"] = float(l1re)
        wandb.run.summary["u_L2RE"] = float(component_l2re[0])
        wandb.run.summary["v_L2RE"] = float(component_l2re[1])
        wandb.run.summary["p_L2RE"] = float(component_l2re[2])
        # Same names as the other PDE trainers in this script.
        wandb.run.summary["rRMSE"] = float(rRMSE)
        wandb.run.summary["rMAE"] = float(rMAE)
        wandb.run.summary["RMAE"] = float(rMAE)
        wandb.run.summary["u_rRMSE"] = float(component_l2re[0])
        wandb.run.summary["v_rRMSE"] = float(component_l2re[1])
        wandb.run.summary["p_rRMSE"] = float(component_l2re[2])
        wandb.run.summary["u_rMAE"] = float(component_rmae[0])
        wandb.run.summary["v_rMAE"] = float(component_rmae[1])
        wandb.run.summary["p_rMAE"] = float(component_rmae[2])

    import matplotlib.tri as mtri
    triangulation = mtri.Triangulation(x_valid[:, 0], x_valid[:, 1])
    names = ["u", "v", "p"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    for row, name in enumerate(names):
        exact_values = exact[:, row]
        pred_values = pred[:, row]
        abs_error = np.abs(error[:, row])

        field_min = min(np.nanmin(exact_values), np.nanmin(pred_values))
        field_max = max(np.nanmax(exact_values), np.nanmax(pred_values))
        if not np.isfinite(field_min) or not np.isfinite(field_max) or field_min == field_max:
            field_min, field_max = -1.0, 1.0

        # Match the convection plot convention: viridis for reference/prediction,
        # coolwarm for error.
        plot_exact = axes[row, 0].tricontourf(
            triangulation, exact_values, levels=levels,
            vmin=field_min, vmax=field_max, cmap="viridis",
        )
        plot_pred = axes[row, 1].tricontourf(
            triangulation, pred_values, levels=levels,
            vmin=field_min, vmax=field_max, cmap="viridis",
        )
        plot_error = axes[row, 2].tricontourf(
            triangulation, abs_error, levels=levels, cmap="coolwarm",
        )

        axes[row, 0].set_title(f"Reference {name}")
        axes[row, 1].set_title(
            f"Predicted {name} (rRMSE={component_l2re[row]:.3e})"
        )
        axes[row, 2].set_title(f"Absolute error in {name}")

        fig.colorbar(plot_exact, ax=axes[row, 0])
        fig.colorbar(plot_pred, ax=axes[row, 1])
        fig.colorbar(plot_error, ax=axes[row, 2])

        for col in range(3):
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlim(ns_data["bbox"][0], ns_data["bbox"][1])
            axes[row, col].set_ylim(ns_data["bbox"][2], ns_data["bbox"][3])

    fig.suptitle(
        f"PINNacle NS2d-C ({ns_data.get('lid_tag', 'a?')}): {strategy}; global rRMSE={rRMSE:.4e}",
        fontsize=16,
    )
    filename = f"ns2d_c_{run_name}_{strategy}.png"
    plt.savefig(filename, dpi=250, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({"Result Plot": wandb.Image(filename)})
    plt.close(fig)

    prediction_filename = f"ns2d_c_predictions_{run_name}_{strategy}.npz"
    np.savez_compressed(
        prediction_filename,
        x=x_valid,
        reference=exact,
        prediction=pred,
        absolute_error=np.abs(error),
        L2RE=np.asarray(l2re),
        L1RE=np.asarray(l1re),
        rRMSE=np.asarray(rRMSE),
        rMAE=np.asarray(rMAE),
        component_L2RE=component_l2re,
        component_rRMSE=component_l2re,
        component_rMAE=component_rmae,
        lid_amplitude=np.asarray(ns_data.get("lid_amplitude", np.nan)),
        lid_tag=np.asarray(ns_data.get("lid_tag", "unknown")),
        datapath=np.asarray(ns_data.get("datapath", "unknown")),
    )
    print(f"Plot saved to {filename}")
    print(f"Predictions saved to {prediction_filename}")

def plot_heat10d_results(model, D, run_name, strategy, model_name='PINN', seed=1234):
    print("Generating Heat equation results plot...")
    q = 101
    dtype = next(model.parameters()).dtype

    def _predict_grid(x_grid, t_grid, shape):
        x_flat = x_grid.reshape(-1, D)
        t_flat = t_grid.reshape(-1, 1)
        u_exact = heat10d_exact_np(x_flat, t_flat, d=D).reshape(shape)

        # model.eval()
        with torch.no_grad():
            ev_data = np.concatenate((x_flat, t_flat), axis=-1)
            if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only', 'PINNMamba']:
                ev_data = make_time_sequence(ev_data, num_step=num_step, step=step_size)
                x_t = torch.tensor(ev_data[..., :D], dtype=dtype, device=device)
                t_t = torch.tensor(ev_data[..., D:D + 1], dtype=dtype, device=device)
            else:
                x_t = torch.tensor(x_flat, dtype=dtype, device=device)
                t_t = torch.tensor(t_flat, dtype=dtype, device=device)

            u_pred = model(x_t, t_t)
            if u_pred.dim() == 3: u_pred = u_pred[:, -1, :]
            u_pred = u_pred.cpu().numpy().reshape(shape)
        return u_exact, u_pred

    vals = np.linspace(-1, 1, q)
    t_vals = np.linspace(0, 1, q)

    A, B = np.meshgrid(vals, vals, indexing="ij")
    x12 = np.zeros((q, q, D))
    x12[..., 0] = A
    x12[..., 1] = B
    t12 = np.full((q, q, 1), 0.5)
    u12, p12 = _predict_grid(x12, t12, (q, q))

    x56 = np.zeros((q, q, D))
    x56[..., 4] = A
    x56[..., 5] = B
    t56 = np.full((q, q, 1), 0.5)
    u56, p56 = _predict_grid(x56, t56, (q, q))

    X9, TT = np.meshgrid(vals, t_vals, indexing="ij")
    x9t = np.zeros((q, q, D))
    x9t[..., 8] = X9
    tt = TT[..., None]
    u9t, p9t = _predict_grid(x9t, tt, (q, q))

    panels = [
        (u12, p12, [-1, 1, -1, 1], "x1", "x2", "x1-x2 plane, t=0.5"),
        (u56, p56, [-1, 1, -1, 1], "x5", "x6", "x5-x6 plane, t=0.5"),
        (u9t, p9t, [-1, 1, 1, 0], "x9", "t", "x9-t plane"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for row, (u_exact, u_pred, extent, xlabel, ylabel, title) in enumerate(panels):
        error = np.abs(u_exact - u_pred)
        rRMSE = np.sqrt(np.sum((u_exact - u_pred) ** 2) / np.sum(u_exact ** 2))
        rMAE = np.sum(np.abs(u_exact - u_pred)) / np.sum(np.abs(u_exact))

        im0 = axes[row, 0].imshow(u_exact.T, extent=extent, aspect='auto', cmap='viridis', origin="upper")
        axes[row, 0].set_title(f"Exact Solution ({title})")
        axes[row, 0].set_xlabel(xlabel); axes[row, 0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(u_pred.T, extent=extent, aspect='auto', cmap='viridis', origin="upper")
        axes[row, 1].set_title(f"Prediction (rRMSE={rRMSE:.4e})")
        axes[row, 1].set_xlabel(xlabel); axes[row, 1].set_ylabel(ylabel)
        plt.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(error.T, extent=extent, aspect='auto', origin="upper",cmap='coolwarm',vmin=0.0,vmax=0.0015)
        axes[row, 2].set_title(f"Absolute Error (rMAE={rMAE:.4e})")
        axes[row, 2].set_xlabel(xlabel); axes[row, 2].set_ylabel(ylabel)
        plt.colorbar(im2, ax=axes[row, 2])

    np.random.seed(seed)
    test_np = get_heat10d_test_data(d=D, Nbc_v=100, Nin_v=7000, T=1.0)
    x_test_np = test_np[:, :D]
    t_test_np = test_np[:, D:D + 1]
    u_exact_test = heat10d_exact_np(x_test_np, t_test_np, d=D)

    # model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_test_np, dtype=dtype, device=device)
        t_t = torch.tensor(t_test_np, dtype=dtype, device=device)
        u_pred_test = model(x_t, t_t)
        if u_pred_test.dim() == 3: u_pred_test = u_pred_test[:, -1, :]
        u_pred_test = u_pred_test.cpu().numpy()

    rRMSE = np.sqrt(np.sum((u_exact_test - u_pred_test) ** 2) / np.sum(u_exact_test ** 2))
    rMAE = np.sum(np.abs(u_exact_test - u_pred_test)) / np.sum(np.abs(u_exact_test))

    print('-' * 40)
    print(f'Relative MAE (rMAE): {rMAE:4f}')
    print(f'Relative L2 Error (rRMSE): {rRMSE:4f}')
    print('-' * 40)

    wandb.run.summary["relative_l1_error"] = rMAE
    wandb.run.summary["rMAE"] = rMAE
    wandb.run.summary["rRMSE"] = rRMSE

    plt.tight_layout()
    filename = f"result_heat{D}d_{run_name}_{strategy}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    wandb.log({"Result Plot": wandb.Image(filename)})
    plt.close(fig)
    print(f"Plot saved to {filename}")


def main():
    import os

    for seed in [0, 1, 2, 3, 4]:

        train_dynamic_precision_fixed_beta_50_curvature(seed=seed)#dynamic
        train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp64')
        train_dynamic_precision_fixed_beta_50_curvature(seed=seed, benchmark='fp32')

        train_dynamic_precision_fixed_rho_curvature(seed=seed)
        train_dynamic_precision_fixed_rho_curvature(seed=seed,benchmark='fp64')
        train_dynamic_precision_fixed_rho_curvature(seed=seed, benchmark='fp32')

        train_dynamic_precision_wave_curvature(seed=seed)
        train_dynamic_precision_wave_curvature(seed=seed,benchmark='fp64')
        train_dynamic_precision_wave_curvature(seed=seed,benchmark='fp32')

        train_dynamic_precision_allen_cahn_curvature(seed=seed)
        train_dynamic_precision_allen_cahn_curvature(seed=seed, benchmark='fp32')
        train_dynamic_precision_allen_cahn_curvature(seed=seed,benchmark='fp64')
        
        train_dynamic_precision_ns2d_c_curvature(seed=seed,a=8)
        train_dynamic_precision_ns2d_c_curvature(seed=seed,a=8, benchmark='fp32')
        train_dynamic_precision_ns2d_c_curvature(seed=seed,a=8, benchmark='fp64')

        train_dynamic_precision_irradiance(seed=seed,benchmark='fp32')
        train_dynamic_precision_irradiance(seed=seed)
        train_dynamic_precision_irradiance(seed=seed, benchmark='fp64')

        #comparison with realted methods:
        train_hayford_2024_repo_mixed_precision_fixed_beta_50(seed=seed)
        train_brdr_fixed_beta_50(seed=seed)
        train_hayford_2024_repo_mixed_precision_fixed_beta_50(seed=seed)
        train_dynamic_precision_fixed_beta_50_curvature_ssbroyden2(seed=seed)


if __name__ == "__main__":
    main()
