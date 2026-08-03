
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains an implementation based on code from:
# Jorge F. Urbán, University of Alicante, and the SciPy Developers.
#
# The applicable copyright notices and BSD 3-Clause license text
# are provided in THIRD_PARTY_NOTICES.

"""
Dense PyTorch implementation of the SSBroyden2 self-scaled quasi-Newton
update based on https://github.com/jorgeurban/self_scaled_algorithms_pinns 

The implementation uses PyTorch's strong-Wolfe line-search routine so that
the model, gradients, inverse-Hessian state, and line search can all run in
the active PyTorch dtype and on the active device.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

try:
    # Private PyTorch API; available in current PyTorch LBFGS implementations.
    from torch.optim.lbfgs import _strong_wolfe
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DenseSSBroyden2 requires torch.optim.lbfgs._strong_wolfe. "
        "Use a PyTorch release that provides the standard LBFGS strong-Wolfe "
        "line search, or replace this import with an equivalent implementation."
    ) from exc


class DenseSSBroyden2(Optimizer):
    """
    Full-memory self-scaled Broyden quasi-Newton optimizer.

    Parameters
    ----------
    params:
        Iterable of trainable PyTorch parameters.
    lr:
        Initial line-search step scale.
    max_iter:
        Maximum quasi-Newton iterations performed by one ``step`` call.
    max_eval:
        Maximum closure evaluations per ``step`` call.
    tolerance_grad:
        Stop an inner solve when max(abs(gradient)) is below this value.
    tolerance_change:
        Stop when the parameter step or loss change is below this value.
    history_size:
        Number of accepted secant pairs retained only for the external
        curvature proxy. The dense inverse Hessian itself retains full history.
    line_search_fn:
        Must be ``"strong_wolfe"``.
    c1, c2:
        Strong-Wolfe line-search constants.
    max_inverse_hessian_bytes:
        Safety limit for the dense inverse-Hessian allocation.
    update_eps:
        Numerical safeguard for denominators in the SSBroyden2 update.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-8,
        tolerance_change: float = 1e-10,
        history_size: int = 10,
        line_search_fn: str = "strong_wolfe",
        c1: float = 1e-4,
        c2: float = 0.9,
        max_inverse_hessian_bytes: int = 12 * 1024**3,
        update_eps: float = 1e-20,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        if max_eval < 1:
            raise ValueError("max_eval must be at least 1.")
        if history_size < 2:
            raise ValueError("history_size must be at least 2.")
        if line_search_fn != "strong_wolfe":
            raise ValueError("Only line_search_fn='strong_wolfe' is supported.")
        if not (0.0 < c1 < c2 < 1.0):
            raise ValueError("Strong-Wolfe constants must satisfy 0 < c1 < c2 < 1.")
        if max_inverse_hessian_bytes <= 0:
            raise ValueError("max_inverse_hessian_bytes must be positive.")

        defaults = dict(
            lr=float(lr),
            max_iter=int(max_iter),
            max_eval=int(max_eval),
            tolerance_grad=float(tolerance_grad),
            tolerance_change=float(tolerance_change),
            history_size=int(history_size),
            line_search_fn=line_search_fn,
            c1=float(c1),
            c2=float(c2),
            update_eps=float(update_eps),
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("DenseSSBroyden2 supports exactly one parameter group.")

        self._params = self.param_groups[0]["params"]
        if len(self._params) == 0:
            raise ValueError("DenseSSBroyden2 received no parameters.")

        n_parameter = self._numel()
        element_size = self._params[0].element_size()
        required_bytes = n_parameter * n_parameter * element_size
        self.inverse_hessian_required_bytes = required_bytes

        if required_bytes > max_inverse_hessian_bytes:
            raise MemoryError(
                "DenseSSBroyden2 cannot be used with this network under the "
                f"configured safety limit. The model has {n_parameter:,} "
                "trainable parameters, so its dense inverse-Hessian state "
                f"requires approximately {required_bytes / 1024**3:.2f} GiB "
                f"in {self._params[0].dtype}. Reduce the network size or use "
                "a limited-memory quasi-Newton optimizer."
            )

    def _numel(self) -> int:
        return sum(parameter.numel() for parameter in self._params)

    def _gather_flat_grad(self) -> Tensor:
        views = []
        for parameter in self._params:
            if parameter.grad is None:
                view = torch.zeros_like(parameter).reshape(-1)
            elif parameter.grad.is_sparse:
                view = parameter.grad.to_dense().reshape(-1)
            else:
                view = parameter.grad.reshape(-1)
            views.append(view)
        return torch.cat(views)

    @torch.no_grad()
    def _clone_param(self) -> list[Tensor]:
        return [
            parameter.clone(memory_format=torch.contiguous_format)
            for parameter in self._params
        ]

    @torch.no_grad()
    def _set_param(self, parameter_data: list[Tensor]) -> None:
        for parameter, value in zip(self._params, parameter_data, strict=True):
            parameter.copy_(value)

    @torch.no_grad()
    def _add_flat_update(self, step_size: float, update: Tensor) -> None:
        offset = 0
        for parameter in self._params:
            numel = parameter.numel()
            parameter.add_(
                update[offset:offset + numel].view_as(parameter),
                alpha=float(step_size),
            )
            offset += numel

        if offset != update.numel():
            raise RuntimeError(
                f"Parameter-vector size mismatch: consumed {offset}, "
                f"but update contains {update.numel()} values."
            )

    def _directional_evaluate(
        self,
        closure: Callable[[], Tensor],
        x: list[Tensor],
        step_size: float,
        direction: Tensor,
    ) -> tuple[float, Tensor]:
        self._add_flat_update(step_size, direction)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @staticmethod
    def _standard_bfgs_inverse_update(
        inverse_hessian: Tensor,
        s: Tensor,
        y: Tensor,
        rho: Tensor,
        y_h_y: Tensor,
    ) -> Tensor:
        h_y = inverse_hessian.mv(y)
        updated = (
            inverse_hessian
            - rho * (torch.outer(h_y, s) + torch.outer(s, h_y))
            + rho * (1.0 + rho * y_h_y) * torch.outer(s, s)
        )
        return 0.5 * (updated + updated.T)

    def _ssbroyden2_inverse_update(
        self,
        inverse_hessian: Tensor,
        s: Tensor,
        y: Tensor,
        accepted_step_size: float,
        old_grad: Tensor,
        eps: float,
    ) -> tuple[Tensor, str]:
        """
        Apply the SSBroyden2 inverse-Hessian update.

        Returns
        -------
        updated_inverse_hessian, update_status

        ``update_status`` is one of:
        - ``"ssbroyden2"``
        - ``"bfgs_fallback"``
        - ``"skipped"``
        """
        y_s = torch.dot(y, s)
        if (not torch.isfinite(y_s)) or y_s <= eps:
            return inverse_hessian, "skipped"

        rho = y_s.reciprocal()
        h_y = inverse_hessian.mv(y)
        y_h_y = torch.dot(y, h_y)

        if (not torch.isfinite(y_h_y)) or y_h_y <= eps:
            return inverse_hessian, "skipped"

        h_k = y_h_y * rho
        b_k = -float(accepted_step_size) * rho * torch.dot(s, old_grad)

        if (
            (not torch.isfinite(h_k))
            or (not torch.isfinite(b_k))
            or h_k <= eps
            or b_k <= eps
        ):
            updated = self._standard_bfgs_inverse_update(
                inverse_hessian, s, y, rho, y_h_y
            )
            return updated, "bfgs_fallback"

        a_k = b_k * h_k - 1.0
        one_plus_a = 1.0 + a_k

        # The paper's formulas become numerically singular when a_k is
        # effectively zero. A standard BFGS update is a conservative fallback.
        if (
            (not torch.isfinite(one_plus_a))
            or one_plus_a <= eps
            or torch.abs(a_k) <= eps
        ):
            updated = self._standard_bfgs_inverse_update(
                inverse_hessian, s, y, rho, y_h_y
            )
            return updated, "bfgs_fallback"

        root_term = torch.sqrt(torch.abs(a_k) / one_plus_a)
        rho_minus = torch.minimum(
            torch.ones_like(h_k),
            h_k * (1.0 - root_term),
        )
        rho_minus = torch.clamp(rho_minus, min=eps)

        theta_minus = (rho_minus - 1.0) / a_k
        theta_plus = rho_minus.reciprocal()
        theta_candidate = (1.0 - b_k) / b_k
        theta_k = torch.maximum(
            theta_minus,
            torch.minimum(theta_plus, theta_candidate),
        )

        rho_k = torch.minimum(torch.ones_like(b_k), b_k.reciprocal())
        sigma_k = 1.0 + theta_k * a_k

        n_parameter = inverse_hessian.shape[0]
        if (
            n_parameter <= 1
            or (not torch.isfinite(sigma_k))
            or torch.abs(sigma_k) <= eps
        ):
            updated = self._standard_bfgs_inverse_update(
                inverse_hessian, s, y, rho, y_h_y
            )
            return updated, "bfgs_fallback"

        sigma_power = torch.abs(sigma_k).pow(1.0 / (1.0 - n_parameter))

        if bool(theta_k <= 0.0):
            tau_k = torch.minimum(rho_k * sigma_power, sigma_k)
        else:
            tau_k = rho_k * torch.minimum(sigma_power, theta_k.reciprocal())

        phi_denominator = 1.0 + a_k * theta_k
        if (
            (not torch.isfinite(tau_k))
            or tau_k <= eps
            or (not torch.isfinite(phi_denominator))
            or torch.abs(phi_denominator) <= eps
        ):
            updated = self._standard_bfgs_inverse_update(
                inverse_hessian, s, y, rho, y_h_y
            )
            return updated, "bfgs_fallback"

        v_k = s * rho - h_y / y_h_y
        phi_k = (1.0 - theta_k) / phi_denominator

        updated = (
            (
                inverse_hessian
                - torch.outer(h_y, h_y) / y_h_y
                + phi_k * y_h_y * torch.outer(v_k, v_k)
            )
            / tau_k
            + rho * torch.outer(s, s)
        )
        updated = 0.5 * (updated + updated.T)

        if not bool(torch.isfinite(updated).all()):
            updated = self._standard_bfgs_inverse_update(
                inverse_hessian, s, y, rho, y_h_y
            )
            return updated, "bfgs_fallback"

        return updated, "ssbroyden2"

    @torch.no_grad()
    def cast_state(
        self,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Cast all floating optimizer state, including the dense inverse Hessian
        and retained secant pairs, to a new dtype/device.
        """
        if device is None:
            device = self._params[0].device

        for state in self.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    state[key] = value.to(device=device, dtype=dtype)
                elif isinstance(value, list):
                    state[key] = [
                        item.to(device=device, dtype=dtype)
                        if torch.is_tensor(item) and item.is_floating_point()
                        else item
                        for item in value
                    ]

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        if closure is None:
            raise ValueError("DenseSSBroyden2 requires a closure.")

        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]

        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]
        c1 = group["c1"]
        c2 = group["c2"]
        eps = group["update_eps"]

        global_state = self.state[self._params[0]]
        if len(global_state) == 0:
            n_parameter = self._numel()
            reference = self._params[0]
            global_state["func_evals"] = 0
            global_state["n_iter"] = 0
            global_state["H_inv"] = torch.eye(
                n_parameter,
                dtype=reference.dtype,
                device=reference.device,
            )
            # These names intentionally match torch.optim.LBFGS state names,
            # allowing the existing curvature-proxy code to be reused.
            global_state["old_dirs"] = []
            global_state["old_stps"] = []
            global_state["ssbroyden2_updates"] = 0
            global_state["bfgs_fallback_updates"] = 0
            global_state["skipped_updates"] = 0

        inverse_hessian = global_state["H_inv"]

        original_loss = closure()
        current_loss = float(original_loss)
        flat_grad = self._gather_flat_grad()

        current_evals = 1
        global_state["func_evals"] += 1

        if bool(flat_grad.abs().max() <= tolerance_grad):
            return original_loss

        inner_iter = 0

        while inner_iter < max_iter:
            inner_iter += 1
            global_state["n_iter"] += 1

            direction = -inverse_hessian.mv(flat_grad)
            grad_direction = torch.dot(flat_grad, direction)

            # Reset to steepest descent if numerical errors destroy descent.
            if (not torch.isfinite(grad_direction)) or grad_direction >= -tolerance_change:
                inverse_hessian = torch.eye(
                    inverse_hessian.shape[0],
                    dtype=inverse_hessian.dtype,
                    device=inverse_hessian.device,
                )
                direction = -flat_grad
                grad_direction = -torch.dot(flat_grad, flat_grad)

            if bool(grad_direction > -tolerance_change):
                break

            if global_state["n_iter"] == 1:
                step_size = min(
                    1.0,
                    1.0 / max(float(flat_grad.abs().sum()), 1e-30),
                ) * lr
            else:
                step_size = lr

            x_initial = self._clone_param()

            def objective_along_direction(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            remaining_evals = max(1, max_eval - current_evals)

            (
                new_loss,
                new_flat_grad,
                accepted_step_size,
                line_search_evals,
            ) = _strong_wolfe(
                objective_along_direction,
                x_initial,
                step_size,
                direction,
                current_loss,
                flat_grad,
                grad_direction,
                c1=c1,
                c2=c2,
                tolerance_change=tolerance_change,
                max_ls=remaining_evals,
            )

            # _strong_wolfe restores x_initial after each trial evaluation.
            # Apply the accepted update once.
            self._add_flat_update(accepted_step_size, direction)

            current_evals += line_search_evals
            global_state["func_evals"] += line_search_evals

            s = direction * float(accepted_step_size)
            y = new_flat_grad - flat_grad

            inverse_hessian, update_status = self._ssbroyden2_inverse_update(
                inverse_hessian=inverse_hessian,
                s=s,
                y=y,
                accepted_step_size=float(accepted_step_size),
                old_grad=flat_grad,
                eps=eps,
            )
            global_state[f"{update_status}_updates"] += 1

            y_s = torch.dot(y, s)
            s_s = torch.dot(s, s)
            if (
                torch.isfinite(y_s)
                and torch.isfinite(s_s)
                and y_s > eps
                and s_s > eps
            ):
                global_state["old_dirs"].append(y.detach().clone())
                global_state["old_stps"].append(s.detach().clone())

                if len(global_state["old_dirs"]) > history_size:
                    global_state["old_dirs"].pop(0)
                    global_state["old_stps"].pop(0)

            previous_loss = current_loss
            current_loss = new_loss
            flat_grad = new_flat_grad

            global_state["last_step_size"] = float(accepted_step_size)
            global_state["last_loss"] = float(current_loss)

            if bool(flat_grad.abs().max() <= tolerance_grad):
                break
            if bool(s.abs().max() <= tolerance_change):
                break
            if abs(current_loss - previous_loss) < tolerance_change:
                break
            if current_evals >= max_eval:
                break

        global_state["H_inv"] = inverse_hessian
        return original_loss
