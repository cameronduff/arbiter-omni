"""
DirectML-native AdamW optimizer [AO-35].

torch.optim.AdamW uses aten::lerp.Scalar_out, which has no DirectML kernel and
silently falls back to the CPU (forcing a device sync on every parameter update).
This implementation expresses the identical update with mul_/add_/addcmul_/addcdiv_,
all of which execute on the GPU.
"""

from __future__ import annotations

from typing import Tuple

import torch


class DirectMLNativeAdamW(torch.optim.Optimizer):
    """
    AdamW with decoupled weight decay, numerically equivalent to torch.optim.AdamW
    (amsgrad=False) but free of the lerp CPU fallback.

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        theta_t = theta_{t-1} * (1 - lr * wd)
                  - (lr / (1 - beta1^t)) * m_t / (sqrt(v_t / (1 - beta2^t)) + eps)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t
                step_size = lr / bias_corr1

                denom = (v / bias_corr2).sqrt_().add_(eps)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

        return loss
