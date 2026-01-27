import torch
from torch.optim import Optimizer

class SAM_Optimizer(Optimizer):
    """
    Two-gradient optimizer with internal q-norm perturbation:
      1) v = ∇L(p)
      2) u = rho * sign(v) * |v|^(q-1) / (||v||_q + eps)
      3) g = ∇L(p + u)
      4) restore p, then p <- p - lr * g

    Args:
        params: iterable of model parameters
        lr: learning rate
        q: q-norm exponent (>= 1)
        rho: perturbation magnitude
        eps: small constant for numerical stability
    """
    def __init__(self, params, lr: float = 1e-3, q: float = 2.0, rho: float = 0.05, eps: float = 1e-12):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if q < 1.0:
            raise ValueError(f"q must be >= 1, got {q}")
        super().__init__(params, dict(lr=lr, q=q, rho=rho, eps=eps))

    def _global_qnorm(self, q: float) -> torch.Tensor:
        # ||v||_q = (sum_i sum_j |v_ij|^q)^(1/q) over all parameter gradients
        total = None
        p_number = 1.0/(1.0 - 1.0/q) if q > 1.0 else float('inf')  # dual norm exponent
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                s = p.grad.detach().abs().pow(q).sum()
                total = s if total is None else total + s
        if total is None:
            return torch.tensor(0.0)
        return total.pow(1.0 / p_number)

    def _perturb_from_grad(self, v: torch.Tensor, norm: torch.Tensor, q: float, rho: float, eps: float) -> torch.Tensor:
        if v is None:
            return None
        # if torch.isnan(norm) or norm.item() == 0.0:
        #     return torch.zeros_like(v)
        u = v.abs().pow(q - 1.0)           # |v|^(q-1)
        u.mul_(v.sign())                   # restore sign
        u.div_(norm + eps)                 # normalize by ||v||_q
        u.mul_(rho)                        # scale by rho
        return u

    def step(self, closure):
        if closure is None:
            raise RuntimeError("CustomPerturbOptimizer requires a closure returning the loss.")

        # 1) v = ∇L(p)
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
            loss.backward()

        # read hyperparams (assume same across groups)
        q = self.param_groups[0]["q"]
        rho = self.param_groups[0]["rho"]
        eps = self.param_groups[0]["eps"]

        # 2) compute global q-norm and perturb all params: p <- p + u
        qnorm = self._global_qnorm(q)
        u_lists = []
        with torch.no_grad():
            for group in self.param_groups:
                u_list = []
                for p in group["params"]:
                    u = self._perturb_from_grad(p.grad, qnorm, q, rho, eps)
                    if u is not None:
                        p.add_(u)
                    u_list.append(u)
                u_lists.append(u_list)

        # 3) g = ∇L(p + u)
        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss_adv = closure()
            loss_adv.backward()

        # 4) restore and update: p <- p - lr * g
        for (group, u_list) in zip(self.param_groups, u_lists):
            lr = group["lr"]
            params = group["params"]
            with torch.no_grad():
                # restore
                for p, u in zip(params, u_list):
                    if u is not None:
                        p.sub_(u)
                # update with g from perturbed point
                for p in params:
                    g = p.grad
                    if g is None:
                        continue
                    p.add_(g, alpha=-lr)

        return loss_adv