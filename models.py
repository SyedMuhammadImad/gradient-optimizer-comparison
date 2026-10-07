"""project implementation retained from author/assisted source; see ATTRIBUTION.md."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class MLP(nn.Module):

    def __init__(self, in_dim=25, hidden=(128, 64), out_dim=4, dropout=0.2):
        super().__init__()
        layers, prev = ([], in_dim)
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BaseOptimizer:
    """Small optimizer base class. Do not use torch.optim."""

    def __init__(self, params, lr):
        self.params = list(params)
        if not math.isfinite(lr) or lr <= 0: raise ValueError("Learning rate must be positive and finite")
        if not self.params: raise ValueError("No optimizer parameters")
        self.lr = lr
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("Duplicate parameters")
        if any(not isinstance(p, torch.Tensor) or not p.is_floating_point() for p in self.params):
            raise ValueError("Parameters must be floating tensors")

    def validate(self):
        # Validate every parameter before changing any of them.
        for p in self.params:
            if not torch.isfinite(p).all(): raise ValueError("Non-finite parameter")
            if p.grad is not None and (p.grad.is_sparse or not torch.isfinite(p.grad).all()):
                raise ValueError("Non-finite or sparse gradient")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        raise NotImplementedError

class SGD(BaseOptimizer):
    """theta <- theta - lr * (grad + wd * theta)."""

    def __init__(self, params, lr=0.01, weight_decay=0.0):
        super().__init__(params, lr)
        if not math.isfinite(weight_decay) or weight_decay < 0: raise ValueError("Invalid weight decay")
        self.wd = weight_decay

    def step(self):
        self.validate()
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.wd != 0:
                    grad = grad.add(p, alpha=self.wd)
                p.add_(grad, alpha=-self.lr)

class SGDMomentum(BaseOptimizer):
    """v <- beta * v + grad; theta <- theta - lr * v."""

    def __init__(self, params, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(params, lr)
        if not math.isfinite(beta) or not 0 <= beta < 1: raise ValueError("Invalid momentum")
        self.beta = beta
        if not math.isfinite(weight_decay) or weight_decay < 0: raise ValueError("Invalid weight decay")
        self.wd = weight_decay
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.validate()
        with torch.no_grad():
            for p, v in zip(self.params, self.v):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.wd != 0:
                    grad = grad.add(p, alpha=self.wd)
                v.mul_(self.beta).add_(grad)
                p.add_(v, alpha=-self.lr)

class RMSProp(BaseOptimizer):
    """s <- rho * s + (1-rho) * grad^2; theta <- theta - lr * grad / (sqrt(s)+eps)."""

    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-08, weight_decay=0.0):
        super().__init__(params, lr)
        if not math.isfinite(rho) or not 0 <= rho < 1: raise ValueError("Invalid decay")
        self.rho = rho
        if not math.isfinite(eps) or eps <= 0: raise ValueError("Invalid epsilon")
        self.eps = eps
        if not math.isfinite(weight_decay) or weight_decay < 0: raise ValueError("Invalid weight decay")
        self.wd = weight_decay
        self.s = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.validate()
        with torch.no_grad():
            for p, s in zip(self.params, self.s):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.wd != 0:
                    grad = grad.add(p, alpha=self.wd)
                s.mul_(self.rho).addcmul_(grad, grad, value=1 - self.rho)
                p.addcdiv_(grad, s.sqrt().add(self.eps), value=-self.lr)

class Adam(BaseOptimizer):
    """Adam with first/second moments and bias correction."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0):
        super().__init__(params, lr)
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in (beta1, beta2)): raise ValueError("Invalid betas")
        self.beta1 = beta1
        self.beta2 = beta2
        if not math.isfinite(eps) or eps <= 0: raise ValueError("Invalid epsilon")
        self.eps = eps
        if not math.isfinite(weight_decay) or weight_decay < 0: raise ValueError("Invalid weight decay")
        self.wd = weight_decay
        self.t = 0
        self.steps = [0 for _ in self.params]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        self.validate()
        with torch.no_grad():
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t
            for index, (p, m, v) in enumerate(zip(self.params, self.m, self.v)):
                if p.grad is None:
                    continue
                grad = p.grad
                if self.wd != 0:
                    grad = grad.add(p, alpha=self.wd)
                m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                self.steps[index] += 1
                bias_correction1 = 1 - self.beta1 ** self.steps[index]
                bias_correction2 = 1 - self.beta2 ** self.steps[index]
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)
