import torch
import math
from torch.optim.optimizer import Optimizer

class HonestFractionalMomentum(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.5, history_size=20):
        defaults = dict(lr=lr, alpha=alpha, history_size=history_size)
        super(HonestFractionalMomentum, self).__init__(params, defaults)

    def _get_gl_coeffs(self, alpha, k_max):
        coeffs = []
        for k in range(k_max + 1):
            if k == 0:
                c = 1.0
            else:
                c = ((-1)**k) * (math.gamma(alpha + 1) / (math.gamma(k + 2) * math.gamma(alpha - k)))
            coeffs.append(c)
        return torch.tensor(coeffs, dtype=torch.float32)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            hist_size = group['history_size']
            
            coeffs = self._get_gl_coeffs(alpha, hist_size).to(next(iter(self.param_groups[0]['params'])).device)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['history'] = []
                
                state['history'].insert(0, grad.clone())
                
                if len(state['history']) > hist_size + 1:
                    state['history'].pop()

                update = torch.zeros_like(p.data)
                for i, g_old in enumerate(state['history']):
                    if i < len(coeffs):
                        update.add_(g_old, alpha=coeffs[i])

                p.data.add_(update, alpha=-lr)

        return loss

class FastFractionalMomentum(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.99, 0.999), coeffs=(0.5, 0.3, 0.2)):
        defaults = dict(lr=lr)
        super(FastFractionalMomentum, self).__init__(params, defaults)
        self.betas = betas
        self.coeffs = coeffs

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['buffers'] = [torch.zeros_like(p.data) for _ in self.betas]

                state['step'] += 1

                for i, (buf, beta) in enumerate(zip(state['buffers'], self.betas)):
                    buf.mul_(beta).add_(grad, alpha=1 - beta)

                update = torch.zeros_like(p.data)
                for buf, coeff in zip(state['buffers'], self.coeffs):
                    update.add_(buf, alpha=coeff)

                p.data.add_(update, alpha=-group['lr'])

        return loss