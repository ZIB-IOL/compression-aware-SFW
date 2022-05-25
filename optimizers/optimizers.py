# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         optimizers.py
# Description:  SFW, SGD, Proximal SGD
# ===========================================================================
import torch
import torch.optim as native_optimizers
from torchmetrics import MeanMetric


class SFW(native_optimizers.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of lr rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, lr=0.1, rescale='diameter', momentum=0, dampening=0, extensive_metrics=False,
                 device='cpu'):
        momentum = momentum or 0
        dampening = dampening or 0
        if rescale is None and not (0.0 <= lr <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"Momentum must be between 0 and 1, got {momentum} of type {type(momentum)}.")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("fDampening must be between 0 and 1, got {dampening} of type {type(dampening)}.")

        if rescale == 'None':
            rescale = None
        if not (rescale in ['diameter', 'gradient', 'fast_gradient', None]):
            raise ValueError(
                f"Rescale type must be either 'diameter', 'gradient', 'fast_gradient' or None, got {rescale} of type {type(rescale)}.")

        self.rescale = rescale
        self.extensive_metrics = extensive_metrics

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(SFW, self).__init__(params, defaults)

        # Metrics
        self.metrics = {
            'grad_norm': MeanMetric().to(device=device),
            'grad_normalizer_norm': MeanMetric().to(device=device),
            'diameter_normalizer': MeanMetric().to(device=device),
            'effective_lr': MeanMetric().to(device=device),
        }

    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def get_metrics(self):
        return {m: self.metrics[m].compute() if self.extensive_metrics else {} for m in self.metrics.keys()}

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state:
                        del param_state['momentum_buffer']

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            constraint = group['constraint']
            momentum = group['momentum']
            dampening = group['dampening']

            # Add momentum, fill grad list with momentum_buffers and concatenate
            grad_list = []
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = param_state['momentum_buffer']
                grad_list.append(d_p)

            # LMO solution
            v_list = constraint.lmo(grad_list)  # LMO optimal solution

            # Determine learning rate rescaling factor
            factor = 1
            if self.rescale == 'diameter':
                # Rescale lr by diameter
                factor = 1. / constraint.get_diameter()
                if self.extensive_metrics:
                    self.metrics['grad_norm'](torch.norm(torch.cat([g.flatten() for g in grad_list]), p=2))
                    self.metrics['diameter_normalizer'](constraint.get_diameter())
            elif self.rescale in ['fast_gradient', 'gradient']:
                # Rescale lr by gradient
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grad_list]), p=2)
                if self.rescale == 'fast_gradient':
                    grad_normalizer_norm = 0.5 * constraint.get_diameter()
                else:
                    grad_normalizer_norm = torch.norm(torch.cat([p.flatten()
                                                                 for p in group['params'] if
                                                                 p.grad is not None]) - torch.cat(
                        [v_i.flatten() for v_i in v_list]), p=2)
                factor = grad_norm / grad_normalizer_norm
                if self.extensive_metrics:
                    self.metrics['grad_norm'](grad_norm)
                    self.metrics['grad_normalizer_norm'](grad_normalizer_norm)
                    self.metrics['diameter_normalizer'](constraint.get_diameter())

            lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

            # Update parameters
            for p_idx, p in enumerate(group['params']):
                p.mul_(1 - lr)
                p.add_(v_list[p_idx], alpha=lr)
            if self.extensive_metrics:
                self.metrics['effective_lr'](lr)
        return loss


class SGD(native_optimizers.SGD):
    """Reimplementation of SGD that allows for decoupling weight decay from the learning rate"""

    def __init__(self, decouple_wd=False, extensive_metrics=False, device='cpu', **kwargs):
        if kwargs['weight_decay'] is None: kwargs['weight_decay'] = 0.
        super().__init__(**kwargs)
        self.decouple_wd = decouple_wd
        self.extensive_metrics = extensive_metrics

        # Metrics
        self.metrics = {
            'grad_norm': MeanMetric().to(device=device),
            'effective_lr': MeanMetric().to(device=device),
        }

        self.it_counter = 0

    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def get_metrics(self):
        return {m: self.metrics[m].compute() if self.extensive_metrics else {} for m in self.metrics.keys()}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.it_counter += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        squared_update_norm = 0  # Collect global squared L_2 distance within one step

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):

                d_p = d_p_list[i]
                if weight_decay != 0 and not self.decouple_wd:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if self.extensive_metrics:
                    # Add costly metrics
                    grad_norm = torch.norm(d_p_list[i], p=2)
                    self.metrics['grad_norm'](grad_norm)

                    # Intermediate save current parameters
                    p_old = param.detach().clone()

                if weight_decay != 0 and self.decouple_wd:
                    param.mul_(1 - weight_decay)

                param.add_(d_p, alpha=-lr)

                if self.extensive_metrics:
                    p_new = param
                    squared_update_norm += torch.sum((p_new - p_old) ** 2)  # Add to squared L2-distance

                    del p_new

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        if self.extensive_metrics:
            self.metrics['effective_lr'](torch.sqrt(squared_update_norm))

        return loss


class Prox_SGD(SGD):
    """Proximal SGD"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Perform an SGD step, then apply the proximal operator to all parameters
        super().step(closure=closure)

        for group in self.param_groups:
            if 'prox_operator' not in group:
                continue
            prox_operator = group['prox_operator']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.copy_(prox_operator(x=p, lr=lr))
