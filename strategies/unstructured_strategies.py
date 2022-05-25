# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         strategies/unstructured_strategies.py
# Description:  Sparsification strategies (unstructured Pruning case)
# ===========================================================================
from collections import OrderedDict
from math import ceil, log

import torch
import torch.nn.utils.prune as prune


#### Basic & IMP-based Strategies
class Dense:
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits."""

    def __init__(self):
        self.masks = dict()
        self.lr_dict = OrderedDict()  # epoch:lr
        self.train_loss_dict = OrderedDict()  # epoch:train_loss
        self.train_acc_dict = OrderedDict()
        self.is_in_finetuning_phase = False

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if
                                    hasattr(module, 'weight')
                                    and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                                                     torch.nn.BatchNorm2d)]
        self.n_prunable_parameters = sum(
            getattr(module, param_type).numel() for module, param_type in self.parameters_to_prune)

    @torch.no_grad()
    def start_forward_mode(self, **kwargs):
        """Function to be called before Forward step."""
        pass

    @torch.no_grad()
    def end_forward_mode(self, **kwargs):
        """Function to be called after Forward step."""
        pass

    # Important: no torch.no_grad
    def before_backward(self, **kwargs):
        """Function to be called after Forward step. Should return loss also if it is not modified."""
        return kwargs['loss']

    @torch.no_grad()
    def during_training(self, **kwargs):
        """Function to be called after loss.backward() and before optimizer.step, e.g. to mask gradients."""
        pass

    @torch.no_grad()
    def after_training_iteration(self, it):
        """Called after each training iteration"""
        pass

    def at_train_begin(self, model, LRScheduler):
        pass

    def at_epoch_end(self, **kwargs):
        self.lr_dict[kwargs['epoch']] = kwargs['epoch_lr']
        self.train_loss_dict[kwargs['epoch']] = kwargs['train_loss']
        self.train_acc_dict[kwargs['epoch']] = kwargs['train_acc']

    def at_train_end(self, **kwargs):
        pass

    def final(self, model, final_log_callback):
        self.remove_pruning_hooks(model=model)

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
        if compute_from_scratch:
            # We have to revert to weight_orig and then compute the mask
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Enforce the equivalence of weight_orig and weight
                    orig = getattr(module, param_type + "_orig").detach().clone()
                    prune.remove(module, param_type)
                    p = getattr(module, param_type)
                    p.copy_(orig)
                    del orig
        elif only_save_mask and len(self.masks) > 0:
            for module, param_type in self.parameters_to_prune:
                if (module, param_type) in self.masks:
                    prune.custom_from_mask(module, param_type, self.masks[(module, param_type)])

        # Do not prune biases and batch norm
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=self.get_pruning_method(),
            amount=pruning_sparsity,
        )

        self.masks = dict()  # Stays empty if we use regular pruning
        if only_save_mask:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Save the mask
                    mask = getattr(module, param_type + '_mask')
                    self.masks[(module, param_type)] = mask.detach().clone()
                    setattr(module, param_type + '_mask', torch.ones_like(mask))
                    # Remove (i.e. make permanent) the reparameterization
                    prune.remove(module=module, name=param_type)
                    # Delete the temporary mask to free memory
                    del mask

    def make_pruning_permanent(self):
        # Make the pruning permant, i.e. set the pruned weights to zero, than reinitialize from the same mask
        # This ensures that we can actually work (i.e. LMO, rescale computation) with the parameters
        # Important: For this to work we require that pruned weights stay zero in weight_orig over training
        # hence training, projecting etc should not modify (pruned) 0 weights in weight_orig
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Save the mask
                mask = getattr(module, param_type + '_mask')
                # Remove (i.e. make permanent) the reparameterization
                prune.remove(module=module, name=param_type)
                # Reinitialize the pruning
                prune.custom_from_mask(module=module, name=param_type, mask=mask)
                # Delete the temporary mask to free memory
                del mask

    def prune_momentum(self, optimizer):
        opt_state = optimizer.state
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the prunedness of momentum buffer
                param_state = opt_state[getattr(module, param_type + "_orig")]
                if 'momentum_buffer' in param_state:
                    mask = getattr(module, param_type + "_mask")
                    param_state['momentum_buffer'] *= mask.to(dtype=param_state['momentum_buffer'].dtype)

    def reset_momentum(self, optimizer):
        opt_state = optimizer.state
        for group in optimizer.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = opt_state[p]
                    if 'momentum_buffer' in param_state: del param_state['momentum_buffer']

    @torch.no_grad()
    def enforce_prunedness(self):
        """Secures that weight_orig has the same entries as weight. Important: At the moment of execution
        .weight might be old, it is updated in the forward pass automatically. Hence we update it by explicit mask application."""
        return

    @torch.no_grad()
    def get_per_layer_thresholds(self):
        """If properly defined, returns a list of per layer thresholds"""
        return []

    def get_pruning_method(self):
        return prune.Identity

    @torch.no_grad()
    def remove_pruning_hooks(self, model):
        # Note: this does not remove the pruning itself, but rather makes it permanent
        if len(self.masks) == 0:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    prune.remove(module, param_type)
        else:
            for module, param_type in self.masks:
                # Get the mask
                mask = self.masks[(module, param_type)]

                # Apply the mask
                orig = getattr(module, param_type)
                orig *= mask
            self.masks = dict()

    def initial_prune(self):
        pass

    def set_to_finetuning_phase(self):
        self.is_in_finetuning_phase = True


class IMP(Dense):
    def __init__(self, desired_sparsity: float, n_phases: int = 1, n_epochs_per_phase: int = 1) -> None:
        super().__init__()
        self.desired_sparsity = desired_sparsity
        self.n_phases = n_phases  # If None, compute this manually using Renda's approach of pruning 20% per phase
        self.n_epochs_per_phase = n_epochs_per_phase
        # Sparsity factor on remaining weights after each round, yields desired_sparsity after all rounds
        if self.n_phases is not None:
            self.pruning_sparsity = 1 - (1 - self.desired_sparsity) ** (1. / self.n_phases)
        else:
            self.pruning_sparsity = 0.2
            self.n_phases = ceil(log(1 - self.desired_sparsity, 1 - self.pruning_sparsity))

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        restore_callback()  # Restore to checkpoint model
        prune_per_phase = self.pruning_sparsity
        for phase in range(1, self.n_phases + 1, 1):
            self.pruning_step(model, pruning_sparsity=prune_per_phase)
            self.current_sparsity = 1 - (1 - prune_per_phase) ** phase
            after_pruning_callback(desired_sparsity=self.current_sparsity, prune_momentum=True)
            self.finetuning_step(desired_sparsity=self.current_sparsity, finetuning_callback=finetuning_callback,
                                 phase=phase)
            save_model_callback(
                model_type=f"{self.current_sparsity}-sparse_final")  # removing of pruning hooks happens in restore_callback

    def finetuning_step(self, desired_sparsity, finetuning_callback, phase):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase,
                            phase=phase)

    def get_pruning_method(self):
        return prune.L1Unstructured

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()
