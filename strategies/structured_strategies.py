# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         strategies/structured_strategies.py
# Description:  Sparsification strategies (structured Pruning case)
# ===========================================================================
from math import ceil, log
from strategies.unstructured_strategies import Dense as DenseUnstructured
import torch
import torch.nn.utils.prune as prune
from torchmetrics import MeanMetric
from utilities.utilities import Utilities as Utils


#### Basic & IMP-based Strategies
class struct_Dense(DenseUnstructured):
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits.
    NOTE: For now, this only supports filter pruning
    """

    def __init__(self):
        super().__init__()

    def after_initialization(self, model):
        """Called after initialization of the strategy"""
        self.conv_parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if
                                         hasattr(module, 'weight')
                                         and not isinstance(module.weight, type(None)) and isinstance(module,
                                                                                                      torch.nn.Conv2d)]
        self.parameters_to_prune = self.conv_parameters_to_prune
        self.n_prunable_parameters = sum(
            getattr(module, param_type).numel() for module, param_type in self.parameters_to_prune)

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

        # We prune filters locally
        for module, param_type in self.parameters_to_prune:
            prune.ln_structured(module, param_type, pruning_sparsity, n=1, dim=0)

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

    # Important: no torch.no_grad
    def before_backward(self, **kwargs):
        """Add group penalty if needed. Needs to return modified loss"""
        loss, group_penalty, group_penalty_type = kwargs['loss'], kwargs['group_penalty'], kwargs['group_penalty_type']
        group_penalty_factor = group_penalty or 0
        if group_penalty_type is not None and group_penalty_factor > 0:
            if group_penalty_type == 'Nuclear_Conv':
                # Add nuclear norm regularization to Conv layers, but reshape them first
                for module, param_type in self.conv_parameters_to_prune:
                    if hasattr(module, param_type + "_orig"):
                        p = getattr(module, param_type + "_orig")
                    else:
                        p = getattr(module, param_type)
                    loss = loss + group_penalty_factor * torch.linalg.norm(p.flatten(start_dim=1), ord='nuc')
            elif group_penalty_type == 'Filter_Conv':
                # Adds a weighted L2-penalty for each filter
                for module, param_type in self.conv_parameters_to_prune:
                    if hasattr(module, param_type + "_orig"):
                        p = getattr(module, param_type + "_orig")
                    else:
                        p = getattr(module, param_type)
                    loss = loss + group_penalty_factor * \
                           torch.sum(torch.linalg.norm(p.flatten(start_dim=1), ord=2, dim=1))
            else:
                raise NotImplementedError(f"group penalty {group_penalty_type} not implemented.")

        return loss

    @torch.no_grad()
    def get_struct_metrics(self, group_penalty, group_penalty_type, **kwargs):
        device = kwargs['device']
        lmo = kwargs['lmo']
        opt = kwargs['opt']
        struct_metrics = {
            'energy_mean': MeanMetric().to(device=device),
            'energy_std': MeanMetric().to(device=device),
            'energy_range': MeanMetric().to(device=device),
        }
        if group_penalty_type is not None or lmo in ['NuclearNormBall', 'SpectralKSparsePolytope',
                                                     'SpectralKSupportNormBall'] or opt == 'Prox_SGD':
            if group_penalty_type == 'Nuclear_Conv' or lmo in ['NuclearNormBall', 'SpectralKSparsePolytope',
                                                               'SpectralKSupportNormBall'] or opt == 'Prox_SGD':
                for module, param_type in self.conv_parameters_to_prune:
                    if hasattr(module, param_type + "_orig"):
                        p = getattr(module, param_type + "_orig")
                    else:
                        p = getattr(module, param_type)
                    svdvals = torch.linalg.svdvals(p.flatten(start_dim=1))
                    struct_metrics['energy_mean'](torch.mean(svdvals))
                    struct_metrics['energy_std'](torch.std(svdvals))
                    struct_metrics['energy_range'](svdvals[0] - svdvals[-1])
        return {f"struct_metrics.{key}": m.compute() for key, m in struct_metrics.items()}


class struct_IMP(struct_Dense):
    # This is essentially PFEC (Li et al., 2016)
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
            if self.n_phases == 1:
                self.current_sparsity = prune_per_phase
            elif phase == self.n_phases:
                self.current_sparsity = self.desired_sparsity
            else:
                self.current_sparsity = 1 - (1 - prune_per_phase) ** phase  # Might lead to numerical errors
            after_pruning_callback(desired_sparsity=self.current_sparsity, prune_momentum=True)
            self.finetuning_step(desired_sparsity=self.current_sparsity, finetuning_callback=finetuning_callback,
                                 phase=phase)
            save_model_callback(
                model_type=f"{self.current_sparsity}-sparse_final")  # removing of pruning hooks happens in restore_callback

    def finetuning_step(self, desired_sparsity, finetuning_callback, phase):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase,
                            phase=phase)

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()


class struct_Decomp(struct_Dense):

    def __init__(self, desired_sparsity: float, n_epochs_per_phase: int = 1, decomp_type: str = 'linear') -> None:
        super().__init__()
        assert decomp_type in ['linear', 'conv']
        self.desired_sparsity = desired_sparsity
        self.n_epochs_per_phase = n_epochs_per_phase
        self.decomp_type = decomp_type

    def at_train_end(self, model, finetuning_callback, restore_callback, save_model_callback, after_pruning_callback,
                     opt):
        # restore_callback()  # Restore to checkpoint model # Does not require reinit since we only do this for one sparsity per run
        self.pruning_step(model, pruning_sparsity=self.desired_sparsity)
        self.after_initialization(model=model)  # Update pruning set for correct metrics
        after_pruning_callback(desired_sparsity=self.desired_sparsity, prune_momentum=False)
        self.finetuning_step(desired_sparsity=self.desired_sparsity, finetuning_callback=finetuning_callback,
                             phase=1)
        save_model_callback(
            model_type=f"{self.desired_sparsity}-sparse_final")  # removing of pruning hooks happens in restore_callback

    def finetuning_step(self, desired_sparsity, finetuning_callback, phase):
        finetuning_callback(desired_sparsity=desired_sparsity, n_epochs_finetune=self.n_epochs_per_phase,
                            phase=phase)

    def final(self, model, final_log_callback):
        super().final(model=model, final_log_callback=final_log_callback)
        final_log_callback()

    @torch.no_grad()
    def pruning_step(self, model, pruning_sparsity):
        Utils.replace_layers_by_decomposition(model=model, svdSparsity=pruning_sparsity, layer_type=self.decomp_type)

    # Important: no torch.no_grad
    def before_backward(self, **kwargs):
        """Just return the loss, we do not want to regularize after decomposing"""
        loss, group_penalty, group_penalty_type = kwargs['loss'], kwargs['group_penalty'], kwargs['group_penalty_type']
        return loss
