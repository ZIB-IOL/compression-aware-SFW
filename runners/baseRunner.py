# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         baseRunner.py
# Description:  Base Runner class, all other runners inherit from this one
# ===========================================================================
import importlib
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from math import sqrt

import torch
import wandb
from barbar import Bar
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric, Accuracy

from config import datasetDict, trainTransformDict, testTransformDict
from metrics import metrics
from optimizers import constraints
from optimizers.operators import ProximalOperator
from optimizers.optimizers import SGD, SFW, Prox_SGD
from strategies import unstructured_strategies, structured_strategies
from utilities import wd_schedulers
from utilities.utilities import SequentialSchedulers, FixedLR
from utilities.utilities import Utilities as Utils


class baseRunner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config, debug_mode=False):
        self.config = config
        self.dataParallel = True if (self.config.dataset == 'imagenet' and torch.cuda.device_count() > 1) else False
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)

        # Dynamic retraining: Catch if the retraining length is dynamic -> it is determined as follows:
        # nepochs = 160, dynamic_retrain_length=200 => Train for 160 epochs, then retrain for 200-160=40 epochs
        if self.config.dynamic_retrain_length is not None:
            assert self.config.n_phases in [None, 1], "Dynamic retraining length only work with a single phase"
            assert self.config.n_epochs_per_phase is None
            assert self.config.dynamic_retrain_length > self.config.nepochs
            assert self.config.strategy == 'Restricted_IMP'
            self.config.update({'n_epochs_per_phase': self.config.dynamic_retrain_length - self.config.nepochs},
                               allow_val_change=True)

        assert not (
                self.config.channels_last and not self.config.use_amp), "Using channels_last with fp32 can dramatically slow things down."

        # Set a couple useful variables
        self.checkpoint_file = None
        self.trained_test_accuracy = None
        self.trained_train_loss = None
        self.after_pruning_metrics = {}
        self.totalTrainTime = None
        self.totalFinetuneTime = None
        self.debug_mode = debug_mode  # If active, use less desired_sparsities, etc
        self.seed = None
        self.squared_model_norm = None
        self.n_warmup_epochs = None
        self.trainIterationCtr = 1
        self.tmp_dir = tempfile.mkdtemp()
        self.ampGradScaler = None  # Note: this must be reset before training, and before retraining
        self.constraintList = None
        self.wd_scheduler = None
        self.lmo_delay_epoch = None
        self.struct_cmp_metrics = {}

        # Budgeted training variables
        self.stability_scaling = None  # Factor to multiply cycle amplitude for BIMP_LC

        # Variables to be set by inheriting classes
        self.strategy = None
        self.trainLoader = None
        self.testLoader = None
        self.n_datapoints = None
        self.model = None

        # Define the loss object and metrics
        # Important note: for the correct computation of loss/accuracy it's important to have reduction == 'mean'
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

        k_accuracy = 5 if self.config.dataset in ['cifar100', 'imagenet', 'tinyimagenet'] else 3
        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'accuracy': Accuracy().to(device=self.device),
                               'k_accuracy': Accuracy(top_k=k_accuracy).to(device=self.device)}
                        for mode in ['train', 'test']}

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

        if self.config.optimizer in ['SGD', 'SFW']:
            self.optimizer.reset_metrics()

    def get_metrics(self):
        with torch.no_grad():
            self.strategy.start_forward_mode()  # Necessary to have correct computations for DPF
            x_input, y_target = next(iter(self.testLoader))
            x_input, y_target = x_input.to(self.device), y_target.to(self.device)  # Move to CUDA if possible
            n_flops, n_nonzero_flops = metrics.get_flops(model=self.model, x_input=x_input)
            if 'trained.nonzero_inference_flops' in self.struct_cmp_metrics:
                # Use the reference value for the full count,
                # otherwise the metrics make no sense since the model changed
                n_flops = self.struct_cmp_metrics['trained.nonzero_inference_flops']
            n_total, n_nonzero = metrics.get_parameter_count(model=self.model)
            use_refval = False
            if 'trained.n_nonzero_params' in self.struct_cmp_metrics:
                # Use the reference value for the full count,
                # otherwise the metrics make no sense since the model changed
                n_total = self.struct_cmp_metrics['trained.n_nonzero_params']
                use_refval = True

            loggingDict = dict(
                train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items()},
                test={metric_name: metric.compute() for metric_name, metric in self.metrics['test'].items()},
                global_sparsity=metrics.global_sparsity(module=self.model, refVal=n_total if use_refval else None),
                modular_sparsity=metrics.modular_sparsity(parameters_to_prune=self.strategy.parameters_to_prune,
                                                          refVal=n_total if use_refval else None),
                modular_compression=metrics.modular_compression(parameters_to_prune=self.strategy.parameters_to_prune,
                                                                refVal=n_total if use_refval else None),
                global_compression=metrics.compression_rate(module=self.model, refVal=n_total if use_refval else None),
                nonzero_inference_flops=n_nonzero_flops,
                baseline_inference_flops=n_flops,
                theoretical_speedup=metrics.get_theoretical_speedup(n_flops=n_flops, n_nonzero_flops=n_nonzero_flops),
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                distance_to_origin=metrics.get_distance_to_origin(self.model),
            )
            if self.config.optimizer in ['SGD', 'SFW']:
                loggingDict.update(self.optimizer.get_metrics())

            # Collect some structural sparsity information
            if self.config.strategy == 'struct_Dense':
                loggingDict.update(self.strategy.get_struct_metrics(group_penalty=self.config.group_penalty,
                                                                    group_penalty_type=self.config.group_penalty_type,
                                                                    device=self.device,
                                                                    lmo=self.config.lmo,
                                                                    opt=self.config.optimizer))

            self.strategy.end_forward_mode()  # Necessary to have correct computations for DPF
        return loggingDict

    def get_dataloaders(self):
        # Determine where the data lies
        for root in ['./datasets_pytorch/']:
            rootPath = f"{root}{self.config.dataset}"
            if os.path.isdir(rootPath):
                break

        if self.config.dataset in ['imagenet']:
            trainData = datasetDict[self.config.dataset](root=rootPath, split='train',
                                                         transform=trainTransformDict[self.config.dataset])
            testData = datasetDict[self.config.dataset](root=rootPath, split='val',
                                                        transform=testTransformDict[self.config.dataset])
        elif self.config.dataset == 'tinyimagenet':
            traindir = os.path.join(rootPath, 'train')
            valdir = os.path.join(rootPath, 'val')
            trainData = datasetDict[self.config.dataset](root=traindir,
                                                         transform=trainTransformDict[self.config.dataset])
            testData = datasetDict[self.config.dataset](root=valdir, transform=testTransformDict[self.config.dataset])
        else:
            trainData = datasetDict[self.config.dataset](root=rootPath, train=True, download=True,
                                                         transform=trainTransformDict[self.config.dataset])
            testData = datasetDict[self.config.dataset](root=rootPath, train=False,
                                                        transform=testTransformDict[self.config.dataset])
        self.n_datapoints = len(trainData)
        if self.config.dataset in ['imagenet', 'cifar100', 'tinyimagenet']:
            num_workers = 4 if torch.cuda.is_available() else 0
        else:
            num_workers = 2 if torch.cuda.is_available() else 0

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, shuffle=True,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers)
        testLoader = torch.utils.data.DataLoader(testData, batch_size=self.config.batch_size, shuffle=False,
                                                 pin_memory=torch.cuda.is_available(), num_workers=num_workers)

        return trainLoader, testLoader

    def get_model(self, reinit: bool, temporary: bool = True) -> torch.nn.Module:
        if reinit:
            # Define the model
            model = getattr(importlib.import_module('models.' + self.config.dataset), self.config.model)()
        else:
            # The model has been initialized already
            model = self.model
            # Note, we have to get rid of all existing prunings, otherwise we cannot load the state_dict as it is unpruned
            if self.strategy:
                self.strategy.remove_pruning_hooks(model=self.model)

        file = self.checkpoint_file
        if file is not None:
            dir = wandb.run.dir if not temporary else self.tmp_dir
            fPath = os.path.join(dir, file)

            # We need to check whether the model was loaded using dataparallel, in that case remove 'module'
            # original saved file with DataParallel
            state_dict = torch.load(fPath, map_location=self.device)
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                new_key = key
                if key.startswith("module."):
                    new_key = key[7:]  # remove `module.`
                new_state_dict[new_key] = val

            # Load the state_dict
            model.load_state_dict(new_state_dict)

        if self.dataParallel and reinit:  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def define_optimizer_scheduler(self):
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = self.config.learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
            scheduler = (name, kwargs)
            self.initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        if self.config.optimizer == 'SGD':
            wd = self.config.weight_decay

            self.optimizer = SGD(params=self.model.parameters(), lr=self.initial_lr,
                                 momentum=self.config.momentum,
                                 weight_decay=wd, nesterov=self.config.nesterov,
                                 decouple_wd=self.config.decouple_wd,
                                 extensive_metrics=self.config.extensive_metrics,
                                 device=self.config.device)
            self.define_wd_scheduler()


        elif self.config.optimizer == 'SFW':
            # Define the param groups
            param_groups = [{'params': param_list, 'constraint': constraint}
                            for constraint, param_list in self.constraintList]
            self.optimizer = SFW(params=param_groups, lr=self.initial_lr,
                                 rescale=self.config.lmo_rescale, momentum=self.config.momentum,
                                 extensive_metrics=self.config.extensive_metrics,
                                 device=self.config.device)

        elif self.config.optimizer == 'Prox_SGD':
            convParams = [p for p in self.model.parameters() if len(p.shape) == 4]
            restParams = [p for p in self.model.parameters() if len(p.shape) != 4]

            param_groups = [
                {'params': convParams,
                 'prox_operator': ProximalOperator.svd_soft_thresholding(threshold=self.config.prox_threshold)},
                {'params': restParams}
            ]
            self.optimizer = Prox_SGD(params=param_groups, lr=self.initial_lr,
                                      momentum=self.config.momentum,
                                      weight_decay=self.config.weight_decay, nesterov=self.config.nesterov,
                                      decouple_wd=self.config.decouple_wd,
                                      extensive_metrics=self.config.extensive_metrics,
                                      device=self.config.device,
                                      fix_norm=self.config.fix_norm,
                                      amplify_grad=self.config.amplify_grad)
        else:
            raise NotImplementedError

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = len(self.trainLoader)
        n_total_iterations = iterations_per_epoch * self.config.nepochs
        n_warmup_iterations = 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = self.initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if self.config.n_epochs_warmup and self.config.n_epochs_warmup > 0:
            assert int(
                self.config.n_epochs_warmup) == self.config.n_epochs_warmup, "At the moment no float warmup allowed."
            n_warmup_iterations = int(float(self.config.n_epochs_warmup) * iterations_per_epoch)
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        name, kwargs = scheduler
        scheduler = None
        if name == 'Constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif name == 'StepLR':
            # Tuple of form ('StepLR', initial_lr, step_size, gamma)
            # Reduces initial_lr by gamma every step_size epochs
            step_size, gamma = int(kwargs[1]), float(kwargs[2])

            # Convert to iterations
            step_size = iterations_per_epoch * step_size

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size,
                                                        gamma=gamma)
        elif name == 'MultiStepLR':
            # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
            # Reduces initial_lr by gamma every epoch that is in the list milestones
            milestones, gamma = kwargs[1].strip(), float(kwargs[2])
            # Remove square bracket
            if milestones[0] == '[':
                milestones = milestones[1:]
            if milestones[-1] == ']':
                milestones = milestones[:-1]
            # Convert to iterations directly
            milestones = [int(ms) * iterations_per_epoch for ms in milestones.split('|')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                             gamma=gamma)
        elif name == 'ExponentialLR':
            # Tuple of form ('ExponentialLR', initial_lr, gamma)
            gamma = float(kwargs[1])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif name == 'Linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=0.,
                                                          total_iters=n_remaining_iterations)
        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        # Reset base lrs to make this work
        scheduler.base_lrs = [self.initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        elif name in ['StepLR', 'MultiStepLR']:
            # We need parallel schedulers, since the steps should be counted during warmup
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def define_wd_scheduler(self, retrain=False):
        if self.config.optimizer == 'SFW':
            return
        config_val = self.config.weight_decay_schedule
        if retrain:
            config_val = self.config.retrain_schedule_wd
        if config_val in [None, 'None', 'none']:
            return

        # wd scheduler in the form (type, kwargs)
        tupleStr = config_val.strip()
        # Remove parenthesis
        tupleStr = tupleStr[1:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['InitialOnly']:
            scheduler = (name, kwargs)
        else:
            raise NotImplementedError(f"Weight decay scheduler {name} not implemented.")

        iterations_per_epoch = len(self.trainLoader)
        n_epochs = self.config.nepochs if not retrain else self.config.n_epochs_per_phase
        n_total_iterations = iterations_per_epoch * n_epochs
        n_warmup_iterations = 0  # Without warmup for now
        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        # Set the optimizer weight decay
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = self.config.weight_decay

        name, kwargs = scheduler
        scheduler = None
        if name == 'InitialOnly':
            # In the form (InitialOnly, x), where 0 <= x <= 1 is the fraction of total iterations until wd -> 0
            # Weight decay only for some epochs, then set to zero
            initial_length = int(n_remaining_iterations * float(kwargs[0]))
            scheduler = wd_schedulers.StepWD(optimizer=self.optimizer, step_size=initial_length, gamma=0.,
                                             verbose=False)

        self.wd_scheduler = scheduler

    def define_constraints(self, retrain=False):
        # Define the constraints for SFW
        if self.config.optimizer != 'SFW' or self.config.strategy == 'struct_Decomp':
            return
        moduleList = [(module, param_type) for module in self.model.modules() for param_type in ['weight', 'bias']
                      if (hasattr(module, param_type) and type(getattr(module, param_type)) != type(None))
                      ]

        delay = self.config.lmo_delay or 0
        lmo_k = self.config.lmo_k
        if delay > 0 and not retrain:
            assert self.config.lmo in ['KSparsePolytope', 'KSupportNormBall']
            assert self.config.lmo_mode == 'initialization'
            lmo_k = 1  # Use full k first
            self.lmo_delay_epoch = max(1, int(delay * self.config.nepochs))

        if self.config.lmo == 'LpBall':
            constraintList = constraints.set_lp_constraints(moduleList=moduleList,
                                                            global_constraint=self.config.lmo_global,
                                                            p=self.config.lmo_ord,
                                                            value=self.config.lmo_value, mode=self.config.lmo_mode)

        elif self.config.lmo in ['KSparsePolytope', 'KSupportNormBall']:
            constr_type = 'k_sparse' if self.config.lmo == 'KSparsePolytope' else 'k_support'
            constraintList = constraints.set_k_constraints(moduleList=moduleList, constr_type=constr_type,
                                                           global_constraint=self.config.lmo_global,
                                                           k=lmo_k,
                                                           value=self.config.lmo_value,
                                                           mode=self.config.lmo_mode,
                                                           adjust_diameter=self.config.lmo_adjust_diameter)

        elif self.config.lmo == 'GroupKSupportNormBall':
            constraintList = constraints.set_structured_k_constraints(moduleList=moduleList,
                                                                      global_constraint=self.config.lmo_global,
                                                                      k=lmo_k,
                                                                      value=self.config.lmo_value,
                                                                      mode=self.config.lmo_mode)

        elif self.config.lmo == 'NuclearNormBall':
            constraintList = constraints.set_structured_decomp_constraints(moduleList=moduleList,
                                                                           lmo_nuc_method=self.config.lmo_nuc_method,
                                                                           global_constraint=self.config.lmo_global,
                                                                           value=self.config.lmo_value,
                                                                           mode=self.config.lmo_mode)
        elif self.config.lmo in ['SpectralKSparsePolytope', 'SpectralKSupportNormBall']:
            constr_type = 'k_sparse' if self.config.lmo == 'SpectralKSparsePolytope' else 'k_support'
            constraintList = constraints.set_structured_k_decomp_constraints(moduleList=moduleList,
                                                                             constr_type=constr_type,
                                                                             lmo_nuc_method=self.config.lmo_nuc_method,
                                                                             global_constraint=self.config.lmo_global,
                                                                             k=lmo_k,
                                                                             value=self.config.lmo_value,
                                                                             mode=self.config.lmo_mode)

        else:
            raise NotImplementedError(f"Oracle {self.config.lmo} not implemented")

        self.constraintList = constraintList
        # In the course of get_avg_init_norms params are changed -> revert to initial_model if necessary
        if self.config.fixed_init:
            sys.stdout.write(f"Avg_init_norm computation modified the weights, reverting to {self.checkpoint_file}\n")
            # Just loading the weights to the existing model does not modify the constraints!
            self.restore_model()

    def change_constraints(self):
        for constraint, param_list in self.constraintList:
            n_constraint = constraint.n
            k_constraint = min(int(self.config.lmo_k * n_constraint), n_constraint)
            if self.config.lmo_k > 1:
                # An actual integer number was specified
                k_constraint = min(int(self.config.lmo_k), n_constraint)
            k_constraint = max(k_constraint, 1)
            constraint.reset_k(k=k_constraint)
        constraints.make_feasible(constraintList=self.constraintList)

    def define_strategy(self):
        assert self.config.n_phases in [None, 1] or 'IMP' in self.config.strategy
        assert not self.config.channels_last or self.config.strategy in ['Dense', 'struct_Dense'], "Cannot use " \
                                                                                                   "channels_last " \
                                                                                                   "with pruning. "

        #### STRUCTURED
        if self.config.strategy == 'struct_Dense':
            return structured_strategies.struct_Dense()
        elif self.config.strategy == 'struct_IMP':
            return structured_strategies.struct_IMP(desired_sparsity=self.config.goal_sparsity,
                                                    n_phases=self.config.n_phases,
                                                    n_epochs_per_phase=self.config.n_epochs_per_phase)
        elif self.config.strategy == 'struct_Decomp':
            return structured_strategies.struct_Decomp(desired_sparsity=self.config.goal_sparsity,
                                                       n_epochs_per_phase=self.config.n_epochs_per_phase,
                                                       decomp_type=self.config.decomp_type)

        #### UNSTRUCTURED
        elif self.config.strategy == 'Dense':
            assert self.config.group_penalty is None, "group_penalty is intended for use with struct_Dense."
            return unstructured_strategies.Dense()

        elif self.config.strategy in ['IMP', 'Restricted_IMP']:
            assert self.config.IMP_selector in ['global']
            # Select the correct class using the IMP_selector variable
            selector_to_class = {
                'global': unstructured_strategies.IMP,
            }
            return selector_to_class[self.config.IMP_selector](desired_sparsity=self.config.goal_sparsity,
                                                               n_phases=self.config.n_phases,
                                                               n_epochs_per_phase=self.config.n_epochs_per_phase)

    def log(self, runTime, finetuning: bool = False, desired_sparsity=None):
        loggingDict = self.get_metrics()
        self.strategy.start_forward_mode()
        loggingDict.update({'epoch_run_time': runTime})
        if not finetuning:
            if self.config.goal_sparsity is not None:
                distance_to_pruned, rel_distance_to_pruned = metrics.get_distance_to_pruned(model=self.model,
                                                                                            sparsity=self.config.goal_sparsity)
                loggingDict.update({'distance_to_pruned': distance_to_pruned,
                                    'relative_distance_to_pruned': rel_distance_to_pruned})

            # Update final trained metrics (necessary to be able to filter via wandb)
            for metric_type, val in loggingDict.items():
                wandb.run.summary[f"trained.{metric_type}"] = val
            if self.totalTrainTime:
                # Total train time captured, hence training is done
                wandb.run.summary["trained.total_train_time"] = self.totalTrainTime
            # The usual logging of one epoch
            wandb.log(
                loggingDict
            )

        else:
            if desired_sparsity is not None:
                finalDict = dict(finetune=loggingDict,
                                 pruned=self.after_pruning_metrics[desired_sparsity],  # Metrics directly after pruning
                                 desired_sparsity=desired_sparsity,
                                 total_finetune_time=self.totalFinetuneTime,
                                 )
                wandb.log(finalDict)
                # Dump sparsity distribution to json and upload
                sparsity_distribution = metrics.per_layer_sparsity(model=self.model)
                fPath = os.path.join(wandb.run.dir, f'sparsity_distribution_{desired_sparsity}.json')
                with open(fPath, 'w') as fp:
                    json.dump(sparsity_distribution, fp)
                wandb.save(fPath)

            else:
                wandb.log(
                    dict(finetune=loggingDict,
                         ),
                )
        self.strategy.end_forward_mode()

    def final_log(self, actual_sparsity=None):
        s = self.config.goal_sparsity
        if actual_sparsity is not None and self.config.goal_sparsity is None:
            wandb.run.summary[f"actual_sparsity"] = actual_sparsity
            s = actual_sparsity
        elif self.config.goal_sparsity is None:
            # Note: This function may only be called if a desired_sparsity has been given upfront, i.e. GSM etc
            raise AssertionError("Final logging was called even though no goal_sparsity was given.")

        # Recompute accuracy and loss
        sys.stdout.write(
            f"\nFinal logging\n")
        self.reset_averaged_metrics()
        self.evaluate_model(data='train')
        self.evaluate_model(data='test')
        # Update final trained metrics (necessary to be able to filter via wandb)
        loggingDict = self.get_metrics()
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"final.{metric_type}"] = val

        # Update after prune metrics
        for logged_s in self.after_pruning_metrics.keys():
            if abs(logged_s - s) <= 1e-7:  # To avoid numerical errors
                for metric_type, val in self.after_pruning_metrics[logged_s].items():
                    wandb.run.summary[f"pruned.{metric_type}"] = val
                break

    def after_pruning_callback(self, desired_sparsity: float, prune_momentum: bool = False,
                               reset_momentum: bool = False) -> None:
        # Note: This function should only be called when the pruning is PERMANENT,
        # This function must be called once for every sparsity, directly after pruning

        # Make the pruning permanent (this is in conflict with strategies that do not have a permanent pruning)
        self.strategy.make_pruning_permanent()

        # Handle the case of missing constraints when the model was decomposed into more layers
        if self.config.strategy == 'struct_Decomp':
            # Reinit the optimizer
            self.define_optimizer_scheduler()  # This results in SGD being used when original training was with SFW

        # Compute losses, accuracies after pruning
        sys.stdout.write(f"\nDesired sparsity {desired_sparsity} - Computing incurred losses after pruning.\n")
        if self.config.strategy in ['BIMP_LC', 'BIMP_LCT']:
            before_pruning_train_accuracy = self.metrics['train']['accuracy'].compute()
        self.reset_averaged_metrics()
        self.evaluate_model(data='train')
        self.evaluate_model(data='test')
        if self.squared_model_norm is not None:
            L2_norm_square = Utils.get_model_norm_square(self.model)
            norm_drop = sqrt(abs(self.squared_model_norm - L2_norm_square))
            if float(sqrt(self.squared_model_norm)) > 0:
                relative_norm_drop = norm_drop / float(sqrt(self.squared_model_norm))
            else:
                relative_norm_drop = {}
        else:
            norm_drop, relative_norm_drop = {}, {}

        pruning_instability, pruning_stability = {}, {}
        train_loss_increase, relative_train_loss_increase_factor = {}, {}
        if self.trained_test_accuracy is not None and self.trained_test_accuracy > 0:
            pruning_instability = (
                                          self.trained_test_accuracy - self.metrics['test'][
                                      'accuracy'].compute()) / self.trained_test_accuracy
            pruning_stability = 1 - pruning_instability
        if self.trained_train_loss is not None and self.trained_train_loss > 0:
            train_loss_increase = self.metrics['train']['loss'].compute() - self.trained_train_loss
            relative_train_loss_increase_factor = train_loss_increase / self.trained_train_loss

        self.after_pruning_metrics[desired_sparsity] = dict(
            train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items()},
            test={metric_name: metric.compute() for metric_name, metric in self.metrics['test'].items()},
            norm_drop=norm_drop,
            relative_norm_drop=relative_norm_drop,
            pruning_instability=pruning_instability,
            pruning_stability=pruning_stability,
            train_loss_increase=train_loss_increase,
            relative_train_loss_increase_factor=relative_train_loss_increase_factor,
        )
        if reset_momentum:
            sys.stdout.write(
                f"Resetting momentum_buffer (if existing) for potential finetuning.\n")
            self.strategy.reset_momentum(optimizer=self.optimizer)
        elif prune_momentum:
            sys.stdout.write(
                f"Pruning momentum_buffer (if existing).\n")
            self.strategy.prune_momentum(optimizer=self.optimizer)

    def change_weight_decay_callback(self, penalty):
        for group in self.optimizer.param_groups:
            group['weight_decay'] = penalty
        print(f"Changed weight decay to {penalty}.")

    def restore_model(self) -> None:
        sys.stdout.write(
            f"Restoring model from {self.checkpoint_file}.\n")
        self.model = self.get_model(reinit=False, temporary=True)

        if self.config.optimizer == 'SFW':
            # Make sure that initial layer weights are in feasible region
            constraints.make_feasible(constraintList=self.constraintList)

    def save_model(self, model_type: str, remove_pruning_hooks: bool = False, temporary: bool = False) -> str:
        if model_type not in ['initial', 'trained']:
            print(f"Ignoring to save {model_type} for now.")
            return None
        fName = f"{model_type}_model.pt"
        fPath = os.path.join(wandb.run.dir, fName) if not temporary else os.path.join(self.tmp_dir, fName)
        if remove_pruning_hooks:
            self.strategy.remove_pruning_hooks(model=self.model)
        torch.save(self.model.state_dict(), fPath)  # Save the state_dict
        return fPath

    def evaluate_model(self, data='train'):
        return self.train_epoch(data=data, is_training=False)

    def define_retrain_schedule(self, n_epochs_finetune, desired_sparsity, pruning_sparsity, phase):
        """Define the retraining schedule.
            - Tuneable schedules all require both an initial value as well as a warmup length
            - Fixed schedules require no additional parameters and are mere conversions such as LRW
        """
        tuneable_schedules = ['constant',  # Constant learning rate
                              'stepped',  # Stepped Budget Aware Conversion (BAC)
                              'cosine',  # Cosine from initial value -> 0
                              'cosine_min',  # Same as 'cosine', but -> smallest original lr
                              'linear',  # Linear from initial value -> 0
                              'linear_min'  # Same as 'linear', but -> smallest original lr
                              ]
        fixed_schedules = ['FT',  # Use last lr of original training as schedule (Han et al.), no warmup
                           'LRW',  # Learning Rate Rewinding (Renda et al.), no warmup
                           'SLR',  # Scaled Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'CLR',  # Cyclic Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'LLR',  # Linear from the largest original lr to 0, maxLR init, 10% warmup
                           ]

        # Define the initial lr, max lr and min lr
        maxLR = max(
            self.strategy.lr_dict.values())
        after_warmup_index = self.config.n_epochs_warmup or 0
        minLR = min(list(self.strategy.lr_dict.values())[after_warmup_index:])  # Ignores warmup in orig. schedule

        n_total_iterations = len(self.trainLoader) * n_epochs_finetune
        if self.config.retrain_schedule in tuneable_schedules:
            assert self.config.retrain_schedule_init is not None
            assert self.config.retrain_schedule_warmup is not None

            n_warmup_iterations = int(self.config.retrain_schedule_warmup * n_total_iterations)
            after_warmup_lr = self.config.retrain_schedule_init
        elif self.config.retrain_schedule in fixed_schedules:
            assert self.config.retrain_schedule_init is None
            assert self.config.retrain_schedule_warmup is None

            # Define warmup length
            if self.config.retrain_schedule in ['FT', 'LRW']:
                n_warmup_iterations = 0
            else:
                # 10% warmup
                n_warmup_iterations = int(0.1 * n_total_iterations)

            # Define the after_warmup_lr
            if self.config.retrain_schedule == 'FT':
                after_warmup_lr = minLR
            elif self.config.retrain_schedule == 'LRW':
                after_warmup_lr = self.strategy.lr_dict[self.config.nepochs - n_epochs_finetune + 1]
            elif self.config.retrain_schedule in ['SLR', 'CLR', 'LLR']:
                after_warmup_lr = maxLR
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Set the optimizer lr
        for param_group in self.optimizer.param_groups:
            if n_warmup_iterations > 0:
                # If warmup, then we actually begin with 0 and increase to after_warmup_lr
                param_group['lr'] = 0.0
            else:
                param_group['lr'] = after_warmup_lr

        # Define warmup scheduler
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_warmup_iterations, eta_min=after_warmup_lr)
            milestone = n_warmup_iterations + 1

        # Define scheduler after the warmup
        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = None
        if self.config.retrain_schedule in ['FT', 'constant']:
            # Does essentially nothing but keeping the smallest learning rate
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif self.config.retrain_schedule == 'LRW':
            rewind_epoch = self.config.nepochs - n_epochs_finetune + 1
            # Convert original schedule to iterations
            iterationsLR = []
            for ep in range(rewind_epoch, self.config.nepochs + 1, 1):
                lr = self.strategy.lr_dict[ep]
                iterationsLR = iterationsLR + len(self.trainLoader) * [lr]
            iterationsLR = iterationsLR[(-n_remaining_iterations):]
            iterationsLR.append(iterationsLR[-1])  # Double the last learning rate so we avoid the IndexError
            scheduler = FixedLR(optimizer=self.optimizer, lrList=iterationsLR)

        elif self.config.retrain_schedule in ['stepped', 'SLR']:
            epochLR = [self.strategy.lr_dict[i] if i >= after_warmup_index else maxLR
                       for i in range(1, self.config.nepochs + 1, 1)]
            # Convert original schedule to iterations
            iterationsLR = []
            for lr in epochLR:
                iterationsLR = iterationsLR + len(self.trainLoader) * [lr]

            interpolation_width = (len(self.trainLoader) * len(
                epochLR)) / n_remaining_iterations  # In general not an integer
            reducedLRs = [iterationsLR[int(j * interpolation_width)] for j in range(n_remaining_iterations)]
            # Add a last LR to avoid IndexError
            reducedLRs = reducedLRs + [reducedLRs[-1]]

            lr_lambda = lambda it: reducedLRs[it] / float(maxLR)  # Function returning the correct learning rate factor
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif self.config.retrain_schedule in ['CLR', 'cosine']:
            stopLR = 0. if self.config.retrain_schedule == 'cosine' else minLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_remaining_iterations, eta_min=stopLR)

        elif self.config.retrain_schedule in ['LLR', 'linear']:
            endFactor = 0.
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=endFactor,
                                                          total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [after_warmup_lr for group in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

        self.define_wd_scheduler(retrain=True)

    def fine_tuning(self, desired_sparsity, n_epochs_finetune, phase=1):
        n_phases = self.config.n_phases or 1
        # Get the pruning sparsity to use it in LCN if necessary
        if self.config.n_phases is not None:
            pruning_sparsity = 1 - (1 - self.config.goal_sparsity) ** (1. / self.config.n_phases)
        else:
            # Iterative Renda case
            pruning_sparsity = 0.2

        if phase == 1:
            self.finetuneStartTime = time.time()

        # Reset the GradScaler for AutoCast
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))

        # Update the retrain schedule individually for every phase/cycle
        self.define_retrain_schedule(n_epochs_finetune=n_epochs_finetune, desired_sparsity=desired_sparsity,
                                     pruning_sparsity=pruning_sparsity, phase=phase)

        self.strategy.set_to_finetuning_phase()
        for epoch in range(1, n_epochs_finetune + 1, 1):
            self.reset_averaged_metrics()
            sys.stdout.write(
                f"\nDesired sparsity {desired_sparsity} - Finetuning: phase {phase}/{n_phases} | epoch {epoch}/{n_epochs_finetune}\n")
            # Train
            t = time.time()
            self.train_epoch(data='train')
            self.evaluate_model(data='test')
            sys.stdout.write(
                f"\nTest accuracy after this epoch: {self.metrics['test']['accuracy'].compute()} (lr = {float(self.optimizer.param_groups[0]['lr'])})\n")

            if epoch == n_epochs_finetune and phase == n_phases:
                # Training complete, log the time
                self.totalFinetuneTime = time.time() - self.finetuneStartTime

            # As opposed to previous runs, push information every epoch, but only link to desired_sparsity at end
            dsParam = None
            if epoch == n_epochs_finetune and phase == n_phases:
                dsParam = desired_sparsity
            self.log(runTime=time.time() - t, finetuning=True, desired_sparsity=dsParam)

    def train_epoch(self, data='train', is_training=True):
        assert not (data == 'test' and is_training), "Can't train on test set."
        loader = self.trainLoader if data == 'train' else self.testLoader
        sys.stdout.write(f"Training:\n") if is_training else sys.stdout.write(f"Evaluation of {data} data:\n")

        with torch.set_grad_enabled(is_training):
            for x_input, y_target in Bar(loader):
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)
                y_target = y_target.to(self.device, non_blocking=True)
                if self.config.channels_last:
                    x_input = x_input.to(self.device, memory_format=torch.channels_last)
                self.optimizer.zero_grad()  # Zero the gradient buffers
                self.strategy.start_forward_mode(enable_grad=is_training)
                if is_training:
                    with autocast(enabled=(self.config.use_amp is True)):
                        output = self.model.train()(x_input)
                        loss = self.loss_criterion(output, y_target)
                    loss = self.strategy.before_backward(loss=loss, weight_decay=self.config.weight_decay,
                                                         group_penalty=self.config.group_penalty,
                                                         group_penalty_type=self.config.group_penalty_type)

                    self.ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                    # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                    # we might add something to the grads with during_training(), this has to be split
                    self.ampGradScaler.unscale_(self.optimizer)
                    # Potentially update the gradients
                    self.strategy.during_training(opt=self.optimizer, trainIteration=self.trainIterationCtr)
                    self.ampGradScaler.step(self.optimizer)
                    self.ampGradScaler.update()

                    self.strategy.end_forward_mode()  # Has no effect for DPF
                    self.strategy.after_training_iteration(it=self.trainIterationCtr)
                    self.scheduler.step()
                    if self.wd_scheduler:
                        self.wd_scheduler.step()
                    self.trainIterationCtr += 1
                else:
                    with autocast(enabled=(self.config.use_amp is True)):
                        output = self.model.eval()(x_input)
                        loss = self.loss_criterion(output, y_target)
                    self.strategy.end_forward_mode()  # Has no effect for DPF

                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                self.metrics[data]['accuracy'](output, y_target)
                self.metrics[data]['k_accuracy'](output, y_target)

    def train(self):
        trainStartTime = time.time()
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))
        for epoch in range(self.config.nepochs + 1):
            self.reset_averaged_metrics()
            sys.stdout.write(f"\n\nEpoch {epoch}/{self.config.nepochs}\n")
            t = time.time()
            if epoch == 0:
                # Just evaluate the model once to get the metrics
                if self.debug_mode:
                    # Skip this step
                    sys.stdout.write(f"Skipping since we are in debug mode")
                    continue
                self.evaluate_model(data='train')
                epoch_lr = float(self.optimizer.param_groups[0]['lr'])
            else:
                # Train
                self.train_epoch(data='train')
                # Save the learning rate for potential rewinding before updating
                epoch_lr = float(self.optimizer.param_groups[0]['lr'])
            self.evaluate_model(data='test')

            self.strategy.at_epoch_end(epoch=epoch, epoch_lr=epoch_lr,
                                       train_loss=self.metrics['train']['loss'].compute(),
                                       train_acc=self.metrics['train']['accuracy'].compute(), optimizer=self.optimizer)

            # Save info to wandb
            if epoch == self.config.nepochs:
                # Training complete, log the time
                self.totalTrainTime = time.time() - trainStartTime
            self.log(runTime=time.time() - t)

            if self.lmo_delay_epoch == epoch:
                print("Changing constraints.")
                self.change_constraints()

        self.trained_test_accuracy = self.metrics['test']['accuracy'].compute()
        self.trained_train_loss = self.metrics['train']['loss'].compute()
