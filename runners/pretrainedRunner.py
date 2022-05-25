# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         pretrainedRunner.py
# Description:  Runner class for starting from a pretrained model
# ===========================================================================
import sys
import warnings
import numpy as np
import torch
from optimizers.optimizers import SGD, SFW
import wandb

from runners.baseRunner import baseRunner
from utilities.utilities import Utilities as Utils


class pretrainedRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_run = None

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the model file."""
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        # Some variables have to be extracted from the filterDict and checked manually, e.g. weight decay in scientific format
        manualVariables = ['weight_decay', 'group_penalty', 'prox_threshold']
        manVarDict = {}
        dropIndices = []
        for var in manualVariables:
            for i in range(len(filterDict['$and'])):
                entry = filterDict['$and'][i]
                s = f"config.{var}"
                if s in entry:
                    dropIndices.append(i)
                    manVarDict[var] = entry[s]
        for idx in reversed(dropIndices): filterDict['$and'].pop(idx)

        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state == 'failed':
                # Ignore this run
                continue
            # Check if run satisfies the manual variables
            conflict = False
            for var, val in manVarDict.items():
                if run.config[var] != val:
                    conflict = True
                    break
            if conflict:
                continue

            self.checkpoint_file = run.summary.get('trained_model_file')
            try:
                if self.checkpoint_file is not None:
                    runsExist = True
                    run.file(self.checkpoint_file).download(root=self.tmp_dir)
                    self.seed = run.config['seed']
                    self.reference_run = run
                    break
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print("Exception:", e)
                self.checkpoint_file = None
        assert not (
                runsExist and self.checkpoint_file is None), "Runs found, but none of them have a model available -> abort."
        outputStr = f"Found {self.checkpoint_file} in run {run.name}" \
            if self.checkpoint_file is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference trained model in project: {outputStr}\n")
        assert self.checkpoint_file is not None, "No reference trained model found, Aborting."

    def get_missing_config(self):
        missing_config_keys = ['momentum',
                               'nesterov',
                               'nepochs',
                               'n_epochs_warmup',
                               'decouple_wd',
                               'lmo_rescale']
        additional_dict = {
            'last_training_lr': self.reference_run.summary['trained.learning_rate'],
            'trained.test.accuracy': self.reference_run.summary['trained.test']['accuracy'],
            'trained.train.loss': self.reference_run.summary['trained.train']['loss'],
        }

        if self.config.strategy == 'struct_Decomp':
            # Collect additional data
            needed_values = ['trained.nonzero_inference_flops', 'trained.n_nonzero_params']
            self.struct_cmp_metrics = {key: self.reference_run.summary[key] for key in needed_values}

        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = self.reference_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict['trained.test.accuracy']
        self.trained_train_loss = additional_dict['trained.train.loss']

    def define_optimizer_scheduler(self):
        # Define the optimizer using the parameters from the reference run
        if self.config.optimizer == 'SGD' or (
                self.config.optimizer == 'SFW' and self.config.strategy == 'struct_Decomp') \
                or (self.config.optimizer == 'Prox_SGD' and self.config.strategy == 'struct_Decomp'):
            used_FW = (self.config.optimizer == 'SFW' and self.config.strategy == 'struct_Decomp')
            if used_FW:
                wd = 0.
            elif self.config.retrain_wd is not None:
                wd = self.config['retrain_wd']
            else:
                wd = self.config['weight_decay']

            self.optimizer = SGD(params=self.model.parameters(), lr=self.config['last_training_lr'],
                                 momentum=self.config['momentum'],
                                 weight_decay=wd,
                                 nesterov=self.config['nesterov'],
                                 decouple_wd=self.config['decouple_wd'],
                                 extensive_metrics=self.config['extensive_metrics'],
                                 device=self.config.device)

        elif self.config.optimizer == 'SFW':
            param_groups = [{'params': param_list, 'constraint': constraint}
                            for constraint, param_list in self.constraintList]
            self.optimizer = SFW(params=param_groups, lr=self.config['last_training_lr'],
                                 rescale=self.config['lmo_rescale'], momentum=self.config['momentum'],
                                 extensive_metrics=self.config['extensive_metrics'],
                                 device=self.config.device)

    def fill_strategy_information(self):
        # Get the wandb information about lr's and losses and fill the corresponding strategy dicts, which can then be used by rewinders
        # Note: this only works if the reference model has "Dense" strategy
        for row in self.reference_run.history(keys=['learning_rate', 'train.loss', 'train.accuracy'], pandas=False):
            epoch, epoch_lr, train_loss, train_acc = row['_step'], row['learning_rate'], row['train.loss'], row[
                'train.accuracy']
            self.strategy.at_epoch_end(epoch=epoch, epoch_lr=epoch_lr, train_loss=train_loss, train_acc=train_acc)

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        # Find the reference run
        filterDict = {"$and": [{"config.run_id": self.config.run_id},
                               {"config.model": self.config.model},
                               {"config.optimizer": self.config.optimizer},
                               ]}
        if 'struct_' in self.config.strategy:
            filterDict["$and"].append({"config.strategy": "struct_Dense"})
        elif self.config.strategy == 'GSM_IMP':
            filterDict["$and"].append({"config.strategy": "GSM_Dense"})
        else:
            filterDict["$and"].append({"config.strategy": "Dense"})

        assert self.config.optimizer in ['SGD', 'SFW', 'Prox_SGD']
        if self.config.optimizer == 'SGD':
            attributeList = ['weight_decay', 'weight_decay_schedule', 'group_penalty', 'group_penalty_type',
                             'decouple_wd']
            if 'momentum' in self.config.keys() and self.config.momentum is not None:
                attributeList.append('momentum')  # Add this manually because we omit this except for GSM

        elif self.config.optimizer == 'SFW':
            attributeList = ['lmo', 'lmo_mode', 'lmo_ord', 'lmo_value', 'lmo_k', 'lmo_rescale', 'lmo_global',
                             'lmo_delay', 'lmo_nuc_method', 'lmo_adjust_diameter']

        elif self.config.optimizer == 'Prox_SGD':
            attributeList = ['weight_decay', 'weight_decay_schedule', 'group_penalty', 'group_penalty_type',
                             'decouple_wd', 'prox_threshold']

        for attr in attributeList:
            name, val = f"config.{attr}", self.config[attr]
            filterDict["$and"].append({name: val})

        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.learning_rate": self.config.learning_rate})

        self.find_existing_model(filterDict=filterDict)

        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True
        self.get_missing_config()  # Load keys that are missing in the config

        self.trainLoader, self.testLoader = self.get_dataloaders()
        self.model = self.get_model(reinit=True)  # Load the trained model
        self.define_constraints(retrain=True)
        self.define_optimizer_scheduler()
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

        # Define strategy
        self.strategy = self.define_strategy()
        self.strategy.after_initialization(model=self.model)  # To ensure that all parameters are properly set
        self.fill_strategy_information()
        # Run the computations
        self.strategy.at_train_end(model=self.model, finetuning_callback=self.fine_tuning,
                                   restore_callback=self.restore_model, save_model_callback=self.save_model,
                                   after_pruning_callback=self.after_pruning_callback, opt=self.optimizer)

        self.strategy.final(model=self.model, final_log_callback=self.final_log)
