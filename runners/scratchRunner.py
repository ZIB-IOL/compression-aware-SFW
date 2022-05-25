# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         scratchRunner.py
# Description:  Runner class for methods that do *not* start from a pretrained model
# ===========================================================================
import os
import sys
import time
import numpy as np
import torch

import wandb

from utilities.utilities import Utilities as Utils
from runners.baseRunner import baseRunner


class scratchRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.artifact = None

        entity, project = wandb.run.entity, wandb.run.project
        self.initial_artifact_name = f"initial-{entity}-{project}-{self.config.model}-{self.config.dataset}-{self.config.run_id}"

    def find_existing_model(self):
        """Finds an existing wandb artifact and downloads the initial model file."""
        # Create a new artifact, this is idempotent, i.e. no artifact is created if this already exists
        try:
            self.artifact = wandb.run.use_artifact(f"{self.initial_artifact_name}:latest")
            seed = self.artifact.metadata["seed"]
            self.artifact.download(root=self.tmp_dir)
            self.checkpoint_file = os.path.join(self.tmp_dir, 'initial_model.pt')
            self.seed = seed

        except Exception as e:
            print(e)

        outputStr = f"Found {self.initial_artifact_name} with seed {seed}" if self.artifact is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference initial model in project: {outputStr}\n")

    def run(self):
        """Function controlling the workflow of scratchRunner"""
        if self.config.fixed_init:
            # If not existing, start a new model, otherwise use existing one with same run-id
            self.find_existing_model()

        if self.seed is None:
            # Generate a random seed
            self.seed = int((os.getpid() + 1) * time.time()) % 2 ** 32

        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True

        self.trainLoader, self.testLoader = self.get_dataloaders()
        self.model = self.get_model(reinit=True, temporary=True)
        # Save initial model before training
        if self.config.fixed_init:
            if self.artifact is None:
                self.artifact = wandb.Artifact(self.initial_artifact_name, type='model', metadata={'seed': self.seed})
                sys.stdout.write(f"Creating {self.initial_artifact_name}.\n")
                self.save_model(model_type='initial', temporary=True)
                self.artifact.add_file(f"{self.tmp_dir}/initial_model.pt")
                wandb.run.use_artifact(self.artifact)
        self.strategy = self.define_strategy()
        self.strategy.after_initialization(model=self.model)
        self.define_constraints()
        self.define_optimizer_scheduler()

        self.strategy.at_train_begin(model=self.model, LRScheduler=self.scheduler)

        # Do initial prune if necessary
        self.strategy.initial_prune()

        # Do proper training
        self.train()
        # Save trained (unpruned) model, if we are doing regular training
        if self.config.strategy in ['Dense', 'struct_Dense']:
            self.checkpoint_file = self.save_model(model_type='trained')
            wandb.summary['trained_model_file'] = 'trained_model.pt'
        else:
            # Save to temporary folder
            self.checkpoint_file = self.save_model(model_type='trained', temporary=True)

        self.strategy.start_forward_mode()
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

        self.strategy.end_forward_mode()

        # Potentially finetune the final model
        self.strategy.at_train_end(model=self.model, finetuning_callback=self.fine_tuning,
                                   restore_callback=self.restore_model, save_model_callback=self.save_model,
                                   after_pruning_callback=self.after_pruning_callback, opt=self.optimizer)

        self.strategy.final(model=self.model, final_log_callback=self.final_log)
