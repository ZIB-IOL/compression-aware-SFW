# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         main.py
# Description:  Starts up a run for the comparison between sparsification strategies
# ===========================================================================
import socket
import sys
import os
import shutil
import torch
import wandb

from runners.scratchRunner import scratchRunner
from runners.pretrainedRunner import pretrainedRunner

# Default wandb parameters
defaults = dict(
    # System
    run_id=None,
    computer=socket.gethostname(),
    fixed_init=None,
    extensive_metrics=False,
    # Setup
    dataset=None,
    model=None,
    nepochs=None,
    batch_size=None,
    # Effiency
    use_amp=True,
    channels_last=False,  # Disabled by default, since pruning will make problems. Enable this for Dense training
    # Optimizer
    optimizer=None,
    learning_rate=None,
    n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
    momentum=None,
    nesterov=None,
    weight_decay=None,
    weight_decay_schedule=None,
    decouple_wd=None,
    group_penalty=None,
    group_penalty_type=None,
    prox_threshold=None,
    # Constraints
    lmo=None,
    lmo_mode=None,
    lmo_ord=None,
    lmo_value=None,
    lmo_k=None,
    lmo_rescale=None,
    lmo_global=None,
    lmo_delay=None,
    lmo_nuc_method=None,
    lmo_adjust_diameter=None,
    # Sparsifying strategy
    strategy=None,
    goal_sparsity=None,
    decomp_type=None,
    # IMP
    IMP_selector=None,  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
    # Retraining
    n_phases=None,  # Should be 1, except when using IMP
    n_epochs_per_phase=None,
    retrain_schedule=None,
    retrain_wd=None,
    retrain_schedule_wd=None,
    retrain_schedule_warmup=None,
    retrain_schedule_init=None,
    dynamic_retrain_length=None,
)
debug_mode = False
if '--debug' in sys.argv:
    debug_mode = True
    defaults.update(dict(
        # System
        run_id=1,
        computer=socket.gethostname(),
        fixed_init=True,
        extensive_metrics=True,
        # Setup
        dataset='mnist',
        model='SimpleCNN',
        nepochs=1,
        batch_size=1024,
        # Efficiency
        use_amp=True,
        channels_last=False,
        # Optimizer
        optimizer='SFW',
        # learning_rate='(MultiStepLR, 0.1, [3|7], 0.1)',
        learning_rate='(Linear, 0.1)',
        n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
        momentum=0.9,
        nesterov=False,
        weight_decay=0.0001,
        weight_decay_schedule=None,
        decouple_wd=False,
        group_penalty=None,
        group_penalty_type='Filter_Conv',
        prox_threshold=1,
        # Constraints
        lmo='SpectralKSupportNormBall',
        lmo_mode='initialization',
        lmo_ord=2,
        lmo_value=10,
        lmo_k=0.3,
        lmo_rescale='fast_gradient',
        lmo_global=False,
        lmo_delay=None,
        lmo_nuc_method='qrpartial',
        lmo_adjust_diameter=True,
        # Sparsifying strategy
        strategy='struct_Dense',
        goal_sparsity=0.5,  # Functions as (1 - goal_energy) simultaneously
        decomp_type='conv',
        # IMP
        IMP_selector='global',  # must be in ['global', 'uniform', 'uniform_plus', 'ERK', 'LAMP']
        # Retraining
        n_phases=1,  # Should be 1, except when using IMP
        n_epochs_per_phase=0,
        retrain_schedule='LLR',
        retrain_wd=0.0005,
        retrain_schedule_wd=None,
        retrain_schedule_warmup=None,
        retrain_schedule_init=None,
        dynamic_retrain_length=None,
    ))

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
ngpus = torch.cuda.device_count()
if ngpus > 0:
    if ngpus > 1 and config.dataset == 'imagenet':
        config.update(dict(device='cuda:' + ','.join(f"{i}" for i in range(ngpus))))
    else:
        config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))

# At the moment, IMP is the only strategy that requires a pretrained model, all others start from scratch
if config.strategy in ['IMP', 'struct_IMP', 'struct_Decomp']:
    # Use the pretrainedRunner
    runner = pretrainedRunner(config=config, debug_mode=debug_mode)
else:
    # Use the scratchRunner
    runner = scratchRunner(config=config, debug_mode=debug_mode)
runner.run()

# Close wandb run
wandb_dir_path = wandb.run.dir
wandb.join()

# Delete the local files
if os.path.exists(wandb_dir_path):
    shutil.rmtree(wandb_dir_path)
# Delete temporary directory
if os.path.exists(runner.tmp_dir):
    shutil.rmtree(runner.tmp_dir)
