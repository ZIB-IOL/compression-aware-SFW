## Compression-aware Training of Neural Networks using Frank-Wolfe
*Authors: Max Zimmer, [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the code to reproduce the experiments from the ["Compression-aware Training of Neural Networks using Frank-Wolfe" (arXiv:2205.11921)](https://arxiv.org/abs/2205.11921) paper.
The code is based on [PyTorch 1.9](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai).


### Structure and Usage
Experiments are started from the following file:
- [`main.py`](main.py): Starts experiments using the dictionary format of Weights & Biases.

The rest of the project is structured as follows:
- [`strategies`](strategies): Contains all used sparsification methods.
- [`runners`](runners): Contains classes to control the training and collection of metrics.
- [`metrics`](metrics): Contains all metrics as well as FLOP computation methods.
- [`models`](models): Contains all model architectures used.
- [`optimizers`](optimizers): Contains reimplementations of SFW, SGD and Proximal SGD.


### Citation
In case you find the paper or the implementation useful for your own research, please consider citing:

```
@Article{zimmer2022,
  author        = {Max Zimmer and Christoph Spiegel and Sebastian Pokutta},
  title         = {Compression-aware Training of Neural Networks using Frank-Wolfe},
  year          = {2022},
  archiveprefix = {arXiv},
  eprint        = {2205.11921},
  primaryclass  = {cs.LG},
}
```
