# Final Task1
> The SimCLR component of this repository is adapted from `p3i0t`'s project [SimCLR-CIFAR10](https://github.com/p3i0t/SimCLR-CIFAR10)


## Dependencies
* pytorch >=1.2
* torchvision >=0.4.0
* hydra >=0.11.3
* tqdm >=4.45.0

### Install Hydra
[Hydra](https://hydra.cc/docs/next/intro/#installation) is a python framework to manage the hyperparameters during
 training and evaluation. Install with:
 
 ``pip install hydra-core --upgrade``

## Code Structure
```pydocstring
models.py           # Define SimCLR model.
simclr.py           # Code to train simclr.
simclr_config.yml   # Config File with all default hyperparameters in training.
main.py             # code to train ResNet18
```

## Usage

Train SimCLR with  ``resnet18`` as backbone:

``python simclr.py backbone=resnet18``


The default ``batch_size`` is 256. All the hyperparameters are available in ``simclr_config.yml``,
 which could be overrided from the command line.

Config the parameters in `main.py` and run `python main.py` to train the final `ResNet18` model


