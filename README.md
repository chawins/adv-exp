# Adversarial Example Experiments

## Requirement
See `environment.yml`.
- pytorch == 1.4
- torchvision == 0.5.0
- numpy == 1.18.1
- pyyaml == 5.3.1
- foolbox == 3.0.2
- numba == 0.48.0

## File Organization
- There are multiple scripts to run the training and the testing. Main portion of code is in `./adv`.
- The naming of the scripts is simply `SCRIPT_DATASET.py` with the YAML config file under the same name. `DATASET` includes `mnist` and `cifar` which combines both CIFAR-10 and CIFAR-100.
- Scripts
  - `train_DATASET.py`: main script for training AT, TRADES, Dynamic AT and ATES models. The options and hyperparameters can be set in `train_DATASET.yml`.s
  - `test_DATASET.py`: test a trained network under one attack (PGD or BB).
- Library
  - `lib/adv_model.py`: wrapper Pytorch Module for AT, TRADES, Dynamic AT and ATES.
  - `lib/pgd_attack.py`: implements PGD attack.
  - `lib/mnist_model.py`: implements MNIST models.
  - `lib/cifar10_model.py`: implements ResNet.
  - `lib/wideresnet.py`: implements WideResNet.
  - `lib/dataset_utils.py`: handles dataset loading.
  - `lib/utils.py`: other utility functions (e.g., quantization, get logger, etc.)
  
## Usage
- We use YAML config files (.yml) for both training and testing.
- Training
  - To use ATES, set `early_stop: True`.
  - To use TRADES, set `loss_func: 'trades'`.
  - To use Dynamic AT, set `use_fosc: True`.
  - Other options are self-explanatory.
- Testing
  - Only need to specify name of the model to test.
  - To use BB attack, set `bb: True` (we use the default parameters which can be changed in the test scripts).
