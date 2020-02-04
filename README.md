# Edited version of CIFAR10 Training
This version saves every 10th epoch during training in addition to the best trained version. This to allow for testing at different stages, which is needed for outlier detecting methods evaluated at different training stages. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Calls:

For [Performance Analysis of Out-of-Distribution Detection on Various Trained Neural Networks](https://ieeexplore.ieee.org/abstract/document/8906748)

[SEAA training](https://github.com/jenshenriksson/pytorch-cifar/blob/master/SEAA_training.sh)

[IST-Journal](https://github.com/jenshenriksson/pytorch-cifar/blob/master/IST_training.sh)

