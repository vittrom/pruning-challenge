# pruning-challenge

The code in this repository investigates the differences between weight and unit prunining in neural networks. We train two neural networks on MNIST and fashion MNIST, apply weight/unit pruning on all layers but the last layer, with different pruning rates and compare the performance (accuracy) of the networks as the pruning rate increases.

This repository contains the following files:

- requirements.txt : requirements file to reproduce the results
- utils.py : utils file implementing the two pruning methods, the functions used to preprocess the data, construct and prune the network
- main.py : file to fit the network and produce the plots
- pruning_results_mnist.png : visualization of results for MNIST
- pruning_results_fashion_mnist.png : visualization of results for fashion MNIST

### Requirements
- python3.6
- keras==2.2.4
- tensorflow==1.12.0
- numpy==1.16.4
- matplotlib==3.1.0

### Performance
Given the results below, weight pruning seems to be a more robust method than unit pruning. Using weight pruning, we can prune 90% of the network while maintaining the accuracy above 80%. On the other hand, using unit pruning results in a significant drop in accuracy when 90% of the network is pruned. By the implementation of unit 

<p float="left">
  <img src="https://github.com/vittrom/pruning-challenge/blob/master/pruning_results_mnist.png" width="400" />
  <img src="https://github.com/vittrom/pruning-challenge/blob/master/pruning_results_fashion_mnist.png" width="400">
</p>
