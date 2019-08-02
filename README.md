# pruning-challenge

The code in this repository investigates the differences between weight and unit prunining in neural networks. We train two neural networks on MNIST and fashion MNIST, apply weight/unit pruning on all layers, with different pruning rates and compare the performance (accuracy) of the networks as the pruning rate increases.

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
Given the results below, weight pruning seems to be a more robust method than unit pruning. Using weight pruning, we can prune 90% of the network while maintaining the accuracy above 80%. On the other hand, using unit pruning results in a significant drop in accuracy when 90% of the network is pruned. Unit pruning might end up removing neurons that contribute significantly to the accuracy of the model even though the overall L2-norm of the incoming weights to the neuron is not large. For instance this could happen when after a high dimensional layer a few incoming weights to a neuron are large in magnitude while the majority of the weights is 0. In contrast, weight pruning would keep the weights of large magnitude, preserving the model accuracy.

<img src="https://github.com/vittrom/pruning-challenge/blob/master/pruning_results_mnist.png" width="400"/><img src="https://github.com/vittrom/pruning-challenge/blob/master/pruning_results_fashion_mnist.png" width="400"/>

Overall, the results show that pruning about 60% of the weights maintains the network performance unchanged. This is not surprising for highly parameterised models such as neural networks. Most likely, such overparameterisation leads to overfitting (which probably explains why unit pruning on mnist improves performance as the pruning rate changes from 97 to 99%). Furthermore, there might be multicollinearity between the outputs of each layer which also leads to overfitting. This result can be used to speed up training and reduce the size of the network. After performing pruning on the weights of a network, dense layers can be converted to sparse layers speeding up matrix multiplications performed during training. Furthermore, when storing the network, sparse matrices can be used instead of dense matrices, reducing the memory required to save the network.

### Running
After cloning the repository and installing the requirements, the code can be run from the terminal with the following command

    python main.py
    
the code also accepts arguments from the command line, for a list of the accepted arguments and a description you can run
   
    python main.py --help  

