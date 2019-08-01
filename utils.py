from keras.models import Sequential
from keras.layers import Flatten, Dense
import copy
import numpy as np

def create_and_compile_network(input_dims, layers_size, activations,
                                loss, metrics, optimizer, bias=False):
        '''
        This functions defines a neural network model and compiles it.
        Assumes the model takes images as input and that all network layers are dense layers.
        :param input_dims: list or tuples containing the dimensions of the input images
        :param layers_size: list of integers containing the number of neurons in each dense layer of the network
        :param activations: list of strings specifying the activation function for each dense layer
        :param loss: string specifying the loss function to use in training
        :param metrics: list of strings of metrics to track during training
        :param optimizer: string with the name of the optimizer to use for training
        :param bias: boolean specifying whether to use a bias in the network layers
        :return: a compiled keras model
        '''
        nn_layers = [Flatten(input_shape=(input_dims))] + \
                    [Dense(ls, activation=act, use_bias=bias) for ls, act in zip(layers_size, activations)]
        model = Sequential(nn_layers)

        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        return model

def prune_network(model, k, x_test, y_test, pruning_method, dict_pruning):
        '''
        This function applies a pruning method to a trained model and returns the model test accuracy after pruning.
        :param model: a trained keras model
        :param k: integer specifying the pruning percentage
        :param x_test: test set features
        :param y_test: test set labels
        :param pruning_method: string specifying the pruning method to use ("weight" or "unit")
        :param dict_pruning: dictionary containing the different pruning methods supported, can be extended to allow for more pruning methods
        :return: test accuracy of the pruned model
        '''

        pruning_func = dict_pruning[pruning_method]
        weights = model.get_weights()

        pruned_weights = [pruning_func(weights_layer, k) for weights_layer in weights]
        pruned_weights[-1] = weights[-1]

        model.set_weights(pruned_weights)

        score = model.evaluate(x_test, y_test, verbose=0)

        model.set_weights(weights)

        return 100 * score[1]

def weight_pruning(weights, k):
        '''
        This function performs weight pruning on a weight matrix of a trained network.
        :param weights: numpy array of weights to prune
        :param k: pruning percentage
        :return: numpy array of pruned weights
        '''
        w = copy.deepcopy(weights)
        threshold = np.percentile(np.abs(w), k)
        w[np.abs(w) <= threshold] = 0
        return w

def unit_pruning(weights, k):
        '''
        This function performs unit pruning on a weight matrix of a trained network.
        :param weights: numpy array of weights to prune
        :param k: pruning percentage
        :return: numpy array of pruned weights
        '''
        w = copy.deepcopy(weights)
        l2_norm = np.linalg.norm(w, axis=0)
        threshold = np.percentile(l2_norm, k)
        w[:,l2_norm < threshold] = 0
        return w

def preprocess_data(data):
        data = data.astype('float32')
        data /= 255.0
        return data