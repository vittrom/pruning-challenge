#Set seed for reproducibility
import tensorflow as tf
import random as rn
import os
import numpy as np

os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(12345)

# Setting the seed for python random numbers
rn.seed(1111)

# Setting the graph-level random seed.
tf.set_random_seed(1234567)

from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)

#Read arguments from terminal and perform the experiment

import utils
from utils import *
from keras.datasets import mnist, fashion_mnist
import keras
import argparse
from inspect import getmembers, isfunction

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to use during training')

parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size to use for training')

parser.add_argument('--dataset', type=str, default="mnist",
                    help='Dataset to use mnist/fashion_mnist')

parser.add_argument('--layers_size', nargs="+", type=int,
                    default=[1000, 1000, 500, 200],
                    help="Dimensions of network layers")

parser.add_argument('--pruning_rate', nargs="+", type=int,
                    default=[0, 25, 50, 60, 70, 80, 90, 95, 97, 99],
                    help="List of pruning rates to inspect")

parser.add_argument('--activations', nargs="+",
                    default=["relu", "relu", "relu", "relu"],
                    help="Activations of network layers")

parser.add_argument('--loss', type=str, default="categorical_crossentropy",
                    help="Loss function to use during training")

parser.add_argument('--metrics', nargs="+",
                    default=["accuracy"],
                    help="Metrics to track during training")

parser.add_argument('--optimizer', type=str, default="adam",
                    help="Optimizer to use for training")

parser.add_argument('--save_dir', type=str, default="./",
                    help="Directory to save the comparison plot")

args = parser.parse_args()

print("Loading " + args.dataset)
if args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif args.dataset == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
    raise Exception("The dataset you want to use is not supported")

print(args.dataset + " loaded")

input_dims = x_train[0].shape
num_classes = len(np.unique(y_train))

#preprocess image data
x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

#convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#create network
print("Creating network")
layers_size = args.layers_size + [num_classes]
activations = args.activations + ["softmax"]

if len(layers_size) != len(activations):
    raise Exception("Number of layers and activations differ when they should be equal")
loss = args.loss
metrics = args.metrics
optimizer = args.optimizer

model = create_and_compile_network(input_dims, layers_size, activations, loss, metrics, optimizer)
print("Network created and compiled")

#train network
print("Training network")
epochs = args.epochs
batch_size = args.batch_size

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=False)
print("Network trained")

#pruning exercise
print("Performing pruning")
functions_list = [f for f in getmembers(utils) if isfunction(f[1])]
dict_pruning = dict([t for t in functions_list if "pruning" in t[0]])
pruning_methods = dict_pruning.keys()
pruning_rate = args.pruning_rate

pruning_methods_results = []
for m in pruning_methods:
    accuracy_method = []
    for k in pruning_rate:
        accuracy_method.append(prune_network(model, k, x_test, y_test, m, dict_pruning))
    pruning_methods_results.append([m, accuracy_method])


#plot accuracy decay
print("Plotting results")
save_dir = args.save_dir
import matplotlib.pyplot as plt

for methods in pruning_methods_results:
    plt.plot(pruning_rate, methods[1])
plt.legend([m[0] for m in pruning_methods_results])
plt.xlabel("Percent sparsity")
plt.ylabel("Percent accuracy")
plt.title("Pruning methods comparison " + args.dataset)
plt.savefig(save_dir + "pruning_results_" + args.dataset + ".png")
plt.show()
