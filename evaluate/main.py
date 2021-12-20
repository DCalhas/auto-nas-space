import tensorflow as tf

from data import cifar10, imagenet, mnist

from models import resnet

from utils import parse_conv, tf_config, train, print_utils

import argparse

import os

import h5py

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['original', 'limited', 'uni_wit'], help="Which method to use")
parser.add_argument('model', choices=['resnet'], help="Which model to run")
parser.add_argument('dataset', choices=['cifar10', 'imagenet', 'mnist'], help="Which dataset to load")
parser.add_argument('-network', default=0, type=int, choices=list(range(20)), help="Which generated neural network should you run?")
parser.add_argument('-na_path', default=str(Path.home())+"/auto-nas-space/evaluate/models/", 
								type=str, help="What is the directory of the neural architectures?")
parser.add_argument('-gpu_mem', default=800, type=int, help="GPU memory limit")
parser.add_argument('-batch_size', default=256, type=int, help="Batch size to use in training.")
parser.add_argument('-learning_rate', default=0.1, type=float, help="Learning rate for training session.")
parser.add_argument('-epochs', default=200, type=int, help="Number epochs for training session.")
parser.add_argument('-out_file', default="out.txt", type=str, help="Output file")
opt = parser.parse_args()

method = opt.method
model = opt.model
dataset = opt.dataset
network = opt.network
na_path = opt.na_path
gpu_mem = opt.gpu_mem
batch_size = opt.batch_size
learning_rate = opt.learning_rate
out_file = opt.out_file
epochs = opt.epochs

file_output = open(out_file, 'w')

tf_config.setup_tensorflow(memory_limit=gpu_mem)

if(dataset=="cifar10"):
	with tf.device('/CPU:0'):
		x_train, y_train, x_test, y_test = cifar10.get()
elif(dataset=="imagenet"):
	with tf.device('/CPU:0'):
		x_train, y_train, x_test, y_test = imagenet.get()
elif(dataset=="mnist"):
	with tf.device('/CPU:0'):
		x_train, y_train, x_test, y_test = mnist.get()

#create datasets
train_set, test_set = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size), 
						tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1))

if(method == "original"):
	if(model=="resnet"):
		with tf.device('/CPU:0'):
			nn_model = resnet.ResNet50(include_top=True,
						weights=None, input_tensor=None,
						input_shape=x_train.shape[1:], pooling=None, classes=y_train.shape[1])

elif(method=="uni_wit" or method=="limited"):
	if(model=="resnet"):
		with tf.device('/CPU:0'):
			neural_architecture = tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1))
			kernels, strides = parse_conv.cnn_to_tuple(neural_architecture)


			nn_model = resnet.ResNet50(include_top=True,  mode="generated",
						weights=None, input_tensor=None, input_shape=x_train.shape[1:],
						conv_kernel=kernels, conv_strides=strides, pooling=None, classes=y_train.shape[1])

#train the model
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#lr_scheduler = tf.keras.experimental.CosineDecay(learning_rate, 200)
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss, test_history = train.train(train_set, nn_model, 
									optimizer, loss_fn, 
									epochs=epochs, val_set=test_set, 
									file_output=file_output,
									verbose=True)

print_utils.print_message(train_loss, file_output=file_output, verbose=True)
print_utils.print_message(test_history, file_output=file_output, verbose=True)
print_utils.print_message("Final test accuracy: " + str(test_history[-1][1]), file_output=file_output, verbose=True)

file_output.close()