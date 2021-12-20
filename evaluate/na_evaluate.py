import sys

sys.path.append("..")

import argparse

import os

import h5py

import numpy as np

from multiprocessing import Process, shared_memory, Manager

import math

from pathlib import Path

import time

from utils import import_utils, batch_utils, process_utils, batch_prediction_utils


parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['original', 'limited', 'uni_wit'], help="Which method to use")
parser.add_argument('model', choices=['resnet'], help="Which model to run")
parser.add_argument('dataset', choices=['cifar10', 'imagenet', 'mnist'], help="Which dataset to load")
parser.add_argument('-networks', default=20, type=int, help="Number of networks being evaluated.")
parser.add_argument('-na_path', default=str(Path.home())+"/auto-nas-space/evaluate/models/", 
								type=str, help="What is the directory of the neural architectures?")
parser.add_argument('-gpu_mem', default=800, type=int, help="GPU memory limit")
parser.add_argument('-batch_size', default=512, type=int, help="Batch size to use in training.")
parser.add_argument('-learning_rate', default=0.1, type=float, help="Learning rate for training session.")
parser.add_argument('-epochs', default=200, type=int, help="Number epochs for training session.")
parser.add_argument('-n_classes', default=10, type=int, help="Number of classes in the classification task.")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generation.")
parser.add_argument('-save_weights', action="store_true", help="Save weights of softmax continuous technique.")
parser.add_argument('-save_weights_path', default=str(Path.home())+"/auto-nas-space/evaluate/weights/", type=str, help="Path of directory to save save weights.")
parser.add_argument('-out_file', default="out.txt", type=str, help="Output file")
opt = parser.parse_args()

method = opt.method
model = opt.model
dataset = opt.dataset
networks = opt.networks
na_path = opt.na_path
gpu_mem = opt.gpu_mem
seed = opt.seed
batch_size = opt.batch_size
learning_rate = opt.learning_rate
out_file = opt.out_file
epochs = opt.epochs
n_classes = opt.n_classes
save_weights = opt.save_weights
save_weights_path = opt.save_weights_path
tmp_batches_path="./tmp_batches"

total_start = time.time()

#create batches directory
process_utils.launch_process(batch_utils.create_batches, (dataset,tmp_batches_path,batch_size))

n_batches = int(len(os.listdir(tmp_batches_path))/2)

for epoch in range(epochs):

	for batch in range(n_batches):
		#do not train with last batch, save weights if needed
		if(batch+1==n_batches):
			if(save_weights):
				process_utils.launch_process(batch_prediction_utils.save_weights, 
										(epoch, na_path, method, save_weights_path))
			continue


		epoch_start = time.time()

		flattened_predictions = Manager().Array('d', range(batch_size*n_classes))
		o_predictions = np.zeros((networks+1,batch_size,n_classes), dtype=np.float32)

		for network in range(-1,networks):
			process_utils.launch_process(batch_prediction_utils.batch_prediction, 
										(flattened_predictions, tmp_batches_path, batch, epoch, network, na_path, method, batch_size, learning_rate, gpu_mem, seed))

			o_predictions[network+1]=np.array(flattened_predictions).reshape((batch_size,n_classes))

		#train weights that allow continuous representation of the neural networks
		process_utils.launch_process(batch_prediction_utils.continuous_training, 
									(o_predictions, tmp_batches_path, batch, learning_rate, epoch, na_path, method, gpu_mem, seed))

		print("Took: ", time.time()-epoch_start, " seconds")

	print("Finished epoch ", epoch)

#delete batches and directory
batch_utils.delete_batches(tmp_batches_path)

print("Total execution time: ", time.time()-total_start, " seconds")