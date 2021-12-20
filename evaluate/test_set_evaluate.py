import sys

sys.path.append("..")

import argparse

import os

import h5py

import numpy as np

from pathlib import Path

from multiprocessing import Process, shared_memory, Manager

import math

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
parser.add_argument('-n_classes', default=10, type=int, help="Number of classes in the classification task.")
parser.add_argument('-save', action="store_true", help="Save results to a file.")
opt = parser.parse_args()

method = opt.method
model = opt.model
dataset = opt.dataset
networks = opt.networks
na_path = opt.na_path
gpu_mem = opt.gpu_mem
n_classes = opt.n_classes
save = opt.save
save_file_npy = method+"_"+model+"_"+dataset+"_results.npy"
save_file_txt = method+"_"+model+"_"+dataset+"_results.txt"

if(save):
	save_file_txt = open(save_file_txt, "w")

acc =  Manager().Value('d', 0)
acc_std =  Manager().Value('d', 0)
loss =  Manager().Value('d', 0)
loss_std =  Manager().Value('d', 0)
results = np.zeros((networks+1,4), dtype=np.float32)

for network in range(-1,networks):
	process_utils.launch_process(batch_prediction_utils.evaluate, 
								(loss, loss_std, acc, acc_std, dataset, network, na_path, method, gpu_mem))

	results[network+1,0] = loss.value
	results[network+1,1] = loss_std.value
	results[network+1,2] = acc.value
	results[network+1,3] = acc_std.value
	if(save):
		print("network: ", network+1, file=save_file_txt)
		print("loss", file=save_file_txt)
		print(results[network+1,0], file=save_file_txt)
		print("loss_std", file=save_file_txt)
		print(results[network+1,1], file=save_file_txt)
		print("acc", file=save_file_txt)
		print(results[network+1,2], file=save_file_txt)
		print("acc_std", file=save_file_txt)
		print(results[network+1,3], file=save_file_txt)
	else:
		print("network: ", network+1)
		print("loss")
		print(results[network+1,0])
		print("loss_std")
		print(results[network+1,1])
		print("acc")
		print(results[network+1,2])
		print("acc_std")
		print(results[network+1,3])

np.save(save_file_npy, results)

if(save):
	save_file_txt.close()