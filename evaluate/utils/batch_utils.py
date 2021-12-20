import os

import numpy as np

from utils import import_utils

def create_batches(dataset,batches_path,batch_size):

	tf, cifar10, imagenet, mnist, os, np = import_utils.create_batches()

	with tf.device('/CPU:0'):
		if(dataset=="mnist"):
			x_train, y_train, x_test, y_test = mnist.get()
		elif(dataset=="cifar10"):
			x_train, y_train, x_test, y_test = cifar10.get()
		elif(dataset=="imagenet"):
			x_train, y_train, x_test, y_test = imagenet.get()

		train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

	#save
	os.mkdir(batches_path)

	batch=1
	#write train batches
	for batch_x, batch_y in train_set.repeat(1):
		np.save(batches_path+"/batch_x_"+str(batch), batch_x.numpy())
		np.save(batches_path+"/batch_y_"+str(batch), batch_y.numpy())
		batch+=1

def get_batch(tensorflow, batches_path, batch, dtype):
	batch_x, batch_y = (np.load(batches_path+"/batch_x_"+str(batch)+".npy"),
						np.load(batches_path+"/batch_y_"+str(batch)+".npy"))
	return (tensorflow.convert_to_tensor(batch_x, dtype=dtype), 
			tensorflow.convert_to_tensor(batch_y, dtype=dtype))

def get_test_set(dataset):
	tf, cifar10, imagenet, mnist, os, np = import_utils.create_batches()

	with tf.device('/CPU:0'):
		if(dataset=="mnist"):
			x_train, y_train, x_test, y_test = mnist.get()
		elif(dataset=="cifar10"):
			x_train, y_train, x_test, y_test = cifar10.get()
		elif(dataset=="imagenet"):
			x_train, y_train, x_test, y_test = imagenet.get()

	return tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
	

def num_test_instances(num_instances, dataset):
	tf, cifar10, imagenet, mnist, os, np = import_utils.create_batches()

	with tf.device('/CPU:0'):
		if(dataset=="mnist"):
			x_train, y_train, x_test, y_test = mnist.get()
		elif(dataset=="cifar10"):
			x_train, y_train, x_test, y_test = cifar10.get()
		elif(dataset=="imagenet"):
			x_train, y_train, x_test, y_test = imagenet.get()

	num_instances.value = y_test.shape[0]

def delete_batches(batches_path):
	for file_path in os.listdir(batches_path):
		os.remove(batches_path+"/"+file_path)
	os.rmdir(batches_path)