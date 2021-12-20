

def setup_state():
	import pickle

	return pickle

def create_batches():
	import tensorflow as tf

	from data import cifar10, imagenet, mnist

	import os

	import numpy as np

	return tf, cifar10, imagenet, mnist, os, np


def continuous_training():
	import tensorflow as tf

	import numpy as np

	import softmax

	from data import cifar10, imagenet, mnist

	from utils import parse_conv, tf_config, train, print_utils, state_utils

	return tf, np, softmax, cifar10, imagenet, mnist, parse_conv, tf_config, train, print_utils, state_utils

def batch_prediction():

	import tensorflow as tf

	from data import cifar10, imagenet, mnist

	from models import resnet

	from utils import parse_conv, tf_config, train, print_utils, state_utils

	return tf, cifar10, imagenet, mnist, resnet, parse_conv, tf_config, train, print_utils, state_utils