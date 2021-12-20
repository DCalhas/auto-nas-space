import tensorflow as tf

def get():
	train, test = tf.keras.datasets.mnist.load_data()

	# normalize inputs
	x_train = (train[0]/255)
	x_test = (test[0]/255)

	x_train = x_train.reshape(x_train.shape+(1,))
	x_test = x_test.reshape(x_test.shape+(1,))

	y_train = tf.keras.utils.to_categorical(train[1],num_classes=10)
	y_test = tf.keras.utils.to_categorical(test[1],num_classes=10)

	return x_train, y_train, x_test, y_test