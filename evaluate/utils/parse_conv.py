import tensorflow as tf



def cnn_to_tuple(architecture):
	kernels = ()
	strides = ()

	for layer in architecture.layers:

		kernels += (layer.kernel_size,)
		strides += (layer.strides,)

	return kernels, strides

