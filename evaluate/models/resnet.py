import tensorflow as tf

from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training



layers = None
L2_WEIGHT_DECAY = 0.0001

def ResNet(stack_fn,
			preact,
			use_bias,
			model_name='resnet',
			include_top=True,
			weights='imagenet',
			input_tensor=None,
			input_shape=None,
			pooling=None,
			classes=1000,
			classifier_activation='linear',
			conv_kernel=None,
			conv_strides=None,
			**kwargs):
	
	
	global layers
	if ('layers' in kwargs):
		layers = kwargs.pop('layers')
	else:
		layers = VersionAwareLayers()

	
	if input_tensor is None:
		img_input = layers.Input(shape=input_shape)
	else:
		if not backend.is_keras_tensor(input_tensor):
			img_input = layers.Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

	x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
	x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, 
						kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY), name='conv1_conv')(x)

	if not preact:
		x = layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
		x = layers.Activation('relu', name='conv1_relu')(x)

	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
	x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

	x = stack_fn(x, conv_kernel=conv_kernel, conv_strides=conv_strides)

	if preact:
		x = layers.BatchNormalization(
			axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
		x = layers.Activation('relu', name='post_relu')(x)

	if include_top:
		x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
		x = layers.Dense(classes, activation=classifier_activation, 
						kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
						name='predictions')(x)
	else:
		if pooling == 'avg':
			x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
		elif pooling == 'max':
			x = layers.GlobalMaxPooling2D(name='max_pool')(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = layer_utils.get_source_inputs(input_tensor)
	else:
		inputs = img_input

	# Create model.
	model = tf.keras.Model(inputs, x, name=model_name)

	return model



def block1(x, filters, kernel_size=3, kernel=1, stride=1, conv_shortcut=True, name=None):
	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
	if conv_shortcut:
		shortcut = layers.Conv2D(
			4 * filters, kernel, strides=stride, 
						kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						name=name + '_0_conv')(x)
		shortcut = layers.BatchNormalization(
						axis=bn_axis, epsilon=1.001e-5, 
   						name=name + '_0_bn')(shortcut)
	else:	
		shortcut = x

	x = layers.Conv2D(filters, kernel, strides=stride, padding='valid', 
						kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						name=name + '_1_conv')(x)
	x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
	x = layers.Activation('relu', name=name + '_1_relu')(x)
	
	#downsampling layer
	x = layers.Conv2D(4*filters, kernel_size, padding='same', 
						kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
   						name=name + '_2_conv')(x)
	x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

	x = layers.Add(name=name + '_add')([shortcut, x])
	x = layers.Activation('relu', name=name + '_out')(x)
	return x


def stack1(x, filters, blocks, kernel1=(1,1), stride1=(2,2), name=None):
	"""A set of stacked residual blocks.
	Args:
	x: input tensor.
	filters: integer, filters of the bottleneck layer in a block.
	blocks: integer, blocks in the stacked blocks.
	stride1: default 2, stride of the first layer in the first block.
	name: string, stack label.
	Returns:
	Output tensor for the stacked blocks.
	"""
	x = block1(x, filters, kernel=kernel1, stride=stride1, name=name + '_block1')
	for i in range(2, blocks + 1):
		x = block1(x, filters, name=name + '_block' + str(i))
	return x



def ResNet50(include_top=True,
			weights='imagenet',
			input_tensor=None,
			input_shape=None,
			conv_kernel=None,
			conv_strides=None,
			mode="original",
			pooling=None,
			classes=1000,
			**kwargs):

	def stack_fn_original(x, conv_kernel=None, conv_strides=None):
		#define kernel1 and stride1 for downsampling layer
		x = stack1(x, 64, 2, kernel1=1, stride1=1, name='conv2')
		x = stack1(x, 128, 2, name='conv3')
		x = stack1(x, 256, 2, name='conv4')
		return stack1(x, 512, 2, name='conv5')

	def stack_fn_generated(x, conv_kernel=None, conv_strides=None):
		#define kernel1 and stride1 for downsampling layer
		x = stack1(x, 64, 2, kernel1=1, stride1=1, name='conv2')
		x = stack1(x, 128, 2, kernel1=conv_kernel[0], stride1=conv_strides[0], name='conv3')
		x = stack1(x, 256, 2, kernel1=conv_kernel[1], stride1=conv_strides[1], name='conv4')
		return stack1(x, 512, 2, kernel1=conv_kernel[2], stride1=conv_strides[2], name='conv5')

	if(mode=="original"):
		return ResNet(stack_fn_original, False, True, 'resnet50', include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
	elif(mode=="generated"):
		return ResNet(stack_fn_generated, False, True, 'resnet50', include_top, weights, 
					input_tensor, input_shape, pooling, classes, 
					conv_kernel=conv_kernel, conv_strides=conv_strides, **kwargs)