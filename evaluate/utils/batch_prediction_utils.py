from utils import import_utils, batch_utils, process_utils


def evaluate(loss, loss_std, acc, acc_std, dataset, network, na_path, method, gpu_mem):
	tf, cifar10, imagenet, mnist, resnet, parse_conv, tf_config, train, print_utils, state_utils = import_utils.batch_prediction()

	tf_config.setup_tensorflow(memory_limit=gpu_mem)

	test_set = batch_utils.get_test_set(dataset)

	nn_model = tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1) + "_training", compile=True)

	l, l_std, a, a_std = train.evaluate(test_set, nn_model, tf.keras.losses.CategoricalCrossentropy(from_logits=True))

	loss.value = l
	loss_std.value = l_std
	acc.value = a
	acc_std.value = a_std

def batch_prediction(o_predictions, batches_path, batch, epoch, network, na_path, method, batch_size, learning_rate, gpu_mem, seed):
	tf, cifar10, imagenet, mnist, resnet, parse_conv, tf_config, train, print_utils, state_utils = import_utils.batch_prediction()

	tf_config.setup_tensorflow(memory_limit=gpu_mem)
	tf.random.set_seed(seed)

	batch_x, batch_y = batch_utils.get_batch(tf, batches_path, batch+1, tf.float32)

	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	if(network==-1):
		if(batch == 0 and epoch == 0):
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			nn_model = resnet.ResNet50(include_top=True,
				weights=None, input_tensor=None,
				input_shape=batch_x.shape[1:], pooling=None, classes=batch_y.shape[1])
			nn_model.compile(optimizer=optimizer)
		else:
			#optimizer = state_utils.setup_state(tf, optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
			nn_model = tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1) + "_training", compile=True)
			state_utils.setup_state(tf, nn_model.optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
												na_path + method + "/architecture_" + str(network+1) + "_training/gen_config")
	else:
		if(batch == 0 and epoch == 0):
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			kernels, strides = parse_conv.cnn_to_tuple(tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1), compile=False))
			nn_model = resnet.ResNet50(include_top=True,  mode="generated",
					weights=None, input_tensor=None, input_shape=batch_x.shape[1:],
					conv_kernel=kernels, conv_strides=strides, pooling=None, classes=batch_y.shape[1])
			nn_model.compile(optimizer=optimizer)
		else:
			#optimizer = state_utils.setup_state(tf, optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
			nn_model = tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1) + "_training", compile=True)
			state_utils.setup_state(tf, nn_model.optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
												na_path + method + "/architecture_" + str(network+1) + "_training/gen_config")

	loss, batch_preds = train.train_step(nn_model, batch_x, batch_y, nn_model.optimizer, loss_fn, return_logits=True)
	loss=loss.numpy()

	flattened_batch_preds=batch_preds.numpy().flatten()
	for i in range(flattened_batch_preds.shape[0]):
		o_predictions[i] = flattened_batch_preds[i]

	#save model
	nn_model.save(na_path + method + "/architecture_" + str(network+1) + "_training", save_format="tf")
	#save state
	state_utils.save_state(tf, nn_model.optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
							na_path + method + "/architecture_" + str(network+1) + "_training/gen_config")
	
def continuous_training(o_predictions, batches_path, batch, learning_rate, epoch, na_path, method, gpu_mem, seed):
	tf, np, softmax, cifar10, imagenet, mnist, parse_conv, tf_config, train, print_utils, state_utils = import_utils.continuous_training()
	tf_config.setup_tensorflow(memory_limit=gpu_mem)
	tf.random.set_seed(seed)

	loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	if(batch==0 and epoch == 0):
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		model=softmax.Softmax((o_predictions.shape[0],))
		model.build(input_shape=o_predictions.shape)
		model.compile(optimizer=opt)
	else:
		model=tf.keras.models.load_model(na_path + method + "/softmax_training", compile=True)
		#opt = state_utils.setup_state(tf, opt, na_path + method + "/softmax_training/opt_config", 
		state_utils.setup_state(tf, model.optimizer, na_path + method + "/softmax_training/opt_config", 
							na_path + method + "/softmax_training/gen_config")
	

	#read batch batch
	_, batch_y = batch_utils.get_batch(tf, batches_path, batch+1, dtype=tf.float32)

	#training step to update weights with batch
	loss = train.train_step(model, o_predictions, batch_y, model.optimizer, loss_fn).numpy()
	
	print("Softmax epoch loss: ", loss)
	print(model.trainable_variables[0].numpy())

	model.save(na_path + method + "/softmax_training", save_format="tf")
	#save state
	state_utils.save_state(tf, model.optimizer, na_path + method + "/softmax_training/opt_config", 
							na_path + method + "/softmax_training/gen_config")


def save_weights(epoch, na_path, method, save_weights_path): 
	tf, np, softmax, cifar10, imagenet, mnist, parse_conv, tf_config, train, print_utils, state_utils = import_utils.continuous_training()

	model=tf.keras.models.load_model(na_path + method + "/softmax_training", compile=False)

	np.save(save_weights_path+"epoch_"+str(epoch), model.trainable_variables[0].numpy())
