import z3

import tensorflow as tf

import time

import matplotlib.pyplot as plt


"""
SMT_CONV

SMT formula implementation using z3-solver

gives the possible combinations of kernels and strides for a convolution from a dimension to another

Variables are kernels and strides, represented as:

	k_d_l_LN
	s_d_l_LN
	where d, l and N are integers representing dimension, layer and total layers, respectively
"""
class SMT_CONV:

	"""
		Inputs:

			* input_shape - tuple representing the input dimension
			* output_shape - tuple representing the output dimension
			* l_bound_layers - integer representing the lower bound for the number of layers
			* u_bound_layers - integer representing the upper bound for the number of layers
	"""
	def __init__(self, input_shape, output_shape, l_bound_layers, u_bound_layers):
		assert type(input_shape) is tuple
		assert type(output_shape) is tuple
		assert len(input_shape) == len(output_shape)


		self.D = len(input_shape)
		self.conv_dim_layer = "Conv" + str(self.D) + "D"
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.l_bound_layers = l_bound_layers
		self.u_bound_layers = u_bound_layers
		self.setup_variables()

		self.possible_combinations = []

	def setup_variables(self):
		self.blockers = [z3.Bool("blocker_l_{0}".format(i)) for i in range(1,self.u_bound_layers+1)]
		self.kernel = []
		self.stride = []
		self.conv_solver = z3.Solver()
		self.setup_variables_layer()

	def setup_variables_layer(self):
		for l in range(self.u_bound_layers):
			for dimension in range(1, self.D+1):
				self.kernel.append(z3.Int('k_' + str(dimension) + "_l" + str(l+1)))
				self.stride.append(z3.Int('s_' + str(dimension) + "_l" + str(l+1)))

	def setup_restrictions(self):

		self.conv_solver.add(self.setup_restrictions_layer(self.u_bound_layers))

	def setup_restrictions_layer(self, number_layers):
		net_restrictions = True

		for l in range(self.u_bound_layers-1):
			net_restrictions = z3.And(net_restrictions, z3.Implies(z3.Not(self.blockers[l]), z3.Not(self.blockers[l+1])))

		for l in range(1, self.u_bound_layers):
			net_restrictions = z3.And(net_restrictions, z3.Implies(self.blockers[l], self.blockers[l-1]))

		net_restrictions = z3.And(net_restrictions, self.blockers[self.l_bound_layers-1])

		for l in range(self.u_bound_layers):

			for d in range(self.D):
				net_restrictions = z3.And(net_restrictions, self.kernel[l*self.D+d] - self.stride[l*self.D+d] > 0)
				net_restrictions = z3.And(net_restrictions, self.kernel[l*self.D+d] > 0)
				net_restrictions = z3.And(net_restrictions, self.stride[l*self.D+d] > 0)

		layer_input_shape = self.input_shape
		next_layer_output_shape = ()

		for l in range(self.u_bound_layers):
			new_layer_input_shape = ()

			layer_restrictions = True

			hidden_layer = True
			output_layer = True

			for d in range(self.D):

				if(l < self.u_bound_layers-1):
					#hidden layer
					out_l = 1+(layer_input_shape[d] - self.kernel[l*self.D+d])/self.stride[l*self.D+d]
					out_next_l = 1+(out_l - self.kernel[(l+1)*self.D+d])/self.stride[(l+1)*self.D+d]
					in_next_l = (out_next_l-1)*self.stride[(l+1)*self.D+d] + self.kernel[(l+1)*self.D+d]
					hidden_layer = z3.And(hidden_layer, out_l == in_next_l)

					output_layer = z3.And(output_layer, (self.output_shape[d] - 1)*self.stride[l*self.D+d] + self.kernel[l*self.D+d] == layer_input_shape[d])

					hidden_layer = z3.And(hidden_layer, layer_restrictions)

					new_layer_input_shape += (in_next_l,)
					if(l == number_layers-2):
						next_layer_output_shape += (out_next_l,)
				else:
					output_layer = z3.And(output_layer, (self.output_shape[d] - 1)*self.stride[l*self.D+d] + self.kernel[l*self.D+d] == layer_input_shape[d])

				output_layer = z3.And(output_layer, layer_restrictions)

			if(l < self.u_bound_layers -1):
				hidden_layer = z3.Or(hidden_layer, z3.Or(z3.Not(self.blockers[l]), z3.Not(self.blockers[l+1])))
				output_layer = z3.Or(output_layer, z3.Or(z3.Not(self.blockers[l]), self.blockers[l+1]))
				net_restrictions = z3.And(net_restrictions, z3.And(hidden_layer, output_layer))
			else:
				net_restrictions = z3.And(net_restrictions, z3.Or(output_layer, z3.Not(self.blockers[l])))

			layer_input_shape = new_layer_input_shape

		return net_restrictions

	def get_solution(self):
		solution = self.conv_solver.model()

		#get total layers
		total_layers = 0
		for l in self.blockers:
			total_layers += z3.is_true(solution[l])
		
		kernels = []
		strides = []
		for l in range(total_layers):
			l_kernel = ()
			l_stride = ()
			for dimension in range(self.D):

				l_kernel += (solution[self.kernel[l*self.D+dimension]].as_long(),)
				l_stride += (solution[self.stride[l*self.D+dimension]].as_long(),)

			kernels.append(l_kernel)
			strides.append(l_stride)

		return kernels, strides, solution

	def restrict_solution(self, solution):
		restriction = False
		total_layers = 0

		for l in range(self.u_bound_layers):
			total_layers += int(z3.is_true(solution[self.blockers[l]]))
			if(z3.is_false(solution[self.blockers[l]])):
				continue
			for d in range(self.D):
				restriction = z3.Or(restriction, self.kernel[l*self.D+d] != solution[self.kernel[l*self.D+d]].as_long())
				restriction = z3.Or(restriction, self.stride[l*self.D+d] != solution[self.stride[l*self.D+d]].as_long())

		if(total_layers < self.u_bound_layers):
			restriction = z3.Or(restriction, z3.Or(z3.Not(self.blockers[total_layers-1]), self.blockers[total_layers]))
		else:
			restriction = z3.Or(restriction, z3.Not(self.blockers[total_layers-1]))

		return restriction

	def add_solution(self):

		kernels, strides, solution = self.get_solution()

		self.possible_combinations.append((kernels, strides))

		self.conv_solver.add(self.restrict_solution(solution))

	def solve(self):
		self.setup_restrictions()

		self.conv_solver.set("timeout", 600000000000)

		#while there are solutions to be returned
		while(self.conv_solver.check() == z3.sat):
			self.add_solution()

		return self.possible_combinations

	def build_architecture(self, solution, verbose=False):
		model = tf.keras.Sequential()

		kernels = solution[0]
		strides = solution[1]

		for layer in range(len(kernels)):
			model.add(getattr(tf.keras.layers,self.conv_dim_layer)(1, kernels[layer], strides=strides[layer]))
		
		model.build(input_shape=(None, ) + self.input_shape +(1,))

		if(verbose):
			print(model.summary())

		return model

	def get_architectures(self, solutions):
		architectures = []
		
		for solution in solutions:
			architectures.append(self.build_architecture(solution, verbose=True))

		return architectures



#number dimensions and number solutions for each diff
diff_analysis = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9], 2: [1, 4, 9, 16, 25, 36, 49, 64, 81], 3: [1, 8, 27, 64, 125, 216, 343, 512, 729]}
solving_times = {1: [0.0045108795166015625,0.00709223747253418,0.004609107971191406,0.003916501998901367,0.005946159362792969,0.004069805145263672,0.004523754119873047,0.0041735172271728516,0.004436969757080078], 
				2: [0.01716756820678711,0.01297760009765625,0.014728546142578125,0.017618894577026367,0.019097089767456055,0.026305437088012695,0.024593830108642578,0.02793407440185547,0.03563833236694336],
				3: [0.015392303466796875,0.023629188537597656,0.09439229965209961,0.04838752746582031,0.15618085861206055,0.19048595428466797,0.19025492668151855,0.1710188388824463,0.26610422134399414], 
				4: [0.04995918273925781,0.3442513942718506,0.293210506439209,0.31862640380859375,2.116061210632324,0.7530810832977295,1.1577751636505127,1.2944204807281494,1.887657642364502],
				5:[0.37785959243774414,0.37280821800231934,0.5188636779785156,0.761563062667846,2.226407289505005,1.48002290725708,3.152721881866455,3.6443755626678467,16.281274795532227],
				6: [0.2137293815612793,0.24453377723693848,1.5915162563323975,0.9848437309265137,24.00694489479065,27.443228721618652,41.5923912525177,19.983529090881348,57.34502840042114]}

def plot_solving_times():
	plt.figure()
	plt.plot(list(range(1,10)), solving_times[1], label="1 diff", linestyle="-")
	plt.plot(list(range(1,10)), solving_times[2], label="2 diff", linestyle="-")
	plt.plot(list(range(1,10)), solving_times[3], label="3 diff", linestyle="-")
	plt.plot(list(range(1,10)), solving_times[4], label="4 diff", linestyle="-")
	plt.plot(list(range(1,10)), solving_times[5], label="5 diff", linestyle="-")
	plt.plot(list(range(1,10)), solving_times[6], label="6 diff", linestyle="-")
	plt.legend()
	plt.title("#Dimensions Layers (Log Scale)")
	plt.xlabel("#Dimensions")
	plt.ylabel("Solving time (s)")
	plt.grid(True)
	plt.yscale("log")
	plt.savefig("solving_time.pdf", format="pdf")


if __name__ == "__main__":
	start_time=time.time()
	lower_bound_layers = 4
	upper_bound_layers = 4
	solver = SMT_CONV((30,), (20,), lower_bound_layers, upper_bound_layers)
	solutions = solver.solve()

	print("Total solutions: ", len(solutions))
	print("Took ", time.time()-start_time, " seconds")

	solver.get_architectures(solutions)


	

