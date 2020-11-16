import z3

import tensorflow as tf


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
				net_restrictions = z3.And(net_restrictions, self.kernel[l*self.D+d] - self.stride[l*self.D+d] >= 0)
				net_restrictions = z3.And(net_restrictions, self.kernel[l*self.D+d] > 0)
				net_restrictions = z3.And(net_restrictions, self.stride[l*self.D+d] > 0)

		layer_input_shape = self.input_shape
		next_layer_output_shape = ()

		for l in range(self.u_bound_layers):
			new_layer_input_shape = ()

			layer_restrictions = True

			layer_restrictions = z3.And(layer_restrictions, self.blockers[l])

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

		return net_restrictions#, z3.PbGe([(z3.Bool('blocker_l_%i' % i),1) for i in range(1, self.u_bound_layers+1)], self.l_bound_layers))

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

		self.conv_solver.set("timeout", 60000)

		#while there are solutions to be returned
		while(self.conv_solver.check() == z3.sat):
			self.add_solution()

		return self.possible_combinations

	def build_architecture(self, solution, verbose=False):
		model = tf.keras.Sequential()

		kernels = solution[0]
		strides = solution[1]

		for layer in range(len(kernels)):
			model.add(tf.keras.layers.Conv3D(1, kernels[layer], strides=strides[layer]))
		
		model.build(input_shape=(None, ) + self.input_shape +(1,))

		if(verbose):
			print(model.summary())

		return model

	def get_architectures(self, solutions):
		architectures = []
		
		for solution in solutions:
			architectures.append(self.build_architecture(solution, verbose=True))

		return architectures



if __name__ == "__main__":

	lower_bound_layers = 1
	upper_bound_layers = 2
	solver = SMT_CONV((30,30,30), (20,20,20), lower_bound_layers, upper_bound_layers)
	solutions = solver.solve()
	solver.get_architectures(solutions)
