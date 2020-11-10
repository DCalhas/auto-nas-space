import numpy as np
import z3
import math

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
		self.kernel = {}
		self.stride = {}
		self.conv_solver = {}

		for number_layers in range(self.l_bound_layers, self.u_bound_layers + 1):
			self.setup_variables_layer(number_layers)

	def setup_variables_layer(self, number_layers):
		kernel = []
		stride = []

		for l in range(number_layers):
			for dimension in range(1, self.D+1):
				kernel.append(z3.Int('k_' + str(dimension) + "_l" + str(l) + "_L" + str(number_layers)))
				stride.append(z3.Int('s_' + str(dimension) + "_l" + str(l) + "_L" + str(number_layers)))

		self.kernel[number_layers] = kernel
		self.stride[number_layers] = stride
		self.conv_solver[number_layers] = z3.Solver()

	def setup_restrictions(self):

		for number_layers in range(self.l_bound_layers, self.u_bound_layers+1):
			self.conv_solver[number_layers].add(self.setup_restrictions_layer(number_layers))

	def setup_restrictions_layer(self, number_layers):
		l_restrictions = True
		
		kernel = self.kernel[number_layers]
		stride = self.stride[number_layers]

		layer_input_shape = self.input_shape
		next_layer_output_shape = ()

		for l in range(number_layers):
			new_layer_input_shape = ()
			for d in range(self.D):
				l_restrictions = z3.And(l_restrictions, kernel[l*self.D+d] - stride[l*self.D+d] >= 0)
				l_restrictions = z3.And(l_restrictions, kernel[l*self.D+d] > 0)
				l_restrictions = z3.And(l_restrictions, stride[l*self.D+d] > 0)
				if(l < number_layers-1):
					#new_layer_input_shape += (1+(layer_input_shape[d] - kernel[l*self.D+d])/ stride[l*self.D+d],)
					out_l = 1+(layer_input_shape[d] - kernel[l*self.D+d])/stride[l*self.D+d]
					out_next_l = 1+(out_l - kernel[(l+1)*self.D+d])/stride[(l+1)*self.D+d]
					in_next_l = (out_next_l-1)*stride[(l+1)*self.D+d] + kernel[(l+1)*self.D+d]
					l_restrictions = z3.And(l_restrictions, out_l == in_next_l)
					new_layer_input_shape += (in_next_l,)
					if(l == number_layers-2):
						next_layer_output_shape += (out_next_l,)
				else:
					if(number_layers > 1):
						l_restrictions = z3.And(l_restrictions, self.output_shape[d] == next_layer_output_shape[d])
					else:
						l_restrictions = z3.And(l_restrictions, (self.output_shape[d] - 1)*stride[l*self.D+d] + kernel[l*self.D+d] == layer_input_shape[d])
			
			layer_input_shape = new_layer_input_shape

		return l_restrictions

	def get_solution(self, number_layers):
		solution = self.conv_solver[number_layers].model()

		kernels = []
		strides = []
		for l in range(number_layers):
			l_kernel = ()
			l_stride = ()
			for dimension in range(self.D):

				l_kernel += (solution[self.kernel[number_layers][l*self.D+dimension]].as_long(),)
				l_stride += (solution[self.stride[number_layers][l*self.D+dimension]].as_long(),)

			kernels.append(l_kernel)
			strides.append(l_stride)

		return kernels, strides

	def restrict_solution(self, solution, D, number_layers):

		for layer in range(number_layers):
			kernel = solution[0][layer][:D]
			stride = solution[1][layer][:D]

			if(D == 1):
				return self.kernel[number_layers][layer*self.D+D-1] != kernel[D-1], self.stride[number_layers][layer*self.D+D-1] != stride[D-1]
			
			self.conv_solver[number_layers].add(z3.Or(*self.restrict_solution(solution, D-1, number_layers)))
			return self.kernel[number_layers][layer*self.D+D-1] != kernel[D-1], self.stride[number_layers][layer*self.D+D-1] != stride[D-1]

	def add_solution(self, number_layers):

		solution = self.get_solution(number_layers)

		self.possible_combinations.append(solution)
		
		#add recursively through dimensions the solution
		self.conv_solver[number_layers].add(z3.Or(*self.restrict_solution(solution, self.D, number_layers)))


	def solve(self):
		self.setup_restrictions()

		for number_layers in range(self.l_bound_layers, self.u_bound_layers+1):
			self.conv_solver[number_layers].set("timeout", 60000)

			#while there are solutions to be returned
			while(self.conv_solver[number_layers].check() == z3.sat):
				self.add_solution(number_layers)

		return self.possible_combinations

	def build_architecture(self, solution, verbose=False):
		model = tf.keras.Sequential()

		kernels = solution[0]
		strides = solution[1]

		print(len(kernels))
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
	upper_bound_layers = 10
	solver = SMT_CONV((30,30,30), (20,20,20), lower_bound_layers, upper_bound_layers)
	solutions = solver.solve()
	solver.get_architectures(solutions)
