from smt_conv import *
import argparse
import numpy as np


def adj_k_metric(solutions, adj_k):
	avg_adj_sim = 0.0
	for i in range(len(solutions) - adj_k):
		sim = 0.0

		for k in range(adj_k):
			sim += np.sum(np.logical_xor(solutions[(i) + k], solutions[(i+1) + k]))

		avg_adj_sim += sim/adj_k

	return avg_adj_sim/(len(solutions)-adj_k)


parser = argparse.ArgumentParser()
parser.add_argument('sampling',
                    choices=['all', 'limited', 'limited_uniform'],
                    help="How to enumerate solutions")
parser.add_argument('-I', default="30", type=str, help="Input dimension")
parser.add_argument('-O', default="20", type=str, help="Output dimension")
parser.add_argument('-n', default=2, type=int, help="Lower bound of layers")
parser.add_argument('-N', default=4, type=int, help="Upper bound of layers")
parser.add_argument('-K', default=2, type=int, help="Number of architectures to generate")
parser.add_argument('-adj_k', default=10, type=int, help="Evaluation metric adjacent K")
parser.add_argument('-uniwit_k', default=2, type=float, help="Uni Wit algorithm k argument")
parser.add_argument('-verbose', default=0, type=int, help="0 - no prints, 1 - level prints...")
opt = parser.parse_args()

np.random.seed(42)

sampling = opt.sampling
I = tuple(int(k) for k in opt.I.split(","))
O = tuple(int(k) for k in opt.O.split(","))
n = opt.n
N = opt.N
K = opt.K
adj_k = opt.adj_k
uniwit_k = opt.uniwit_k
verbose = opt.verbose


start_time=time.time()

F = SMT_CONV(I, O, n, N)

if(verbose):
	print("#Variables: ", len(F.get_variables()))
	pi = 1.0
	for i in range(len(I)):
		pi*=(I[i]-O[i])

	#pi=1.0
	estimated = 0.0

	print("Estimated number solutions: ", (pi*N - pi)**(N-1))

	for n_layer in range(n, N+1):
		n_factorial = 1.0
		for i in range(1,n_layer+1):
			n_factorial *= i
		estimated += pi*n_factorial
	print("Estimated number solutions: ", estimated-pi)

if(sampling == 'limited_uniform'):
	uni_wit = UniWit(K)
	solutions = uni_wit.sample(F, k=uniwit_k, binary=True, verbose=True, viz_verbose=True)
elif(sampling == 'limited'):
	solutions = F.solve(K=K, binary=True)
elif(sampling == 'all'):
	solutions = F.solve(binary=True)

binary_solutions = F.get_binary_solutions()
binary_solutions = np.array(binary_solutions)

dist_adj_k = adj_k_metric(binary_solutions, adj_k)
if(verbose):
	print("Average adjacent_k, for k =", adj_k, ", evaluation metric of enumerated solutions: ", dist_adj_k)
dist_adj_k = adj_k_metric(binary_solutions, 3)
if(verbose):
	print("Average adjacent_k, for k =", 3, ", evaluation metric of enumerated solutions: ", dist_adj_k)
dist_adj_k = adj_k_metric(binary_solutions, 5)
if(verbose):
	print("Average adjacent_k, for k =", 5, ", evaluation metric of enumerated solutions: ", dist_adj_k)
dist_adj_k = adj_k_metric(binary_solutions, 10)
if(verbose):
	print("Average adjacent_k, for k =", 10, ", evaluation metric of enumerated solutions: ", dist_adj_k)


if(verbose):
	print("Total solutions: ", len(solutions))
	print("Took ", time.time()-start_time, " seconds")

#F.get_architectures(solutions)