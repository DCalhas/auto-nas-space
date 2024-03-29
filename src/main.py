from smt_conv import *

import argparse

import numpy as np
import math

from pathlib import Path


def adj_k_metric(solutions, adj_k):
	avg_adj_sim = 0.0

	for i in range(len(solutions)-1):

		avg_adj_sim += np.sum(np.logical_xor(solutions[(i)], solutions[(i+1)]))

	return avg_adj_sim/(len(solutions))


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
parser.add_argument('-previous_i', default=0, type=int, help="Uni Wit iteration to start at. One should have experience history.")
parser.add_argument('-n_threads', default=1, type=int, help="Number threads to run consecutively")
parser.add_argument('-maxpool', action="store_true", help="Insert maxpool layers after each conv layer")
parser.add_argument('-maxpool_k', default="2", type=str, help="Maxpool kernel size")
parser.add_argument('-maxpool_s', default="1", type=str,help="Maxpool stride size")
parser.add_argument('-save', action="store_true", help="Option to save architectures")
parser.add_argument('-save_path', default=str(Path.home())+"/auto-nas-space/evaluate/models/", type=str, help="Input dimension")
parser.add_argument('-verbose', default=0, type=int, help="0 - no prints, 1 - level prints...")
parser.add_argument('-seed', default=42, type=int, help="Seed for random generator")
opt = parser.parse_args()



sampling = opt.sampling
I = tuple(int(k) for k in opt.I.split(","))
O = tuple(int(k) for k in opt.O.split(","))
n = opt.n
N = opt.N
K = opt.K
adj_k = opt.adj_k
uniwit_k = opt.uniwit_k
previous_i = opt.previous_i
maxpool = opt.maxpool
maxpool_k = tuple(int(k) for k in opt.maxpool_k.split(","))
maxpool_s = tuple(int(k) for k in opt.maxpool_s.split(","))
n_threads = opt.n_threads
verbose = opt.verbose
save = opt.save
save_path = opt.save_path
seed = opt.seed

np.random.seed(seed)

start_time=time.time()


if(sampling == 'limited_uniform'):
	save_path += "uni_wit/"
	F = SMT_CONV(I, O, n, N, opt_maxpool=maxpool, opt_maxpool_k=maxpool_k, opt_maxpool_s=maxpool_s)
	uni_wit = UniWit(K)
	solutions = uni_wit.sample(F, k=uniwit_k, previous_i=previous_i, binary=True, verbose=True, viz_verbose=True)
elif(sampling == 'limited'):
	save_path += sampling + "/"
	F = SMT_CONV(I, O, n, N, opt_maxpool=maxpool, opt_maxpool_k=maxpool_k, opt_maxpool_s=maxpool_s)
	solutions = F.solve(K=K, binary=True)
elif(sampling == 'all'):
	F = SMT_CONV(I, O, n, N, opt_maxpool=maxpool, opt_maxpool_k=maxpool_k, opt_maxpool_s=maxpool_s)
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

if(save):
	F.get_architectures(solutions, path=save_path)
