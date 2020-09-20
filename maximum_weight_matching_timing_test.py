
from profiler import Profiler

import networkx as nx
import numpy as np
import random
import scipy.optimize as scipy_opt

MAX_WEIGHT_MATCHING_LIBRARYS = ['networkx', 'scipy']

def generateRandomGraph(num_nodes, min_weight, max_weight, edge_probability, int_weights_only, max_weight_matching_library):
	# Generate a random bipartite graph based on the parameters
	graph = np.zeros((num_nodes, num_nodes))

	for i in range(num_nodes):
		for j in range(num_nodes):
			# Skips self edges
			if i == j:
				continue

			if int_weights_only:
				w = random.randint(min_weight, max_weight)
			else:
				w = random.uniform(min_weight, max_weight)

			if random.random() <= edge_probability:
				graph[i][j] = w

	if max_weight_matching_library == 'scipy':
		return graph
	if max_weight_matching_library == 'networkx':
		raise Exception('Not implemented')

if __name__ == '__main__':
	# Initial parameters
	num_nodes = 250
	min_weight = 1
	max_weight = 10
	edge_probability = 0.0
	int_weights_only = True
	max_weight_matching_library = 'scipy'
	num_iters = 10

	if max_weight_matching_library not in MAX_WEIGHT_MATCHING_LIBRARYS:
		raise Exception('Invalid max weight matching library: ' + max_weight_matching_library)

	Profiler.turnOn()

	Profiler.start('total_test')
	for i in range(num_iters):
		print 'Running test', i + 1, 'of', num_iters
		# Generate random graph
		Profiler.start('generateRandomGraph')
		random_weighted_graph = generateRandomGraph(num_nodes, min_weight, max_weight, edge_probability, int_weights_only, max_weight_matching_library)
		Profiler.end('generateRandomGraph')

		if max_weight_matching_library == 'scipy':
			Profiler.start('scipy.optimize.linear_sum_assignment')
			matching = scipy_opt.linear_sum_assignment(-random_weighted_graph)
			Profiler.end('scipy.optimize.linear_sum_assignment')

		if max_weight_matching_library == 'networkx':
			raise Exception('Not implemented')

	Profiler.end('total_test')

	Profiler.stats()
