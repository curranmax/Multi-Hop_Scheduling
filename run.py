
import algos
import input_utils

import random

if __name__ == '__main__':
	# random.seed(10)

	# Get command line args
	num_nodes = 10
	max_route_length = 3
	window_size = 1000
	reconfig_delta = 10

	# Get input
	input_source = 'test'
	if input_source == 'test':
		flows = input_utils.generateTestFlows(num_nodes, max_route_length)
	if input_source == 'microsoft':
		traffic = input_utils.Traffic(num_nodes = num_nodes, max_hop = max_route_length, random_seed = 0)
	
		flows = traffic.microsoft(1)  # cluster 1 is somewhat dense
		for k in flows:
			print(flows[k])

		flows = traffic.microsoft(2)  # cluster 2 is between 1 and 3
		for k in flows:
			print(flows[k])

		flows = traffic.microsoft(3)  # cluster 3 is sparse
		for k in flows:
			print(flows[k])

	# flows = traffic.sigmetrics( ... )
	# flows = traffic.microsort( ... )
	# flows = traffic.university( ... )

	# Run test
	algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta)

	# Output result
