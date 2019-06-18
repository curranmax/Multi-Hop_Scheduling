
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
	input_source = 'test'

	# Get input
	# Random Test data
	if input_source == 'test':
		flows = input_utils.generateTestFlows(num_nodes, max_route_length)

	# Data based on real measurements
	if input_source in ['microsoft', 'sigmetrics']:
		traffic = input_utils.Traffic(num_nodes=num_nodes, max_hop=max_route_length, window_size=window_size, random_seed=0)

	if input_source == 'microsoft':
		flows = traffic.microsoft(1)  # cluster 1 is somewhat dense
		for k in flows:
			print(flows[k])
		print(traffic)

		flows = traffic.microsoft(2)  # cluster 2 is between 1 and 3
		for k in flows:
			print(flows[k])
		print(traffic)

		flows = traffic.microsoft(3)  # cluster 3 is sparse (many zeros)
		for k in flows:
			print(flows[k])
		print(traffic)

	if input_source == 'sigmetrics':
		flows = traffic.sigmetrics(c_l=0.7, n_l=4, c_s=0.3, n_s=12)
		for k in flows:
			print(flows[k])
		print(traffic)

	# flows = traffic.facebook( ... )
	# flows = traffic.university( ... )

	# Run test
	schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta)

	print result_metric

	# Output result
