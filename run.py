
import algos
import input_utils
from profiler import Profiler

import argparse
import random

INPUT_SOURCES = ['test', 'microsoft', 'sigmetrics']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Creates a multi-hop schedule for a dynamic network')

	parser.add_argument('-nn', '--num_nodes', metavar = 'NODES', type = int, nargs = 1, default = [1], help = 'Number of nodes in the network')
	parser.add_argument('-rl', '--max_route_length', metavar = 'MAX_ROUTE_LENGTH', type = int, nargs = 1, default = [1], help = 'Maximum route length allowed')
	parser.add_argument('-ws', '--window_size', metavar = 'WINDOW_SIZE', type = int, nargs = 1, default = [1], help = 'Total time to create the schedule for')
	parser.add_argument('-rd', '--reconfig_delta', metavar = 'RECONFIG_DELTA', type = int, nargs = 1, default = [1], help = 'Time it takes to reconfigure the network')
	parser.add_argument('-is', '--input_source', metavar = 'INPUT_SOURCE', type = str, nargs = 1, default = ['test'], help = 'Source to generate the Flows. Must be one of (' + ', '.join(INPUT_SOURCES) + ')')

	parser.add_argument('-profile', '--profile_code', action = 'store_true', help = 'If given, then the code is profiled, and results are outputted at the end')

	args = parser.parse_args()

	# Get command line args
	num_nodes        = args.num_nodes[0]
	max_route_length = args.max_route_length[0]
	window_size      = args.window_size[0]
	reconfig_delta   = args.reconfig_delta[0]
	input_source     = args.input_source[0]

	if input_source not in INPUT_SOURCES:
		raise Exception('Invalid input_source: ' + str(input_source))

	if args.profile_code:
		Profiler.turnOn()

	# Get input
	# Random Test data
	if input_source == 'test':
		flows = input_utils.generateTestFlows(num_nodes, max_route_length)
		for key in flows:
			print(flows[key])
		print('number of flows', len(flows))

	# Data based on real measurements
	if input_source in ['microsoft', 'sigmetrics']:
		traffic = input_utils.Traffic(num_nodes = num_nodes, max_hop = max_route_length, window_size = window_size, random_seed = 0)

	if input_source == 'microsoft':
		flows = traffic.microsoft(1)  # cluster 1 is somewhat dense
		for k in flows:
			print(flows[k])
		print(traffic)

		# flows = traffic.microsoft(2)  # cluster 2 is between 1 and 3
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

		# flows = traffic.microsoft(3)  # cluster 3 is sparse (many zeros)
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

	if input_source == 'sigmetrics':
		flows = traffic.sigmetrics(c_l = 0.7, n_l = 4500, c_s = 0.3, n_s = 4500)
		for k in flows:
			print(flows[k])
		print(traffic)

	# flows = traffic.facebook( ... )
	# flows = traffic.university( ... )

	# Run test
	try:
		# pass
		schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta)
		
		print result_metric
	except KeyboardInterrupt:
		pass

	# Output result
	Profiler.stats()
