
import algos
import benchmark_utils
import input_utils
from profiler import Profiler

import argparse
import random

INPUT_SOURCES = ['test', 'microsoft', 'sigmetrics', 'facebook']
METHODS = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse']

# python run.py -nn 100 -rl 4 -ws 100 -rd 1 -is sigmetrics -profile

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Creates a multi-hop schedule for a dynamic network')

	parser.add_argument('-nn', '--num_nodes', metavar = 'NODES', type = int, nargs = 1, default = [1], help = 'Number of nodes in the network')
	parser.add_argument('-rl', '--max_route_length', metavar = 'MAX_ROUTE_LENGTH', type = int, nargs = 1, default = [1], help = 'Maximum route length allowed')
	parser.add_argument('-ws', '--window_size', metavar = 'WINDOW_SIZE', type = int, nargs = 1, default = [1], help = 'Total time to create the schedule for')
	parser.add_argument('-rd', '--reconfig_delta', metavar = 'RECONFIG_DELTA', type = int, nargs = 1, default = [1], help = 'Time it takes to reconfigure the network')
	parser.add_argument('-nr', '--num_routes', metavar = 'NUM_ROUTES', type = int, nargs = 1, default = [1], help = 'Number of routes to generate')
	parser.add_argument('-is', '--input_source', metavar = 'INPUT_SOURCE', type = str, nargs = 1, default = ['test'], help = 'Source to generate the Flows. Must be one of (' + ', '.join(INPUT_SOURCES) + ')')

	parser.add_argument('-m', '--methods', metavar = 'METHODS', type = str, nargs = '+', default = ['octopus-r'], help = 'Set of methods to run. Must be in the set of (' + ', '.join(METHODS) + ') or "all"')

	parser.add_argument('-profile', '--profile_code', action = 'store_true', help = 'If given, then the code is profiled, and results are outputted at the end')

	args = parser.parse_args()

	# Get command line args
	num_nodes        = args.num_nodes[0]
	max_route_length = args.max_route_length[0]
	window_size      = args.window_size[0]
	reconfig_delta   = args.reconfig_delta[0]
	num_routes       = args.num_routes[0]
	input_source     = args.input_source[0]

	methods = args.methods

	if 'all' in methods:
		methods = METHODS

	if input_source not in INPUT_SOURCES:
		raise Exception('Invalid input_source: ' + str(input_source))

	if args.profile_code:
		Profiler.turnOn()

	# Get input
	# Random Test data
	if input_source == 'test':
		flows = input_utils.generateTestFlows(num_nodes, max_route_length, num_routes, flow_prob = 1.0)
		# for key in flows:
		# 	print(flows[key])
		print('number of flows', len(flows))

	# Data based on real measurements
	if input_source in ['microsoft', 'sigmetrics', 'facebook']:
		traffic = input_utils.Traffic(num_nodes = num_nodes, max_hop = max_route_length, window_size = window_size, num_routes = num_routes, random_seed = 0)

	if input_source == 'microsoft':
		flows = traffic.microsoft(1)  # cluster 1 is somewhat dense
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

		# flows = traffic.microsoft(2)  # cluster 2 is between 1 and 3
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

		# flows = traffic.microsoft(3)  # cluster 3 is sparse (many zeros)
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

	if input_source == 'sigmetrics':
		args = None
		if num_nodes in [64, 100]:
			args = {'c_l': 0.7, 'n_l': 4, 'c_s': 0.3, 'n_s': 12}

		if num_nodes == 200:
			args = {'c_l': 0.7, 'n_l': 8, 'c_s': 0.3, 'n_s': 24}

		if num_nodes == 300:
			args = {'c_l': 0.7, 'n_l': 12, 'c_s': 0.3, 'n_s': 36}

		if num_nodes == 400:
			args = {'c_l': 0.7, 'n_l': 16, 'c_s': 0.3, 'n_s': 48}

		if num_nodes == 500:
			args = {'c_l': 0.7, 'n_l': 20, 'c_s': 0.3, 'n_s': 60}

		if args is None:
			raise Exception('Invalid number of nodes used with "sigmetrics" data')

		# TODO adjust these parameters if # of nodes increases to 200, 300, 400, and 500
		flows = traffic.sigmetrics(**args)
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)
	
	if input_source == 'facebook':
		flows = traffic.facebook(cluster='A')
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

	# Run test
	try:
		results = {}
		for method in methods:
			if method not in METHODS:
				print 'Invalid method:', method
				continue

			if method == 'octopus-r':
				# Runs the main "vannilla" method
				schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta)
				print method, result_metric

			if method == 'octopus-s':
				shortest_route_flows = benchmark_utils.useShortestRouteFlow(flows)

				schedule, result_metric = algos.computeSchedule(num_nodes, single_hop_flows, window_size, reconfig_delta)
				print method, result_metric

			if method == 'upper-bound':
				# Runs the upper-bound where all flows only have a single hop
				single_hop_flows = benchmark_utils.reduceToOneHop(flows)

				schedule, result_metric = algos.computeSchedule(num_nodes, single_hop_flows, window_size, reconfig_delta)
				print method, result_metric

			if method == 'split':
				# Splits the window and flows into max_route_length sections
				number_of_splits = max_route_length
				splits = benchmark_utils.splitWindow(window_size, number_of_splits, reconfig_delta, flows)

				# Used to check if all hops of a flow are completed
				flow_by_id = {flow.id: flow for _, flow in flows.iteritems()}
				split_flows_completed = {flow.id: 0 for _, flow in flows.iteritems()}

				# Holds the schedules and result_metrics from each split
				schedules = []
				result_metrics = []
				x = 1
				for this_window_size, this_flows in splits:
					print 'Running split', x, 'with window_size of', this_window_size
					x += 1

					# Runs the algorithm for one split
					schedule, result_metric, finished_flow_ids = algos.computeSchedule(num_nodes, this_flows, this_window_size, reconfig_delta, return_completed_flow_ids = True)

					schedules.append(schedule)
					result_metrics.append(result_metric)

					# Tracks which flows has been completed
					for k in finished_flow_ids:
						split_flows_completed[k] += 1

				# Combined the schedules into one big one
				total_schedule = algos.combineSchedules(schedules)
				if total_schedule.totalDuration(reconfig_delta) != window_size:
					print total_schedule.totalDuration(reconfig_delta), window_size
					raise Exception('Split total schedule doesn\'t match window size')

				# Checks if all hops of a flow are delivered
				total_packets_delivered = 0
				total_packets_not_delivered = 0
				for fid, flow in flow_by_id.iteritems():
					if split_flows_completed[fid] == len(flow.route) - 1:
						# If all splits of a flow have completed, then the overall flow was delivered
						total_packets_delivered += flow.size
					else:
						# If any split of a flow was not completed, then the overall flow was not delivered
						total_packets_not_delivered += flow.size

				# Calculates the time slots used and not used. Includes the reconfiguration delta between splits.
				total_time_slots_used = sum(rm.time_slots_used for rm in result_metrics)
				total_time_slots_not_used = sum(rm.time_slots_not_used for rm in result_metrics) + (num_nodes * reconfig_delta * (number_of_splits - 1))

				# Gets the total result metric
				total_result_metric = algos.ResultMetric(total_schedule.getTotalMatchingWeight(), total_packets_delivered, total_packets_not_delivered, total_schedule.totalDuration(reconfig_delta), total_time_slots_used, total_time_slots_not_used)
				
				# Renames the variables to fit the convention of the loop
				schedule = total_schedule
				result_metric = total_result_metric

			if method == 'eclipse':
				single_hop_flows = benchmark_utils.reduceToOneHop(flows)

				schedule, _ = algos.computeSchedule(num_nodes, single_hop_flows, window_size, reconfig_delta)
				
				schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, precomputed_schedule = schedule)

				print result_metric

			results[method] = (schedule, result_metric)
	except KeyboardInterrupt:
		pass

	# Output result
	Profiler.stats()
