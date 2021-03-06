
import algos
from time import time
import benchmark_utils
import input_utils
from profiler import Profiler

import argparse
import random

INPUT_SOURCES = ['test', 'microsoft', 'sigmetrics', 'facebook']
METHODS = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse', 'octopus-b', 'octopus+', 'octopus-e', 'projector', 'rotornet', 'octopus-greedy']


def boolFromStr(val):
	return val in ['True', 'true', 't', 'T']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Creates a multi-hop schedule for a dynamic network')

	parser.add_argument('-nn', '--num_nodes', metavar = 'NODES', type = int, nargs = 1, default = [1], help = 'Number of nodes in the network')
	parser.add_argument('-rl', '--max_route_length', metavar = 'MAX_ROUTE_LENGTH', type = int, nargs = 1, default = [1], help = 'Maximum route length allowed')
	parser.add_argument('-ws', '--window_size', metavar = 'WINDOW_SIZE', type = int, nargs = 1, default = [1], help = 'Total time to create the schedule for')
	parser.add_argument('-rd', '--reconfig_delta', metavar = 'RECONFIG_DELTA', type = int, nargs = 1, default = [1], help = 'Time it takes to reconfigure the network')
	parser.add_argument('-nr', '--num_routes', metavar = 'NUM_ROUTES', type = int, nargs = 1, default = [1], help = 'Number of routes to generate')
	parser.add_argument('-is', '--input_source', metavar = 'INPUT_SOURCE', type = str, nargs = 1, default = ['test'], help = 'Source to generate the Flows. Must be one of (' + ', '.join(INPUT_SOURCES) + ')')

	parser.add_argument('-min_rl', '--min_route_length', metavar = 'MIN_ROUTE_LENGTH', type = int, nargs = 1, default = [1], help = 'Minimum route length possible')
	
	parser.add_argument('-eps', '--use_eps', metavar = 'TRUE_FALSE', type = boolFromStr, nargs = 1, default = [False], help = 'Whether to use epsilon trick')

	parser.add_argument('-m', '--methods', metavar = 'METHODS', type = str, nargs = '+', default = ['octopus-r'], help = 'Set of methods to run. Must be in the set of (' + ', '.join(METHODS) + ') or "all"')

	parser.add_argument('-profile', '--profile_code', action = 'store_true', help = 'If given, then the code is profiled, and results are outputted at the end')
	parser.add_argument('-v', '--verbose', action = 'store_true', help = 'If given, then outputs intermediate updates')
	parser.add_argument('-runner', '--runner_output', action = 'store_true', help = 'If given, outputs results appropriate for runner.py')
	
	parser.add_argument('-nl', '--num_large',  type=int, nargs = 1,   default=[4],   help='For the sigmetric data only. Number of large flows. Indicates sparsity')
	parser.add_argument('-ns', '--num_small',  type=int, nargs = 1,   default=[12],  help='For the sigmetric data only. Number of small flows. Indicates sparsity')
	parser.add_argument('-cl', '--capa_large', type=float, nargs = 1, default=[0.7], help='For the sigmetric data only. Capacity of large flows. Indicates skewness')
	parser.add_argument('-cs', '--capa_small', type=float, nargs = 1, default=[0.3], help='For the sigmetric data only. Capacity of small flows. Indicates skewness')
	parser.add_argument('-clus', '--cluster',  type=int,   nargs = 1, default=[1],   help='For Microsoft and Facebook only. Cluster number')
	
	args = parser.parse_args()

	# Get command line args
	num_nodes        = args.num_nodes[0]
	max_route_length = args.max_route_length[0]
	window_size      = args.window_size[0]
	reconfig_delta   = args.reconfig_delta[0]
	num_routes       = args.num_routes[0]
	input_source     = args.input_source[0]
	num_large        = args.num_large[0]
	num_small        = args.num_small[0]
	capa_large       = args.capa_large[0]
	capa_small       = args.capa_small[0]
	cluster          = args.cluster[0]

	use_eps = args.use_eps[0]
	algos.setUseEps(use_eps)

	min_route_length = args.min_route_length[0]

	methods = args.methods

	if 'all' in methods:
		methods = METHODS

	if input_source not in INPUT_SOURCES:
		raise Exception('Invalid input_source: ' + str(input_source))
	
	profile_code  = args.profile_code
	verbose       = args.verbose
	runner_output = args.runner_output

	if runner_output:
		# Overrides profile and verbose flags
		profile_code = False
		verbose      = False

		# Outputs the other input params to stdout
		print 'num_nodes|'        + str(num_nodes)
		print 'min_route_length|' + str(min_route_length)
		print 'max_route_length|' + str(max_route_length)
		print 'window_size|'      + str(window_size)
		print 'reconfig_delta|'   + str(reconfig_delta)
		print 'num_routes|'       + str(num_routes)
		print 'use_eps|'          + str(use_eps)
		print 'input_source|'     + str(input_source)
		print 'nl|'               + str(num_large)
		print 'ns|'               + str(num_small)
		print 'cl|'               + str(capa_large)
		print 'cs|'               + str(capa_small)
		print 'cluster|'          + str(cluster)

		print 'methods|' + ','.join(methods)

	if profile_code:
		Profiler.turnOn()

	# Get input
	# Random Test data
	if input_source == 'test':
		flows = input_utils.generateTestFlows(num_nodes, max_route_length, num_routes, flow_prob = 1.0)

	# Data based on real measurements
	if input_source in ['microsoft', 'sigmetrics', 'facebook']:
		random_seed = int((time() - int(time())) * 1000000)
		traffic = input_utils.Traffic(num_nodes = num_nodes, max_hop = max_route_length, window_size = window_size, num_routes = num_routes, min_route_length = min_route_length, random_seed = random_seed)

	if input_source == 'sigmetrics':
		args = {'c_l': capa_large, 'n_l': num_large, 'c_s': capa_small, 'n_s': num_small}

		flows = traffic.sigmetrics(**args)
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

	if input_source == 'microsoft':
		flows = traffic.microsoft(cluster, multiply=1)  # cluster 1 is somewhat dense
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)
	
	if input_source == 'facebook':
		mydic = {1:'A', 2:'B', 3:'C'}
		flows = traffic.facebook(cluster = mydic[cluster], multiply = 1)
		# for k in flows:
		# 	print(flows[k])
		# print(traffic)

	compute_upper_bound = False
	if compute_upper_bound:
		total_packets = sum(flow.size for _, flow in flows.iteritems()) / float(num_nodes)
		total_packets_by_route_length = {1: total_packets / 3.0, 2: total_packets / 3.0, 3: total_packets / 3.0}

		print 'Total packets:', total_packets
		window_size_unused = window_size
		upper_bound = 0.0
		for rl, packets in sorted(total_packets_by_route_length.iteritems()):
			if rl * packets >= window_size_unused:
				print 'From rl', rl, '-->', float(window_size_unused) / float(rl)
				upper_bound += float(window_size_unused) / float(rl)
				break
			else:
				print 'a'
				print 'From rl', rl, '-->', float(packets) / float(rl)
				upper_bound += float(packets) / float(rl)
				window_size_unused -= packets * rl

		print 'Upper bound:', upper_bound / window_size * 100.0

		quit()

	# Run test
	results = {}
	for method in methods:
		if method not in METHODS:
			print 'Invalid method:', method
			continue

		if verbose:
			print 'Running method:', method

		if method == 'octopus-r':
			# Runs the main "vanilla" method
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, verbose = verbose)

		if method == 'octopus-greedy':
			# Runs the main "vanilla" method and using the greedy maximum weight matching
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, verbose = verbose, greedy=True)

		if method == 'octopus-s':
			# Runs the "vanilla" method, but uses the shortest route instead of a random route
			shortest_route_flows = benchmark_utils.useShortestRouteFlow(flows)

			schedule, result_metric = algos.computeSchedule(num_nodes, shortest_route_flows, window_size, reconfig_delta, verbose = verbose)

		if method == 'upper-bound':
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, upper_bound = True, verbose = verbose)

		if method == 'split':
			# Splits the window and flows into max_route_length sections
			number_of_splits = max_route_length
			splits = benchmark_utils.splitWindow(window_size, number_of_splits, reconfig_delta, flows)

			# Used to check if all hops of a flow are completed
			flow_by_id                   = {flow.id: flow for _, flow in flows.iteritems()}
			packets_delivered_by_flow_id = {flow.id: None for _, flow in flows.iteritems()}

			# Holds the schedules and result_metrics from each split
			schedules = []
			result_metrics = []
			for this_window_size, this_flows in splits:

				# Runs the algorithm for one split
				schedule, result_metric, this_packets_delivered_by_flow_id = algos.computeSchedule(num_nodes, this_flows, this_window_size, reconfig_delta, return_packets_delivered_by_flow_id = True, flows_have_given_invweight = True, verbose = verbose)

				schedules.append(schedule)
				result_metrics.append(result_metric)

				# Tracks which flows has been completed
				for k, packets_delivered in this_packets_delivered_by_flow_id.iteritems():
					if packets_delivered_by_flow_id[k] is None or packets_delivered < packets_delivered_by_flow_id[k]:
						packets_delivered_by_flow_id[k] = packets_delivered

			# Combined the schedules into one big one
			total_schedule = algos.combineSchedules(schedules)
			if total_schedule.totalDuration(reconfig_delta) != window_size:
				raise Exception('Split total schedule doesn\'t match window size')

			# Checks if all hops of a flow are delivered
			total_packets_delivered = 0
			total_packets_not_delivered = 0
			for fid, flow in flow_by_id.iteritems():
				packets_delivered_for_this_flow = packets_delivered_by_flow_id[fid]

				total_packets_delivered     += packets_delivered_for_this_flow
				total_packets_not_delivered += flow.size - packets_delivered_for_this_flow

				if packets_delivered_for_this_flow is None:
					raise Exception('Flow unaccounted for')

			total_packets = sum(flow.size for _, flow in flows.iteritems())

			if total_packets != total_packets_delivered + total_packets_not_delivered:
				raise Exception('Not all packets accounted for: total_packets = ' + str(total_packets) + ', packets_delivered = ' + str(total_packets_delivered) + ', packets_not_delivered = ' + str(total_packets_not_delivered))

			# Calculates the time slots used and not used. Includes the reconfiguration delta between splits.
			total_time_slots_used = sum(rm.time_slots_used for rm in result_metrics)
			total_time_slots_not_used = sum(rm.time_slots_not_used for rm in result_metrics) + (num_nodes * reconfig_delta * (number_of_splits - 1))

			# Gets the total result metric
			total_result_metric = algos.ResultMetric(total_schedule.getTotalMatchingWeight(), total_packets_delivered, total_packets_not_delivered, total_schedule.totalDuration(reconfig_delta), total_time_slots_used, total_time_slots_not_used)
			
			# Renames the variables to fit the convention of the loop
			schedule = total_schedule
			result_metric = total_result_metric

		if method == 'eclipse':
			# Runs "eclipse" method. First computes schedule using "upper-bound" method, then routes upper-bound.
			single_hop_flows = benchmark_utils.reduceToOneHop(flows)

			schedule, _ = algos.computeSchedule(num_nodes, single_hop_flows, window_size, reconfig_delta, verbose = verbose)
			
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, precomputed_schedule = schedule, verbose = verbose)

		if method == 'octopus+':
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, consider_all_routes = True, backtrack = False, verbose = verbose)

		if method == 'octopus-e':
			orig_eps = algos.setUseEps(True)
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, verbose = verbose)
			algos.setUseEps(orig_eps)

		if method == 'octopus-b':
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, alpha_search_method = 'search', verbose = verbose)

		if method == 'projector':
			fixed_alpha = 10.0 * reconfig_delta
			alternate_matching = 'random_stable-ignore_weights'
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, fixed_alpha = fixed_alpha, use_alternate_matching = alternate_matching, global_override_weight = 1.0, verbose = verbose)

		if method == 'rotornet':
			fixed_alpha = 10.0 * reconfig_delta
			alternate_matching = 'rotornet'
			schedule, result_metric = algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta, fixed_alpha = fixed_alpha, use_alternate_matching = alternate_matching, global_override_weight = 1.0, verbose = verbose)


		results[method] = (schedule, result_metric)

	# Output result

	if runner_output:
		print 'OUTPUT'
		for method in METHODS:
			if method in results:
				print 'method|' + method

				_, result_metric = results[method]

				result_metric.runnerOutput()

	if verbose:
		for method in METHODS:
			if method in results:
				print 'Method:', method, '--->',results[method][1]

	Profiler.stats()


# python run.py -nn 100 -min_rl 1 -rl 3 -ws 10000 -rd 500 -nr 10 -eps False -is sigmetrics -nl 4 -ns 12 -cl 0.7 -cs 0.3 -clus 1 -m octopus-r -runner