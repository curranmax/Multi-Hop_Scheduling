
import algos
import input_utils
from profiler import Profiler

from collections import defaultdict
import math
import random

# For each flow in flows, sets flow.route to the shortest route in flows.all_routes
# Input:
#   flows --> dictionary of flows (indexed by (src node, dst node)) to be reduced
# Output:
#   single_hop_flows --> Essentially a deep copy of flows, but all flow.route are set the the shortest route in flow.all_routes
def useShortestRouteFlow(flows):
	Profiler.start('useShortestRouteFlow')
	shortest_route_flow = {}

	for k, flow in flows.iteritems():
		shortest_route = None
		for route in flow.all_routes:
			if shortest_route is None or len(shortest_route) > len(route):
				shortest_route = route

		shortest_route_flow[k] = input_utils.Flow(flow.id, flow.src, flow.dst, flow.size, shortest_route, flow.all_routes)

	Profiler.end('useShortestRouteFlow')

	return shortest_route_flow


# Creates a copies of flows, but all routes only have one hop
# Input:
#   flows --> dictionary of flows (indexed by (src node, dst node)) to be reduced
# Output:
#   single_hop_flows --> Essentially a deep copy of flows, but all routes are reduced to one hop.
def reduceToOneHop(flows):
	Profiler.start('reduceToOneHop')
	single_hop_flows = {}

	for k, flow in flows.iteritems():
		single_hop_flows[k] = input_utils.Flow(flow.id, flow.src, flow.dst, flow.size, [flow.src, flow.dst], flow.all_routes)

	Profiler.end('reduceToOneHop')

	return single_hop_flows

# Splits the given time window, and assigns each hop of a flow to a split
# Input:
#   window_size --> The total window size to split
#   num_of_splits --> The number of ways to divide the window
#   reconfig_delta --> The reconfiguraiton detla, there must be a reconfiguration between each split
#   flows --> Overall dictionary of flows
# Output:
#   splits --> list of 2-tuple (window_size, flows)
def splitWindow(window_size, number_of_splits, reconfig_delta, flows):
	Profiler.start('splitWindow')

	window_split_floor = int(math.floor(float(window_size - (reconfig_delta * (number_of_splits - 1))) / float(number_of_splits)))

	windows = [window_split_floor] * number_of_splits
	extra = window_size - (sum(windows) + reconfig_delta * (number_of_splits - 1))

	if extra >= len(windows):
		raise Exception('Unexpected values when splitting')

	for i in range(extra):
		windows[i] += 1

	if sum(windows) + reconfig_delta * (number_of_splits - 1) != window_size:
		raise Exception('Unexpected values when splitting')

	splits = [(w, {}) for w in windows]
	for k, flow in flows.iteritems():
		assigned_slots = sorted(random.sample(range(number_of_splits), len(flow.route) - 1))

		for i, split in enumerate(assigned_slots):
			this_src = flow.route[i]
			this_dst = flow.route[i + 1]
			this_route = [this_src, this_dst]
			split_flow = input_utils.Flow(flow.id, this_src, this_dst, flow.size, this_route, [this_route])

			splits[split][1][k] = split_flow

	Profiler.end('splitWindow')

	return splits

# Routes multi-hop flows through precomputed schedule. (Mostly a copy of algos.computeSchedule)
# Input:
#   num_nodes --> Number of nodes in the network
#   schedule --> Precomputed schedule. Must be a algos.Schedule instance
#   flows --> dictionary of flows keyed by (src_node, dst_node)
#   window_size --> The total time to route the flows over
#   reconfig_delta --> Time it takes to change matchings
# Output:
#   result_metric --> Traffic metrics of how flows were routed through 
def routeFlowsThroughSchedule(num_nodes, schedule, flows, window_size, reconfig_delta):
	Profiler.start('routeFlowsThroughSchedule')

	# Initialzie subflows (same as the remaining traffic)
	current_subflows = [algos.SubFlow(flow = flows[k]) for k in flows]
	completed_subflows = []

	# Track time slots where packets are and aren't sent
	time_slots_used = 0
	time_slots_not_used = 0

	for ind, (matching, alpha) in enumerate(zip(schedule.matchings, schedule.durations)):
		Profiler.start('routeFlowsThroughSchedule-iteration')

		# Gets the subflows_by_next_hop information from remaining_flows
		# Keys are (curNode(), nextNode()) and values are a list of NextHopFlows
		Profiler.start('routeFlowsThroughSchedule-subflows_by_next_hop')
		subflows_by_next_hop = defaultdict(list)
		for subflow in current_subflows:
			subflows_by_next_hop[(subflow.curNode(), subflow.nextNode())].append(subflow)
		Profiler.end('routeFlowsThroughSchedule-subflows_by_next_hop')

		# Sorts each list in subflows_by_next_hop
		for k in subflows_by_next_hop:
			algos.sortSubFlows(subflows_by_next_hop[k])

		# Calculate matching's weight for multi-hop routes
		this_matching_weight = sum(algos.calculateTotalWeight(subflows_by_next_hop[(i, j)], alpha) for i, j in matching)
		schedule.matching_weights[ind] = this_matching_weight

		# Update remaining
		for edge in matching:
			packets_sent, unused_alpha, finished_subflows, new_subflows = algos.updateSubFlows(subflows_by_next_hop[edge], alpha)
			
			time_slots_used += packets_sent
			time_slots_not_used += unused_alpha

			# Removes any subflows that have finished from current_subflows and places adds them to completed_subflows
			for fin_subflow in finished_subflows:
				current_subflows.remove(fin_subflow)
				completed_subflows.append(fin_subflow)

			# Adds any sublfows that were split to current_subflows
			for new_subflow in new_subflows:
				current_subflows.append(new_subflow)

		Profiler.end('routeFlowsThroughSchedule-iteration')

	# Compute result metrics
	total_objective_value = schedule.getTotalMatchingWeight()

	packets_delivered = sum(subflow.getSize() for subflow in completed_subflows)
	packets_not_delivered = sum(subflow.getSize() for subflow in current_subflows)

	# Add in reconfiguration delay to unused time slots
	time_slots_not_used += reconfig_delta * num_nodes * schedule.numReconfigs()

	result_metric = algos.ResultMetric(total_objective_value, packets_delivered, packets_not_delivered, window_size, time_slots_used, time_slots_not_used)

	Profiler.end('routeFlowsThroughSchedule')
	return result_metric
