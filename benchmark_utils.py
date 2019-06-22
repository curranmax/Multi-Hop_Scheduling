
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

			splits[split][1][k] = (split_flow, flow.invweight())

	Profiler.end('splitWindow')

	return splits
