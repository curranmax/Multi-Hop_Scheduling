
from profiler import Profiler

from collections import defaultdict
import copy
import networkx as nx
import numpy as np
import scipy.optimize as scipy_opt
import time
import random

# MAX_WEIGHT_MATCHING_LIBRARY = 'networkx'
MAX_WEIGHT_MATCHING_LIBRARY = 'scipy'

EPS = 0.0
def setUseEps(use_eps):
	global EPS
	rv = (EPS > 0.00001)

	if use_eps:
		EPS = 0.01
	else:
		EPS = 0.0

	return rv

# Holds a multi-hop schedule
class Schedule:
	def __init__(self):
		self.matchings = []
		self.durations = []

		self.matching_weights = []

	def addMatching(self, matching, duration, matching_weight):
		self.matchings.append(matching)
		self.durations.append(duration)

		self.matching_weights.append(matching_weight)

	def numMatchings(self):
		return len(self.matchings)

	def numReconfigs(self):
		if len(self.matchings) <= 0:
			return 0
		else:
			return len(self.matchings) - 1

	def totalDuration(self, reconfig_delta):
		return sum(self.durations) + reconfig_delta * self.numReconfigs()

	def getTotalMatchingWeight(self):
		return sum(self.matching_weights)

# Combines all schedules into one
def combineSchedules(schedules):
	total_schedule = Schedule()

	for schedule in schedules:
		for m, d, w in zip(schedule.matchings, schedule.durations, schedule.matching_weights):
			total_schedule.addMatching(m, d, w)

	return total_schedule

# Holds the different metrics for the result of one run
class ResultMetric:
	def __init__(self, total_objective_value, packets_delivered, packets_not_delivered, total_duration, time_slots_used, time_slots_not_used, packets_delivered_by_tag = None, computation_duration = 0.0):
		self.total_objective_value = total_objective_value
		self.packets_delivered     = packets_delivered
		self.packets_not_delivered = packets_not_delivered
		self.total_duration        = total_duration
		self.time_slots_used       = time_slots_used
		self.time_slots_not_used   = time_slots_not_used

		# Time it took to compute this result. Value in seconds
		self.computation_duration = computation_duration

		if packets_delivered_by_tag is not None and len(packets_delivered_by_tag) > 0:
			self.packets_delivered_by_tag = packets_delivered_by_tag
		else:
			self.packets_delivered_by_tag = None

	def getLinkUtilization(self):
		return float(self.time_slots_used) / float(self.time_slots_used + self.time_slots_not_used)

	def runnerOutput(self):
		print 'total_objective_value|' + str(self.total_objective_value)
		print 'packets_delivered|'     + str(self.packets_delivered)
		print 'packets_not_delivered|' + str(self.packets_not_delivered)
		print 'time_slots_used|'       + str(self.time_slots_used)
		print 'time_slots_not_used|'   + str(self.time_slots_not_used)

		print 'computation_duration|' + str(self.computation_duration)

		if self.packets_delivered_by_tag is not None and len(self.packets_delivered_by_tag) > 0:
			print 'packets_by_tag|' + ','.join(map(lambda x: str(x[0]) + '=' + str(x[1]) , self.packets_delivered_by_tag.iteritems()))

	def __str__(self):
		normalize = lambda v: float(v) / float(self.total_duration)

		return '{Total Objective Value: ' + str(self.total_objective_value) + ', ' + \
				'Normalized Objective Value: ' + str(normalize(self.total_objective_value)) + ', ' + \
				'Packets Delivered: ' + str(self.packets_delivered) + ', ' + \
				'Normalized Packets Delivered: ' + str(normalize(self.packets_delivered)) + ', ' + \
				'Packets Not Delivered: ' + str(self.packets_not_delivered) + ', ' + \
				'Normalized Packets Not Delivered: ' + str(normalize(self.packets_not_delivered)) + ', ' + \
				('' if self.packets_delivered_by_tag is None else 'Packets delivered by tag: ' + str(self.packets_delivered_by_tag) + ', ') + \
				'Time Slots Used: ' + str(self.time_slots_used) + ', ' + \
				'Time Slots Unused: ' + str(self.time_slots_not_used) + ', ' + \
				'Link Utilization: ' + str(self.getLinkUtilization() * 100.0) + ' %, ' + \
				'Computation Duration: ' + str(self.computation_duration) + ' seconds}'

# Holds a subflow of a input_utils.Flow
class SubFlow:
	next_subflow_id = 0

	# Input must a input_utils.flow
	def __init__(self, flow = None, subflow = None, override_route = None, override_subflow_id = None, override_size = None, override_weight = None, override_invweight = None, tag = None):
		if flow is not None:
			# Reference to the parent flow
			self.flow = flow

			# ID of this subflow
			if override_subflow_id is None:
				self.subflow_id = self.flow.id
			else:
				self.subflow_id = override_subflow_id

			if self.subflow_id >= SubFlow.next_subflow_id:
				SubFlow.next_subflow_id = self.subflow_id + 1

			# Remaining route of the subflow (starting at curNode() and ending at self.flow.dst)
			if override_route is None:
				self.rem_route = list(self.flow.route)
				self.route     = list(self.flow.route)
			else:
				self.rem_route = list(override_route)
				self.route     = list(override_route)

			self.invweight = len(self.rem_route) - 1
			self.weight    = 1.0 / float(self.invweight)

			# This subflow's size.
			if override_size is None:
				self.size = self.flow.size
			else:
				self.size = override_size

			if override_weight is not None and override_invweight is not None:
				raise Exception('Cannot supply both override_weight and override_invweight')

			if override_weight is not None:
				self.weight    = override_weight
				self.invweight = 1.0 / self.weight

			if override_invweight is not None:
				self.invweight = override_invweight
				self.weight    = 1.0 / float(self.invweight)

			self.tag = tag

			self.is_backtrack = False
			self.parent_subflow = None
		elif subflow is not None:
			self.flow = subflow.flow

			if override_subflow_id is None:
				self.subflow_id = SubFlow.next_subflow_id
				SubFlow.next_subflow_id += 1
			else:
				self.subflow_id = override_subflow_id
				if self.subflow_id >= SubFlow.next_subflow_id:
					SubFlow.next_subflow_id = self.subflow_id + 1

			if override_route is None:
				self.rem_route = list(subflow.rem_route)
				self.route     = list(subflow.route)
			else:
				self.rem_route = list(override_route)
				self.route     = list(override_route)

			if override_size is None:
				self.size = self.flow.size
			else:
				self.size = override_size

			self.invweight = subflow.invweight
			self.weight    = subflow.weight

			if override_weight is not None and override_invweight is not None:
				raise Exception('Cannot supply both override_weight and override_invweight')

			if override_weight is not None:
				self.weight    = override_weight
				self.invweight = 1.0 / self.weight

			if override_invweight is not None:
				self.invweight = override_invweight
				self.weight    = 1.0 / float(self.invweight)
			
			if tag is not None:
				self.tag = tag
			elif subflow.tag is not None:
				self.tag = subflow.tag
			else:
				self.tag = None

			self.is_backtrack = subflow.is_backtrack
			if subflow.is_backtrack:
				self.parent_subflow = subflow
			else:
				self.parent_subflow = None
		else:
			raise Exception('Must specify either a flow or subflow to create a subflow')

	def curNode(self):
		if len(self.rem_route) < 1:
			raise Exception('Subflow doesn\'t have a current node')

		return self.rem_route[0]

	def nextNode(self):
		if len(self.rem_route) < 2:
			raise Exception('Subflow doesn\'t have a next node')

		return self.rem_route[1]

	def srcNode(self):
		if len(self.route) < 1:
			raise Exception('Subflow does\'t have a src node')

		return self.route[0]

	def destNode(self):
		if len(self.route) < 2:
			raise Exception('Subflow doesn\'t have a dest node')

		return self.route[-1]

	def getWeight(self, eps = 0.0):
		rem, tot = self.remainingRouteLength(), self.totalRouteLength()

		if (rem == 1 and tot == 1) or (rem == 2 and tot == 3):
			add = 0.0

		elif (rem == 1 and tot == 2) or (rem == 1 and tot == 3):
			add = eps

		elif (rem == 2 and tot == 2) or (rem == 3 and tot == 3):
			add = -eps
		else:
			raise Exception('Do not know what eps to use')

		return self.weight + add

	def getInvweight(self):
		return self.invweight

	def flowID(self):
		return self.flow.id

	def subflowID(self):
		return self.subflow_id

	def getSize(self):
		return self.size

	def remainingRouteLength(self):
		return len(self.rem_route) - 1

	def totalRouteLength(self):
		return len(self.route) - 1

	# Sends as much of this subflow as possible in time alpha.
	# Input:
	#   alpha --> Time available for the subflow to send.
	# Output:
	#   packets_sent --> Number of packets sent.
	#   remaining_alpha --> The time available for toher subflows after this subflow has been sent.
	#   split_subflow --> If the subflow becomes split, this is the portion that is not sent. If the subflow isn't split this is None.
	#   is_subflow_done --> Is true iff the subflow reaches its destination.
	# Addiitionaly updates self.rem_route, and (if the subflow is split) self.size.
	# The first three values are updated to reflect that the subflow has been sent along the first edge in self.rem_route.
	def send(self, alpha):
		Profiler.start('SubFlow.send')
		if alpha <= 0:
			raise Exception('Trying to send a subflow with no time')

		# If the subflow cannot be sent within alpha, the subflow is split
		if self.getSize() > alpha:
			split_subflow = SubFlow(subflow = self)
			split_subflow.size = self.getSize() - alpha

			self.size = alpha
		else:
			split_subflow = None

		# Advances this subflow to the next node
		self.rem_route = self.rem_route[1:]

		# Checks if this subflow has reached its destination
		is_subflow_done = (len(self.rem_route) <= 1)
		
		# Calculates the remaining time that is still unused after sending this subflow
		remaining_alpha = alpha - self.getSize()

		# Gets the number of packets sent
		packets_sent = self.getSize()

		Profiler.end('SubFlow.send')
		return packets_sent, remaining_alpha, split_subflow, is_subflow_done

# Sorts subflows in place
# Input:
#   subflows
def sortSubFlows(subflows):
	# TODO sort by subflow id as well incase there are two subflows of the same size from the same flow

	# Sorts flows by shortest route to longest route. Uses flow.id as a tiebreaker in order to keep things deterministic.
	Profiler.start('sortSubFlows')
	if EPS > 0.00000000001:
		subflows.sort(key = lambda x: (x.getInvweight(), x.remainingRouteLength(), x.flowID()))
	else:
		subflows.sort(key = lambda x: (x.getInvweight(), x.flowID()))
	Profiler.end('sortSubFlows')

# Given the next hop traffic, find the set of alphas to consider
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
# Output:
#   Returns a set of alphas to be considered.
def getUniqueAlphas(subflows_by_next_hop):
	Profiler.start('getUniqueAlphas')

	alphas = set()

	for _, subflows in subflows_by_next_hop.iteritems():
		# Ignores length zero lists
		if len(subflows) == 0:
			continue
		
		this_alpha = 0

		# Loops over all subflows
		prev_invweight = subflows[0].getInvweight()
		for subflow in subflows:
			# Once we have passed all subflows of a certain invweight, add the number of packets from subflows with invweight less than or equal to the previous subflow's invweight as a possible alpha.
			if prev_invweight != subflow.getInvweight():
				alphas.add(this_alpha)

				prev_invweight = subflow.getInvweight()

			# Keep track of total number of packets.
			this_alpha += subflow.getSize()

		# Once all subflows have been processed, adds the total packet size as a possible alpha.
		alphas.add(this_alpha)

	Profiler.end('getUniqueAlphas')
	return alphas

# Truncates alphas that would go over the window_size
# Input:
#   alphas --> Set of alphas to consider
#   total_duration_so_far --> Duration that has been scheduled so far
#   num_matchings --> The current of matchings added so far
#   window_size --> The maximum time that can be used
#   reconfig_delta --> The amount of time it takes to reconfigure the network
# Output:
#   new_alphas --> Set of alphas that have been truncated (if the alpha would go over the window_size) or extended (if the reconfiguration after the alpha would go over the window size)
def filterAlphas(alphas, total_duration_so_far, num_matchings, window_size, reconfig_delta):
	Profiler.start('filterAlphas')

	new_alphas = set()

	for alpha in alphas:
		# Check if adding a matching with this alpha would go over the window size
		if alpha + (reconfig_delta if num_matchings >= 1 else 0) + total_duration_so_far > window_size:
			# Truncates alpha in this case
			alpha = window_size - total_duration_so_far - (reconfig_delta if num_matchings >= 1 else 0)

		# Checks if adding this matching would make it impossible to add another matching (even with alpha 1)
		elif alpha + (reconfig_delta if num_matchings >= 1 else 0) + total_duration_so_far + reconfig_delta >= window_size:
			# Extends alpha in this case
			alpha = window_size - total_duration_so_far - (reconfig_delta if num_matchings >= 1 else 0)

		new_alphas.add(alpha)

	Profiler.end('filterAlphas')
	return new_alphas

# Finds a random stable matching based on the given weights
def findRandomStableMatching(num_nodes, graph):
	# TODO - Add some randomness to this.

	# input_weight[i] is the prefences of input port i from highest weight to lowest weight. Each element is a 2-tuple of (index of output port, weight).
	input_weight = [sorted([(j, graph[i][j]) for j in range(num_nodes)], key = lambda x: x[1], reverse = True) for i in range(num_nodes)]

	# output_weight[i] is the prefences of input port i from highest weight to lowest weight. Each element is a 2-tuple of (index of output port, weight).
	output_weight = [sorted([(i, graph[i][j]) for i in range(num_nodes)], key = lambda x: x[1], reverse = True) for j in range(num_nodes)]

	input_assignment =  [-1 for i in range(num_nodes)]
	output_assignment = [-1 for j in range(num_nodes)]

	while any(j == -1 for j in input_assignment) or any(i == -1 for i in output_assignment):
		this_input = next(i for i in range(num_nodes) if input_assignment[i] == -1)

		for this_output, this_weight in input_weight[this_input]:
			check_weight = next(w for i, w in output_weight[this_output] if i == this_input)
			if this_weight != check_weight:
				raise Exception('Unexpected weights')

			if output_assignment[this_output] == -1:
				input_assignment[this_input]   = this_output
				output_assignment[this_output] = this_input
				break
			else:
				other_input = output_assignment[this_output]

				other_weight = next(w for j, w in input_weight[other_input] if j == this_output)

				if this_weight > other_weight:
					input_assignment[other_input] = -1

					input_assignment[this_input]   = this_output
					output_assignment[this_output] = this_input
					break
	matching = set((i, j) for i, j in enumerate(input_assignment))
	return matching

def findRandomStableMatchingAllWeightsOne(num_nodes, graph):
	new_graph = np.zeros((num_nodes, num_nodes))
	for i in range(num_nodes):
		for j in range(num_nodes):
			if graph[i][j] > 0.0:
				new_graph[i][j] = 1.0

	return findRandomStableMatching(num_nodes, graph)

# Finds a random matching for the complete graph of the given size network.
def findRandomMatching(num_nodes):
	input_nodes  = [i for i in range(num_nodes)]
	output_nodes = [i for i in range(num_nodes)]

	random.shuffle(output_nodes)

	matching = set((i, j) for i, j in zip(input_nodes, output_nodes))
	return matching

# Computes the optimal matching for the given value of alpha.
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
#   alpha --> Duration of matching
#   num_nodes --> The number of nodes in the network
#   all_weights --> weight of edges in the graph
#   use_random_matching --> If True, then will return a random matching instead of a maximum weight matching.
# Output:
#   matching --> Optimal matching
#   matching_weight --> weight of the optimal matching
def getMatching(subflows_by_next_hop, alpha, num_nodes, all_weights, use_random_matching = False, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	# Creates the graph for the given alpha
	graph = createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes, all_weights, max_weight_matching_library = max_weight_matching_library)

	if use_random_matching:
		# Random stable matching using Octopus weights
		# Profiler.start('findRandomStableMatching')
		# matching = findRandomStableMatching(num_nodes, graph)
		# Profiler.end('findRandomStableMatching')

		# Random matching over edges wtih some amount of traffic
		Profiler.start('findRandomStableMatchingAllWeightsOne')
		matching = findRandomStableMatchingAllWeightsOne(num_nodes, graph)
		Profiler.end('findRandomStableMatchingAllWeightsOne')

		# Completely random matching
		# Profiler.start('findRandomMatching')
		# matching = findRandomMatching(num_nodes)
		# Profiler.end('findRandomMatching')

		matching_weight = sum(graph[a][b] for a, b in matching)

	# Find the maximum weight matching of the graph using a 3rd party library
	elif max_weight_matching_library is 'networkx':
		Profiler.start('networkx.max_weight_matching')
		matching = nx.algorithms.matching.max_weight_matching(graph)
		Profiler.end('networkx.max_weight_matching')

		matching_weight = sum(graph[a][b]['weight'] for a, b in matching)
	elif max_weight_matching_library is 'scipy':
		Profiler.start('scipy.optimize.linear_sum_assignment')
		matching = scipy_opt.linear_sum_assignment(-graph)
		Profiler.end('scipy.optimize.linear_sum_assignment')

		matching_weight = graph[matching[0], matching[1]].sum()
	else:
		raise Exception('Invalid max_weight_matching_library: ' + str(max_weight_matching_library))

	matching = convertMatching(graph, matching, num_nodes, use_random_matching = use_random_matching, max_weight_matching_library = max_weight_matching_library)

	return matching, matching_weight

# Finds the matching and alpha that maximizes sum of weights of the matching / (alpha + reconfig_delta)
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
#   alphas --> Set of all alpha values to try
#   num_nodes --> The number of nodes in the network
#   reconfig_delta --> the reconfiguraiton delta of the network
#   search_method --> Search method to find the best alpha. Must be either 'iterative' or 'search'
#   use_random_matching --> If true, then will find random matchings instead of maximum weight matchings.
# Output:
#   best_objective_value --> The objective value of the best matching and alpha. (i.e. total weight of matching / (alpha + reconfig_delta))
#   best_matching_weight --> The total weight of the matching that maximizes the objective.
#   best_alpha --> The best alpha found
#   best_matching --> The best matching found. It is a set of (src, dst) edges.
def findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta, search_method = 'iterative', use_random_matching = False, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	if search_method not in ['iterative', 'search']:
		raise Exception('Unexpected search_method: ' + str(search_method))

	Profiler.start('findBestMatching')

	best_objective_value = None
	best_matching_weight = None
	best_alpha           = None
	best_matching        = None

	# Calculates weights for all edges and alphas at once
	all_weights_by_alpha = calculateAllWeights(subflows_by_next_hop, alphas)

	# Tries all values of alpha
	if search_method == 'iterative':
		# Tries all alpha values in alphas and finds the maximum weighted matching in each
		for alpha in sorted(alphas):
			matching, matching_weight = getMatching(subflows_by_next_hop, alpha, num_nodes, all_weights_by_alpha[alpha], use_random_matching = use_random_matching, max_weight_matching_library = max_weight_matching_library)

			# Track the graph that maximizes value(G) / (alpha + reconfig_delta)
			this_objective_value = matching_weight / (alpha + reconfig_delta)
			if best_objective_value is None or this_objective_value > best_objective_value:
				best_objective_value = this_objective_value
				best_matching_weight = matching_weight
				best_alpha           = alpha
				best_matching        = matching

	# Does a search that tries only a limited number of alphas
	if search_method == 'search':
		num_queries = 4
		low_ind  = 0
		high_ind = len(alphas) - 1
		list_alphas = sorted(list(alphas))
		matchings_by_alpha = {alpha: None for alpha in list_alphas}

		while True:
			query_inds = sorted(list(set(int(float(high_ind - low_ind) / float(num_queries - 1) * i + low_ind) for i in range(num_queries))))
			is_last_iter = all(a + 1 == b for a, b in zip(query_inds[:-1], query_inds[1:]))

			objective_values = []
			for qi in query_inds:
				this_alpha = list_alphas[qi]

				if matchings_by_alpha[this_alpha] is None:
					matching, matching_weight = getMatching(subflows_by_next_hop, this_alpha, num_nodes, all_weights_by_alpha[this_alpha], use_random_matching = use_random_matching, max_weight_matching_library = max_weight_matching_library)
					
					matchings_by_alpha[this_alpha] = (matching, matching_weight)

				objective_values.append(matchings_by_alpha[this_alpha][1] / (this_alpha + reconfig_delta))

			max_ov = max(objective_values)
			max_inds = [i for i, ov in enumerate(objective_values) if abs(max_ov - ov) < 0.0000000001]

			# if len(max_inds) != 1:
			# 	raise Exception('Unexpected values')

			max_ind = max_inds[0]

			if is_last_iter:
				this_alpha = list_alphas[query_inds[max_ind]]

				best_objective_value = objective_values[max_ind]
				best_matching_weight = matchings_by_alpha[this_alpha][1]
				best_alpha           = this_alpha
				best_matching        = matchings_by_alpha[this_alpha][0]

				break

			if max_ind == 0:
				low_ind  = query_inds[0]
				high_ind = query_inds[1]

			elif max_ind == len(query_inds) - 1:
				low_ind  = query_inds[-2]
				high_ind = query_inds[-1]

			else:
				low_ind  = query_inds[max_ind - 1]
				high_ind = query_inds[max_ind + 1]

	Profiler.end('findBestMatching')
	return best_objective_value, best_matching_weight, best_alpha, best_matching

# Given the subflows (grouped by their next hop) and a matching duration, computes the bipartite graph with weighted edges.
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
#   alpha --> The matching duration given in # of packets.
#   num_nodes --> Number of nodes in the network.
#   all_weights --> weights for each edge (src, dst)
# Ouput:
#   Returns a nx.Graph object that is a complete bipartite graph with edge weights. Note the edge (x, y) in the returned graph corresponds the edge (x, y - num_node) in the rest of the code.
def createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes, all_weights, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	Profiler.start('createBipartiteGraph')

	# Creates the graph
	if max_weight_matching_library is 'networkx':
		graph = nx.Graph()

		# Creates nodes for each input port. The input port of i is node i in the graph.
		graph.add_nodes_from(xrange(        0,     num_nodes), bipartite = 0)

		# Creates nodes for each output port. The output port of i is node i + num_nodes in the graph.
		graph.add_nodes_from(xrange(num_nodes, 2 * num_nodes), bipartite = 1)

		# Creates the weighted edges in bipartite graph
		graph.add_weighted_edges_from([(i, j + num_nodes, all_weights[(i, j)]) for (i, j), subflows in subflows_by_next_hop.iteritems()])

	elif max_weight_matching_library is 'scipy':
		graph = np.zeros((num_nodes, num_nodes))

		for (i, j), subflows in subflows_by_next_hop.iteritems():
			graph[i][j] = all_weights[(i, j)]

	else:
		raise Exception('Invalid max_weight_matching_library: ' + str(max_weight_matching_library))

	Profiler.end('createBipartiteGraph')
	return graph

# |------------------|
# |     OUTDATED     |
# |------------------|
# Finds the weighted number of packets that can be sent of subflows in alpha time.
# Input:
#   subflows --> list of SubFlows sorted by increasing getInvweight / decreasing getWeight()
#   alpha --> duration that data can be sent
# Return
#   Returns maximum weighted sum of packets that can be sent within alpha time.
def calculateTotalWeight(subflows, alpha):
	Profiler.start('calculateTotalWeight')

	unused_time = alpha
	weighted_sum = 0.0
	for subflow in subflows:
		if subflow.getSize() < unused_time:
			weighted_sum += subflow.getSize() * subflow.getWeight()
			unused_time  -= subflow.getSize()
		else:
			weighted_sum += unused_time * subflow.getWeight()
			unused_time  -= unused_time
			break

	Profiler.end('calculateTotalWeight')
	return weighted_sum

# Calculates the weighted sum for each edge and for each possible alpha
# Input:
#   subflows_by_next_hop --> Subflows grouped by next hop
#   alphas --> set of all possible alphas
# Outpu:
#   all_weights_by_alpha --> Dict keyed by alpha, each value is a dictionary keyed by edge (src, dst) with values of the edge weight.
def calculateAllWeights(subflows_by_next_hop, alphas):
	Profiler.start('calculateAllWeights')
	all_weights_by_alpha = defaultdict(dict)

	sorted_alphas = sorted(alphas)

	for (i, j), subflows in subflows_by_next_hop.iteritems():
		# Goes through alphas and subflows together
		cur_alpha_ind = 0
		cur_subflow_ind = 0

		weighted_sum = 0.0   # Weighted packets for subflows already processed
		used_time    = 0     # Number of packets from subflows already processed
		while cur_subflow_ind < len(subflows) and cur_alpha_ind < len(sorted_alphas):
			if sorted_alphas[cur_alpha_ind] <= used_time + subflows[cur_subflow_ind].getSize():
				# If the alpha lands halfway through a subflow, calculates this alpha, and advance the alpha
				all_weights_by_alpha[sorted_alphas[cur_alpha_ind]][(i, j)] = weighted_sum + subflows[cur_subflow_ind].getWeight(eps = EPS) * (sorted_alphas[cur_alpha_ind] - used_time)
				cur_alpha_ind += 1
			else:
				# If the alpha goes into the next subflow, update weighted_sum and used_time, and advance the subflow
				weighted_sum += subflows[cur_subflow_ind].getWeight(eps = EPS) * subflows[cur_subflow_ind].getSize()
				used_time    += subflows[cur_subflow_ind].getSize()
				
				cur_subflow_ind += 1

		# Add the weighted_sum for any alpha tha tis longer than all of the subflows
		while cur_alpha_ind < len(sorted_alphas):
			all_weights_by_alpha[sorted_alphas[cur_alpha_ind]][(i, j)] = weighted_sum
			cur_alpha_ind += 1

	Profiler.end('calculateAllWeights')

	return all_weights_by_alpha

# Converts the networkx matching to our matching
# Input:
#   matching --> The output of nx.algorithms.matching.max_weight_matching, a set of (node_id, node_id)
#   num_nodes --> number of nodes in the network
#   use_random_matching --> Indicates if we created a random matching.
# Output:
#   Returns the matching in our format. For each edge (x, y) in the inputted matching, the output will contain (min(x, y), max(x, y) - num_nodes)
def convertMatching(graph, matching, num_nodes, use_random_matching = False, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	Profiler.start('convertMatching')
	if use_random_matching:
		new_matching = matching
	elif max_weight_matching_library is 'networkx':
		new_matching = set()
		for edge in matching:
			src = min(edge)
			dst = max(edge)

			new_matching.add((src, dst - num_nodes))
	elif max_weight_matching_library is 'scipy':
		new_matching = set()

		for i, j in zip(matching[0], matching[1]):
			if graph[i, j] >= 10e-6:
				new_matching.add((i, j))
	else:
		raise Exception('Invalid max_weight_matching_library: ' + str(max_weight_matching_library))

	Profiler.end('convertMatching')
	return new_matching

# Sends as many subflows as possible within the given time period
# Input:
#   subflows --> list of subflows to update
#   alpha --> total number of packets that can be sent
# Output:
#   total_packets_sent --> The total number of packets sent
#   remaing_alpha --> The number of packets that can still be sent.
#   updated_subflows --> List of all subflows that were updated. Doesn't include subflows that were split from existing ones.
#   finished_subflows --> List of subflows that have reached their destination.
#   new_subflows --> List of new subflows that were split from one the inputted subflows.
def updateSubFlows(subflows, alpha, track_updated_subflows = False):
	Profiler.start('updateSubFlows')

	updated_subflows  = [] # All subflows that are updated (not included any split_subflows)
	finished_subflows = [] # All subflows that have reached their destination
	new_subflows      = [] # All new subflows that were split from existing subflows

	total_packets_sent = 0

	for subflow in subflows:
		packets_sent, alpha, split_subflow, is_subflow_done = subflow.send(alpha)

		total_packets_sent += packets_sent

		if track_updated_subflows:
			# Only maintains this list if requested in order to conserve memory
			updated_subflows.append(subflow)

		if is_subflow_done:
			finished_subflows.append(subflow)

		if split_subflow is not None:
			new_subflows.append(split_subflow)

		if alpha <= 0:
			break

	remaing_alpha = alpha

	Profiler.end('updateSubFlows')
	return total_packets_sent, remaing_alpha, updated_subflows, finished_subflows, new_subflows

# Computes the multi-hop schedule for the given flows
# Input:
#   num_nodes --> Number of nodes in the network
#   flows --> dict with keys (src_node_id, dst_node_id) and values of input_utils.Flow
#   window_size --> The total duration to compute the schedule
#   reconfig_delta --> The time it takes to reconfigure the network
#   upper_bound --> Runs the upper-bound method, where there are no order constraints for routes
#   return_packets_delivered_by_flow_id --> If given, the number of packets delivered for each flow
#   flows_have_given_invweight --> If given, then flows has values of (input_utils.Flow, invweight) instead of just input_utils.Flow
#   precomputed_schedule --> If given, uses the given schedule instead of computed maximum weight matchings
#   alpha_search_method --> Method used to find the optimal alpha. Must be either 'iterative' or 'search'
#   use_random_matching --> Uses random matchings instead of maximum weight matching
#   global_override_weight --> If a non-None given, then all subflows will have the given weight.
# Output:
#   schedule --> The computed schedule, containing the matchings and durations
#   result_metric --> Holds the various metrics to judge the algorithm
#   (Optional) completed_flow_ids --> List of flow_ids of subflows that reach their destination
def computeSchedule(num_nodes, flows, window_size, reconfig_delta, upper_bound = False, return_packets_delivered_by_flow_id = False, flows_have_given_invweight = False, precomputed_schedule = None, consider_all_routes = False, backtrack = False, alpha_search_method = 'iterative', fixed_alpha = None, use_random_matching = False, global_override_weight = None, verbose = False):
	Profiler.start('computeSchedule')
	# TODO Check that the set of flags are consistent

	# Initialzie subflows (same as the remaining traffic)
	if upper_bound:
		starting_subflows = []
		sf_id = 0
		for _, flow in flows.iteritems():
			for i in range(len(flow.route) - 1):
				starting_subflows.append(SubFlow(flow = flow, override_subflow_id = sf_id, override_route = [flow.route[i], flow.route[i + 1]], override_invweight = flow.invweight()))

				sf_id += 1

	elif consider_all_routes:
		starting_subflows = [SubFlow(flow = flow, override_route = route, tag = str(len(route) - 1) + '-hop') for _, flow in flows.iteritems() for route in flow.all_routes]
	elif flows_have_given_invweight:
		starting_subflows = [SubFlow(flow = flow, override_invweight = invweight) for _, (flow, invweight) in flows.iteritems()]
	elif global_override_weight is not None:
		starting_subflows = [SubFlow(flow = flow, override_weight = global_override_weight) for _, flow in flows.iteritems()]
	else:
		starting_subflows = [SubFlow(flow = flow) for _, flow in flows.iteritems()]

	track_updated_subflows = consider_all_routes or backtrack

	current_subflows_by_subflow_id = defaultdict(list)
	for subflow in starting_subflows:
		current_subflows_by_subflow_id[subflow.subflowID()].append(subflow)
	completed_subflows = []

	# Initialize the schedule, which will hold the result
	schedule = Schedule()

	# Track time slots where packets are and aren't sent
	time_slots_used = 0
	time_slots_not_used = 0

	if precomputed_schedule is not None:
		if precomputed_schedule.totalDuration(reconfig_delta) != window_size:
			raise Exception('Precomputed schedule does not match the given window_size')

		precomputed_schedule_ind = 0

	# Used to measure computation time
	compute_start = time.time()
	
	while schedule.totalDuration(reconfig_delta) < window_size:
		if verbose:
			print 'Starting iteration with', window_size - schedule.totalDuration(reconfig_delta), 'time left'

		Profiler.start('computeSchedule-iteration')

		if backtrack:
			Profiler.start('computeSchedule-add_backtrack')
			# Check if a subflow can be "backtracking", if so adds the backtrack subflow
			for _, subflows in current_subflows_by_subflow_id.iteritems():
				# Can only backtrack on single subflows with two or more hops left, and that has a direct route in all_routes
				if len(subflows) == 1 and subflows[0].remainingRouteLength() >= 2 and any(len(flow_route) == 2 for flow_route in subflows[0].flow.all_routes):
					this_subflow = subflows[0]

					# The backtrack route is just a direct route to the destination
					backtrack_route = [this_subflow.srcNode(), this_subflow.destNode()]

					# Finds the weight of the backtrack subflow. It is equal to the sum of the weights of the remaining hops in the subflow
					backtrack_weight = this_subflow.remainingRouteLength() * this_subflow.getWeight()

					if global_override_weight is None:
						raise Exception('Cannot both backtrack and use global_override_weight at the same time')

					# Creates the backtrack_subflow and adds it to the system.
					backtrack_subflow = SubFlow(subflow = this_subflow, override_route = backtrack_route, override_size = this_subflow.getSize(), override_subflow_id = this_subflow.subflowID(), override_weight = backtrack_weight, tag = '1-hop_backtrack')

					this_subflow.is_backtrack = True
					backtrack_subflow.is_backtrack = True
					
					current_subflows_by_subflow_id[backtrack_subflow.subflowID()].append(backtrack_subflow)
			Profiler.end('computeSchedule-add_backtrack')

		# Gets the subflows_by_next_hop information from remaining_flows
		# Keys are (curNode(), nextNode()) and values are a list of NextHopFlows
		Profiler.start('computeSchedule-subflows_by_next_hop')
		subflows_by_next_hop = defaultdict(list)
		for _, subflows in current_subflows_by_subflow_id.iteritems():
			for subflow in subflows:
				subflows_by_next_hop[(subflow.curNode(), subflow.nextNode())].append(subflow)
		Profiler.end('computeSchedule-subflows_by_next_hop')

		# Sorts each list in subflows_by_next_hop
		for k in subflows_by_next_hop:
			sortSubFlows(subflows_by_next_hop[k])

		if precomputed_schedule is None:
			if fixed_alpha is None:
				# Get set of unique alphas to consider
				alphas = getUniqueAlphas(subflows_by_next_hop)

			else:
				# If a fixed alpha is given, then we consider only that one.
				alphas = set([fixed_alpha])

			# Truncate alphas that go over the duration
			alphas = filterAlphas(alphas, schedule.totalDuration(reconfig_delta), schedule.numMatchings(), window_size, reconfig_delta)

			# Track the best alpha and matching
			objective_value, matching_weight, alpha, matching = findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta, search_method = alpha_search_method, use_random_matching = use_random_matching)

		else:
			# Get next matching and alpha from precomputed_schedule
			matching = precomputed_schedule.matchings[precomputed_schedule_ind]
			alpha    = precomputed_schedule.durations[precomputed_schedule_ind]
			precomputed_schedule_ind += 1

			# Calculate this matchings weight
			matching_weight = sum(calculateTotalWeight(subflows_by_next_hop[(i, j)], alpha) for i, j in matching)

		# Add best matching and its associated alpha to Schedule
		schedule.addMatching(matching, alpha, matching_weight = matching_weight)

		# Update remaining
		Profiler.start('computeSchedule-updateAllSubflows')
		if backtrack:
			backtrack_updated_subflows_by_subflow_ids  = defaultdict(list)
			backtrack_finished_subflows_by_subflow_ids = defaultdict(list)
			backtrack_new_subflows_by_subflow_ids      = defaultdict(list)
		
		for edge in matching:
			packets_sent, unused_alpha, updated_subflows, finished_subflows, new_subflows = updateSubFlows(subflows_by_next_hop[edge], alpha, track_updated_subflows = track_updated_subflows)
			
			time_slots_used += packets_sent
			time_slots_not_used += unused_alpha

			# Removes subflows of other routes once one of the subflows was chosen.
			if consider_all_routes or backtrack:
				for updated_subflow in updated_subflows:
					# Gets the other subflows with the same subflows
					related_subflows = current_subflows_by_subflow_id[updated_subflow.subflowID()]

					# If there is only one other related subflow it should be the updated_subflow itself, so there is nothing to be done
					if len(related_subflows) == 1:
						if updated_subflow is not related_subflows[0]:
							raise Exception('Unexpected subflow in related_subflows')

						continue

					if len(related_subflows) == 0:
						raise Exception('No related_subflows found')

					if updated_subflow.is_backtrack:
						if backtrack and len(related_subflows) == 2 and all(sf.is_backtrack for sf in related_subflows):
							backtrack_updated_subflows_by_subflow_ids[updated_subflow.subflowID()].append(updated_subflow)
						else:
							raise Exception('Unexpected is_backtrack flag')
					else:
						# Remove all other related subflows, by simply replacing the list with a list with just this subflow
						current_subflows_by_subflow_id[updated_subflow.subflowID()] = [updated_subflow]

			# Removes any subflows that have finished from current_subflows_by_subflow_id and places adds them to completed_subflows
			for fin_subflow in finished_subflows:
				if fin_subflow.is_backtrack:
					if backtrack and fin_subflow.subflowID() in backtrack_updated_subflows_by_subflow_ids and fin_subflow in backtrack_updated_subflows_by_subflow_ids[fin_subflow.subflowID()]:
						backtrack_finished_subflows_by_subflow_ids[fin_subflow.subflowID()].append(fin_subflow)
					else:
						raise Exception('Quit')
				else:
					current_subflows_by_subflow_id[fin_subflow.subflowID()].remove(fin_subflow)

					if len(current_subflows_by_subflow_id[fin_subflow.subflowID()]) == 0:
						del current_subflows_by_subflow_id[fin_subflow.subflowID()]

					completed_subflows.append(fin_subflow)

			# Adds any sublfows that were split to current_subflows_by_subflow_id
			for new_subflow in new_subflows:
				if new_subflow.is_backtrack:
					if not backtrack or new_subflow.parent_subflow is None:
						raise Exception('NOOOOOOOOOOOOOO')
					backtrack_new_subflows_by_subflow_ids[new_subflow.parent_subflow.subflowID()].append(new_subflow)

				else:
					current_subflows_by_subflow_id[new_subflow.subflowID()].append(new_subflow)

					# If the split subflow is still at the beginning then we still need to consider all routes
					if consider_all_routes and len(new_subflow.route) == len(new_subflow.rem_route) and all(a == b for a, b in zip(new_subflow.route, new_subflow.rem_route)):
						this_subflow_id = new_subflow.subflowID()
						
						for other_route in new_subflow.flow.all_routes:
							if len(other_route) == len(new_subflow.route) and all(a == b for a, b in zip(other_route, new_subflow.route)):
								continue

							other_subflow = SubFlow(flow = new_subflow.flow, override_subflow_id = this_subflow_id, override_route = other_route, override_size = new_subflow.getSize(), tag = str(len(other_route) - 1) + '-hop', override_weight = global_override_weight)
				
							current_subflows_by_subflow_id[other_subflow.subflowID()].append(other_subflow)

		if backtrack:
			Profiler.start('computeSchedule-process_backtrack')
			for sf_id in backtrack_updated_subflows_by_subflow_ids:
				# Figure out the six subflows
				updated_subflows  = backtrack_updated_subflows_by_subflow_ids[sf_id]
				finished_subflows = backtrack_finished_subflows_by_subflow_ids[sf_id]
				new_subflows      = backtrack_new_subflows_by_subflow_ids[sf_id]
				old_subflows      = current_subflows_by_subflow_id[sf_id]

				backtrack_updated = None
				backtrack_split   = None
				backtrack_old     = None
				regular_updated   = None
				regular_split     = None
				regular_old       = None

				if len(finished_subflows) > 1:
					raise Exception('Error with backtracking')

				if len(finished_subflows) == 1:
					backtrack_updated = finished_subflows[0]
					finished_subflows.remove(backtrack_updated)
					updated_subflows.remove(backtrack_updated)

				if len(updated_subflows) > 1:
					raise Exception('Error with backtracking')

				if len(updated_subflows) == 1:
					regular_updated = updated_subflows[0]
					updated_subflows.remove(regular_updated)

				if len(updated_subflows) != 0 or len(finished_subflows) != 0:
					raise Exception('Error with backtracking')

				if backtrack_updated is None and regular_updated is None:
					raise Exception('What is even happening')

				if backtrack_updated is not None and regular_updated is None:
					if len(new_subflows) > 1:
						raise Exception('Error with backtracking')

					if len(new_subflows) == 1:
						backtrack_split = new_subflows[0]
						new_subflows.remove(backtrack_split)

				if backtrack_updated is None and regular_updated is not None:
					if len(new_subflows) > 1:
						raise Exception('Error with backtracking')

					if len(new_subflows) == 1:
						regular_split = new_subflows[0]
						new_subflows.remove(regular_split)

				if backtrack_updated is not None and regular_updated is not None:
					if len(new_subflows) > 2:
						raise Exception('Error with backtracking')

					if len(new_subflows[0].route) - 1 == 1 and (len(new_subflows) == 1 or len(new_subflows[1].route) - 1 > 1):
						backtrack_split = new_subflows[0]
						if len(new_subflows) == 2:
							regular_split   = new_subflows[1]

					elif len(new_subflows[0].route) - 1 > 1 and (len(new_subflows) == 1 or len(new_subflows[1].route) - 1 == 1):
						regular_split   = new_subflows[0]
						if len(new_subflows) == 2:
							backtrack_split = new_subflows[1]
					else:
						raise Exception('Error with backtracking')

					if backtrack_split is not None:
						new_subflows.remove(backtrack_split)

					if regular_split is not None:
						new_subflows.remove(regular_split)

				if len(old_subflows) != 2:
					raise Exception('Error with backtracking')

				if len(old_subflows[0].route) - 1 == 1 and len(old_subflows[1].route) - 1 > 1:
					backtrack_old = old_subflows[0]
					regular_old   = old_subflows[1]

				elif  len(old_subflows[0].route) - 1 > 1 and len(old_subflows[1].route) - 1 == 1:
					regular_old   = old_subflows[0]
					backtrack_old = old_subflows[1]
				else:
					raise Exception('Error with backtracking')

				is_none = [v is None for v in [backtrack_updated, backtrack_split, regular_updated, regular_split]]

				# TODO update is_backtrack

				if all(a == b for a, b in zip(is_none, [True,  True,  True,  True])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, True,  True,  True])):
					# In this case all packets are delivered via the backtrack route, so we can just abandon the other route
					del current_subflows_by_subflow_id[backtrack_updated.subflowID()]

					completed_subflows.append(backtrack_updated)

					backtrack_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [True,  False, True,  True])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, False, True,  True])):
					# In this case part of the backtrack flow finished.
					# Mark that subflow finished
					current_subflows_by_subflow_id[backtrack_updated.subflowID()].remove(backtrack_updated)

					completed_subflows.append(backtrack_updated)

					# Change backtrack_split's subflow_id to match regular_old
					backtrack_split.subflow_id = regular_old.subflowID()
					current_subflows_by_subflow_id[backtrack_split.subflowID()].append(backtrack_split)

					# Change regular_old's size to match backtrack_split
					regular_old.size = backtrack_split.getSize()

					backtrack_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [True,  True,  False, True])):
					# All packets were sent via the regular route, we can just abandon the backtrack route
					current_subflows_by_subflow_id[regular_updated.subflowID()] = [regular_updated]

					regular_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [False, True,  False, True])):
					# All packets sent via both the backtrack and regular route, ignore the regular route
					del current_subflows_by_subflow_id[backtrack_updated.subflowID()]

					completed_subflows.append(backtrack_updated)

					backtrack_updated.is_backtrack = False
					regular_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [True,  False, False, True])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, False, False, True])):
					# Some packets sent via backtrack route, but all sent via regular.
					# Packets sent via backtrack finish, and rest are set to the regular route.
					# Can just abandon backtrack_split
					current_subflows_by_subflow_id[backtrack_updated.subflowID()].remove(backtrack_updated)
					completed_subflows.append(backtrack_updated)

					regular_updated.size = backtrack_split.getSize()

					regular_updated.is_backtrack = False
					backtrack_split.is_backtrack = False
					backtrack_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [True,  True,  True,  False])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, True,  True,  False])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [True,  False, True,  False])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, False, True,  False])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [True,  True,  False, False])):
					# Some packets sent via regular. Just discard backtrack_old. Regular backtrack code will add appropriate flows as needed.
					current_subflows_by_subflow_id[backtrack_old.subflowID()].remove(backtrack_old)
					current_subflows_by_subflow_id[regular_split.subflowID()].append(regular_split)

					regular_split.is_backtrack = False
					regular_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [False, True,  False, False])):
					# All packets delivered via backtracking, so can discard everything else
					current_subflows_by_subflow_id[regular_updated.subflowID()].remove(regular_updated)

					current_subflows_by_subflow_id[backtrack_updated.subflowID()].remove(backtrack_updated)
					completed_subflows.append(backtrack_updated)

					backtrack_updated.is_backtrack = False
					regular_split.is_backtrack = False
					regular_updated.is_backtrack = False

				if all(a == b for a, b in zip(is_none, [True,  False, False, False])):
					raise Exception('Error with backtracking')
				if all(a == b for a, b in zip(is_none, [False, False, False, False])):
					if regular_split.getSize() + regular_updated.getSize() != backtrack_split.getSize() + backtrack_updated.getSize():
						raise Exception('Error with backtracking')

					backtrack_split.is_backtrack = False

					# Backtrack route takes first priority
					current_subflows_by_subflow_id[backtrack_updated.subflowID()].remove(backtrack_updated)

					backtrack_updated.is_backtrack = False
					completed_subflows.append(backtrack_updated)

					# If possible take delivered packets away from regular_split
					if regular_split.getSize() - backtrack_updated.getSize() > 0:
						regular_split.size = regular_split.getSize() - backtrack_updated.getSize()

						current_subflows_by_subflow_id[regular_split.subflowID()].append(regular_split)

						regular_split.is_backtrack = False
						regular_updated.is_backtrack = False
					else:
						regular_updated.size -= backtrack_updated.getSize() - regular_split.getSize()

						regular_updated.is_backtrack = False
			Profiler.end('computeSchedule-process_backtrack')
		Profiler.end('computeSchedule-updateAllSubflows')
		Profiler.end('computeSchedule-iteration')

	# Used to measure computation time
	compute_end = time.time()
	compute_dur = compute_end - compute_start

	# Compute result metrics
	total_objective_value = schedule.getTotalMatchingWeight()

	if flows_have_given_invweight:
		total_packets = sum(flow.size for _, (flow, _) in flows.iteritems())
	else:
		total_packets = sum(flow.size for _, flow in flows.iteritems())


	if upper_bound:

		completed_subflows_by_flow_id = {flow.id: (flow, []) for _, flow in flows.iteritems()}
		for fin_subflow in completed_subflows:
			completed_subflows_by_flow_id[fin_subflow.flowID()][1].append(fin_subflow)

		packets_delivered = 0
		packets_not_delivered = 0
		for _, (flow, fin_subflows) in completed_subflows_by_flow_id.iteritems():
			packets_delivered_by_route = {(flow.route[i], flow.route[i + 1]): 0 for i in range(len(flow.route) - 1)}
			for fin_subflow in fin_subflows:
				packets_delivered_by_route[tuple(fin_subflow.route)] += fin_subflow.getSize()

			this_packets_delivered = min(packets for _, packets in packets_delivered_by_route.iteritems())

			packets_delivered     += this_packets_delivered
			packets_not_delivered += flow.size - this_packets_delivered

		packets_delivered_by_tag = None
	else:
		packets_delivered = sum(subflow.getSize() for subflow in completed_subflows)

		packets_delivered_by_tag = defaultdict(lambda: 0)
		for subflow in completed_subflows:
			if subflow.tag is not None:
				packets_delivered_by_tag[subflow.tag] += subflow.getSize()

		# If subflows are tagged, then all subflows should be tagged, and then the sum of packets delivered to each tag should equal the total number of packets delivered
		if len(packets_delivered_by_tag) > 0 and packets_delivered != sum(packets for _, packets in packets_delivered_by_tag.iteritems()):
			raise Exception('Delivered tagged packets do not equal total number of packets delivered')

		# All subflows with the same subflow ID represent one undelivered subflow. All of these subflows should have the same size, and there should be no empty lists.
		for _, subflows in current_subflows_by_subflow_id.iteritems():
			if len(subflows) == 0:
				raise Exception('Still have a subflow list with no subflows')

			if any(subflow.getSize() != subflows[0].getSize() for subflow in subflows):
				raise Exception('Some subflows with the same ID don\'t have the same size')

		packets_not_delivered = sum(subflows[0].getSize() for _, subflows in current_subflows_by_subflow_id.iteritems())

	if total_packets != packets_delivered + packets_not_delivered:
		raise Exception('Not all packets accounted for: total_packets = ' + str(total_packets) + ', packets_delivered = ' + str(packets_delivered) + ', packets_not_delivered = ' + str(packets_not_delivered))

	# Add in reconfiguration delay to unused time slots
	time_slots_not_used += reconfig_delta * num_nodes * schedule.numReconfigs()

	result_metric = ResultMetric(total_objective_value, packets_delivered, packets_not_delivered, window_size, time_slots_used, time_slots_not_used, packets_delivered_by_tag = packets_delivered_by_tag, computation_duration = compute_dur)

	rvs = [schedule, result_metric]

	# If requested, returns the set of flow_ids that were completed
	if return_packets_delivered_by_flow_id:
		# TODO Really need to return the number of packets delivered for each flowID. This is to handle split subflows
		if flows_have_given_invweight:
			packets_delivered_by_flow_id = {flow.id: 0 for _, (flow, invweight) in flows.iteritems()}
		else:
			packets_delivered_by_flow_id = {flow.id: 0 for _, flow in flows.iteritems()}

		for fin_subflow in completed_subflows:
			packets_delivered_by_flow_id[fin_subflow.flowID()] += fin_subflow.getSize()

		rvs.append(packets_delivered_by_flow_id)

	Profiler.end('computeSchedule')
	return tuple(rvs)
