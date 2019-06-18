
from profiler import Profiler

from collections import defaultdict
import networkx as nx

# Holds a multi-hop schedule
class Schedule:
	def __init__(self):
		self.matchings = []
		self.durations = []

		self.matching_weights = []

	def addMatching(self, matching, duration, matching_weight = None):
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

# Holds the different metrics for the result of one run
class ResultMetric:
	def __init__(self, total_objective_value, packets_delivered, packets_not_delivered, total_duration, time_slots_used, time_slots_not_used):
		self.total_objective_value = total_objective_value
		self.packets_delivered     = packets_delivered
		self.packets_not_delivered = packets_not_delivered
		self.total_duration        = total_duration
		self.time_slots_used       = time_slots_used
		self.time_slots_not_used   = time_slots_not_used

	def getLinkUtilization(self):
		return float(self.time_slots_used) / float(self.time_slots_used + self.time_slots_not_used)

	def __str__(self):
		normalize = lambda v: float(v) / float(self.total_duration)

		return '{Total Objective Value: ' + str(self.total_objective_value) + ', ' + \
				'Normalized Objective Value: ' + str(normalize(self.total_objective_value)) + ', ' + \
				'Packets Delivered: ' + str(self.packets_delivered) + ', ' + \
				'Normalized Packets Delivered: ' + str(normalize(self.packets_delivered)) + ', ' + \
				'Packets Not Delivered: ' + str(self.packets_not_delivered) + ', ' + \
				'Normalized Packets Not Delivered: ' + str(normalize(self.packets_not_delivered)) + ', ' + \
				'Time Slots Used: ' + str(self.time_slots_used) + ', ' + \
				'Time Slots Unused: ' + str(self.time_slots_not_used) + ', ' + \
				'Link Utilization: ' + str(self.getLinkUtilization() * 100.0) + ' %}'

# Holds a subflow of a input_utils.Flow
class SubFlow:
	# Input must a input_utils.flow
	def __init__(self, flow = None, subflow = None):
		if flow is not None:
			# Reference to the parent flow
			self.flow = flow

			# Remaining route of the subflow (starting at curNode() and ending at self.flow.dst)
			self.rem_route = list(self.flow.route)

			# This subflow's size.
			self.size = self.flow.size
		elif subflow is not None:
			self.flow = subflow.flow

			self.rem_route = list(subflow.rem_route)

			self.size = subflow.size
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

	def weight(self):
		return self.flow.weight()

	def invweight(self):
		return self.flow.invweight()

	def flowID(self):
		return self.flow.id

	def getSize(self):
		return self.size

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
	subflows.sort(key = lambda x: (x.invweight(), x.flowID()))
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
		prev_invweight = subflows[0].invweight()
		for subflow in subflows:
			# Once we have passed all subflows of a certain invweight, add the number of packets from subflows with invweight less than or equal to the previous subflow's invweight as a possible alpha.
			if prev_invweight != subflow.invweight():
				alphas.add(this_alpha)

				prev_invweight = subflow.invweight()

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

# Finds the matching and alpha that maximizes sum of weights of the matching / (alpha + reconfig_delta)
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
#   alphas --> Set of all alpha values to try
#   num_nodes --> The number of nodes in the network
#   reconfig_delta --> the reconfiguraiton delta of the network
# Output:
#   best_objective_value --> The objective value of the best matching and alpha. (i.e. total weight of matching / (alpha + reconfig_delta))
#   best_matching_weight --> The total weight of the matching that maximizes the objective.
#   best_alpha --> The best alpha found
#   best_matching --> The best matching found. It is a set of (src, dst) edges.
def findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta):
	Profiler.start('findBestMatching')

	best_objective_value = None
	best_matching_weight = None
	best_alpha           = None
	best_matching        = None

	# Tries all alpha values in alphas and finds the maximum weighted matching in each
	for alpha in sorted(alphas):
		# Creates the graph for the given alpha
		graph = createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes)

		# Find the maximum weight matching of the graph using a 3rd party library
		Profiler.start('networkx.max_weight_matching')
		matching = nx.algorithms.matching.max_weight_matching(graph)
		Profiler.end('networkx.max_weight_matching')
		matching_weight = sum(graph[a][b]['weight'] for a, b in matching)

		matching = convertMatching(matching, num_nodes)

		# Track the graph that maximizes value(G) / (alpha + reconfig_delta)
		this_objective_value = matching_weight / (alpha + reconfig_delta)
		if best_objective_value is None or this_objective_value > best_objective_value:
			best_objective_value = this_objective_value
			best_matching_weight = matching_weight
			best_alpha           = alpha
			best_matching        = matching

	Profiler.end('findBestMatching')
	return best_objective_value, best_matching_weight, best_alpha, best_matching

# Given the subflows (grouped by their next hop) and a matching duration, computes the bipartite graph with weighted edges.
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (curNode(), nextNode()) and value of list of SubFlow.
#   alpha --> The matching duration given in # of packets.
# Ouput:
#   Returns a nx.Graph object that is a complete bipartite graph with edge weights. Note the edge (x, y) in the returned graph corresponds the edge (x, y - num_node) in the rest of the code.
def createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes):
	Profiler.start('createBipartiteGraph')

	# Creates the (complete) graph. Note that edge (i, j) is represented as edge (i, j + num_nodes) in the graph.
	graph = nx.complete_bipartite_graph(num_nodes, num_nodes)

	# Initializes all edges to weight 0.0
	for _, _, d in graph.edges(data = True):
		d['weight'] = 0.0

	# Compute the weight of each edge in G'
	for (i, j), subflows in subflows_by_next_hop.iteritems():
		this_weight = calculateTotalWeight(subflows, alpha)

		graph[i][j + num_nodes]['weight'] = this_weight

	Profiler.end('createBipartiteGraph')
	return graph

# Finds the weighted number of packets that can be sent of subflows in alpha time.
# Input:
#   subflows --> list of SubFlows sorted by increasing invweight() / decreasing weight()
#   alpha --> duration that data can be sent
# Return
#   Returns maximum weighted sum of packets that can be sent within alpha time.
def calculateTotalWeight(subflows, alpha):
	Profiler.start('calculateTotalWeight')

	unused_time = alpha
	weighted_sum = 0.0
	for subflow in subflows:
		if subflow.getSize() < unused_time:
			weighted_sum += subflow.getSize() * subflow.weight()
			unused_time  -= subflow.getSize()
		else:
			weighted_sum += unused_time * subflow.weight()
			unused_time  -= unused_time
			break

	Profiler.end('calculateTotalWeight')
	return weighted_sum

# Converts the networkx matching to our matching
# Input:
#   matching --> The output of nx.algorithms.matching.max_weight_matching, a set of (node_id, node_id)
#   num_nodes --> number of nodes in the network
# Output:
#   Returns the matching in our format. For each edge (x, y) in the inputted matching, the output will contain (min(x, y), max(x, y) - num_nodes)
def convertMatching(matching, num_nodes):
	Profiler.start('convertMatching')
	new_matching = set()
	for edge in matching:
		src = min(edge)
		dst = max(edge)

		new_matching.add((src, dst - num_nodes))

	Profiler.end('convertMatching')
	return new_matching

# Sends as many subflows as possible within the given time period
# Input:
#   subflows --> list of subflows to update
#   alpha --> total number of packets that can be sent
# Output:
#   total_packets_sent --> The total number of packets sent
#   remaing_alpha --> The number of packets that can still be sent.
#   finished_subflows --> List of subflows that have reached their destination.
#   new_subflows --> List of new subflows that were split from one the inputted subflows.
def updateSubFlows(subflows, alpha):
	Profiler.start('updateSubFlows')

	finished_subflows = []
	new_subflows = []

	total_packets_sent = 0

	for subflow in subflows:
		packets_sent, alpha, split_subflow, is_subflow_done = subflow.send(alpha)

		total_packets_sent += packets_sent

		if split_subflow is not None:
			new_subflows.append(split_subflow)

		if is_subflow_done:
			finished_subflows.append(subflow)

		if alpha <= 0:
			break

	remaing_alpha = alpha

	Profiler.end('updateSubFlows')
	return total_packets_sent, remaing_alpha, finished_subflows, new_subflows

# Computes the multi-hop schedule for the given flows
# Input:
#   flows --> dict with keys (src_node_id, dst_node_id) and values of input_utils.Flow
def computeSchedule(num_nodes, flows, window_size, reconfig_delta):
	Profiler.start('computeSchedule')

	# Initialzie subflows (same as the remaining traffic)
	current_subflows = [SubFlow(flow = flows[k]) for k in flows]
	completed_subflows = []

	# Initialize the schedule, which will hold the result
	schedule = Schedule()

	# Track time slots where packets are and aren't sent
	time_slots_used = 0
	time_slots_not_used = 0

	while schedule.totalDuration(reconfig_delta) < window_size:
		Profiler.start('computeSchedule-iteration')

		# Gets the subflows_by_next_hop information from remaining_flows
		# Keys are (curNode(), nextNode()) and values are a list of NextHopFlows
		Profiler.start('computeSchedule-subflows_by_next_hop')
		subflows_by_next_hop = defaultdict(list)
		for subflow in current_subflows:
			subflows_by_next_hop[(subflow.curNode(), subflow.nextNode())].append(subflow)
		Profiler.end('computeSchedule-subflows_by_next_hop')

		# Sorts each list in subflows_by_next_hop
		for k in subflows_by_next_hop:
			sortSubFlows(subflows_by_next_hop[k])

		# Get set of unique alphas to consider
		alphas = getUniqueAlphas(subflows_by_next_hop)

		# Truncate alphas that go over the duration
		alphas = filterAlphas(alphas, schedule.totalDuration(reconfig_delta), schedule.numMatchings(), window_size, reconfig_delta)

		# Track the best alpha and matching
		obective_value, matching_weight, alpha, matching = findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta)

		# Add best matching and its associated alpha to Schedule
		schedule.addMatching(matching, alpha, matching_weight = matching_weight)

		# Update remaining
		for edge in matching:
			packets_sent, unused_alpha, finished_subflows, new_subflows = updateSubFlows(subflows_by_next_hop[edge], alpha)
			
			time_slots_used += packets_sent
			time_slots_not_used += unused_alpha

			# Removes any subflows that have finished from current_subflows and places adds them to completed_subflows
			for fin_subflow in finished_subflows:
				current_subflows.remove(fin_subflow)
				completed_subflows.append(fin_subflow)

			# Adds any sublfows that were split to current_subflows
			for new_subflow in new_subflows:
				current_subflows.append(new_subflow)

		Profiler.end('computeSchedule-iteration')

	# Compute result metrics
	total_objective_value = schedule.getTotalMatchingWeight()

	packets_delivered = sum(subflow.getSize() for subflow in completed_subflows)
	packets_not_delivered = sum(subflow.getSize() for subflow in current_subflows)

	# Add in reconfiguration delay to unused time slots
	time_slots_not_used += reconfig_delta * num_nodes * schedule.numReconfigs()

	result_metric = ResultMetric(total_objective_value, packets_delivered, packets_not_delivered, window_size, time_slots_used, time_slots_not_used)

	Profiler.end('computeSchedule')
	return schedule, result_metric
