
from profiler import Profiler

from collections import defaultdict
import networkx as nx
import numpy as np
import scipy.optimize as scipy_opt

# MAX_WEIGHT_MATCHING_LIBRARY = 'networkx'
MAX_WEIGHT_MATCHING_LIBRARY = 'scipy'

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
	next_subflow_id = 0

	# Input must a input_utils.flow
	def __init__(self, flow = None, subflow = None, override_route = None, override_subflow_id = None, override_weight = None):
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
			else:
				self.rem_route = list(override_route)

			# This subflow's size.
			self.size = self.flow.size

			# If this is not none, this value overrides self.flow.weight() and self.flow.invweight()
			self.override_weight = override_weight
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
			else:
				self.rem_route = list(override_route)

			self.size = subflow.size

			# If this is not none, this value overrides self.flow.weight() and self.flow.invweight()
			self.override_weight = override_weight
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

	def destNode(self):
		if len(self.rem_route) < 2:
			raise Exception('Subflow doesn\'t have a dest node')

		return self.rem_route[-1]

	def weight(self):
		if self.override_weight is None:
			return self.flow.weight()
		else:
			return self.override_weight

	def invweight(self):
		if self.override_weight is None:
			return self.flow.invweight()
		else:
			return self.override_weight

	def flowID(self):
		return self.flow.id

	def subflowID(self):
		return self.subflow_id

	def getSize(self):
		return self.size

	def remainingRouteLength(self):
		return len(self.rem_route) - 1

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
def findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	Profiler.start('findBestMatching')

	best_objective_value = None
	best_matching_weight = None
	best_alpha           = None
	best_matching        = None

	# Tries all alpha values in alphas and finds the maximum weighted matching in each
	for alpha in sorted(alphas):
		# Creates the graph for the given alpha
		graph = createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes, max_weight_matching_library = max_weight_matching_library)

		# Find the maximum weight matching of the graph using a 3rd party library
		if max_weight_matching_library is 'networkx':
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

		matching = convertMatching(graph, matching, num_nodes, max_weight_matching_library = max_weight_matching_library)

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
#   num_nodes --> Number of nodes in the network.
#   calc_weight_func --> Function that returns the weighted number of packets that can be sent by a set of subflows in a given duration. Must be a function with two arguments (a list of SubFlows, duration).
# Ouput:
#   Returns a nx.Graph object that is a complete bipartite graph with edge weights. Note the edge (x, y) in the returned graph corresponds the edge (x, y - num_node) in the rest of the code.
def createBipartiteGraph(subflows_by_next_hop, alpha, num_nodes, calc_weight_func = None, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	Profiler.start('createBipartiteGraph')

	# If no calc_weight_func given, use default function
	if calc_weight_func is None:
		calc_weight_func = calculateTotalWeight

	# Creates the graph
	if max_weight_matching_library is 'networkx':
		graph = nx.Graph()

		# Creates nodes for each input port. The input port of i is node i in the graph.
		graph.add_nodes_from(xrange(        0,     num_nodes), bipartite = 0)

		# Creates nodes for each output port. The output port of i is node i + num_nodes in the graph.
		graph.add_nodes_from(xrange(num_nodes, 2 * num_nodes), bipartite = 1)

		# Creates the weighted edges in bipartite graph
		graph.add_weighted_edges_from([(i, j + num_nodes, calc_weight_func(subflows, alpha)) for (i, j), subflows in subflows_by_next_hop.iteritems()])

	elif max_weight_matching_library is 'scipy':
		graph = np.zeros((num_nodes, num_nodes))

		for (i, j), subflows in subflows_by_next_hop.iteritems():
			graph[i][j] = calc_weight_func(subflows, alpha)

	else:
		raise Exception('Invalid max_weight_matching_library: ' + str(max_weight_matching_library))

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
def convertMatching(graph, matching, num_nodes, max_weight_matching_library = MAX_WEIGHT_MATCHING_LIBRARY):
	Profiler.start('convertMatching')
	if max_weight_matching_library is 'networkx':
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
#   return_completed_flow_ids --> If given, returns the flow_ids of completed subflows
#   precomputed_schedule --> If given, uses the given schedule instead of computed maximum weight matchings
# Output:
#   schedule --> The computed schedule, containing the matchings and durations
#   result_metric --> Holds the various metrics to judge the algorithm
#   (Optional) completed_flow_ids --> List of flow_ids of subflows that reach their destination
def computeSchedule(num_nodes, flows, window_size, reconfig_delta, return_completed_flow_ids = False, precomputed_schedule = None, consider_all_routes = False, backtrack = False, verbose = False):
	Profiler.start('computeSchedule')
	# TODO Check that the set of flags are consistent

	# Initialzie subflows (same as the remaining traffic)
	if consider_all_routes:
		starting_subflows = [SubFlow(flow = flow, override_route = route) for _, flow in flows.iteritems() for route in flow.all_routes]

		track_updated_subflows = True
	else:
		starting_subflows = [SubFlow(flow = flow) for _, flow in flows.iteritems()]

		track_updated_subflows = False

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

	while schedule.totalDuration(reconfig_delta) < window_size:
		if verbose:
			print 'Starting iteration with', window_size - schedule.totalDuration(reconfig_delta), 'time left'

		Profiler.start('computeSchedule-iteration')

		if backtrack:
			Profiler.start('computeSchedule-backtrack')
			# Check if a subflow can be "backtracking", if so adds the backtrack subflow
			for _, subflows in current_subflows_by_subflow_id.iteritems():
				# Can only backtrack on single subflows with two or more hops left.
				if len(subflows) == 1 and subflows[0].remainingRouteLength() >= 2:
					this_subflow = subflows[0]

					# The backtrack route is just a direct route to the destination
					backtrack_route = [this_subflow.curNode(), this_subflow.destNode()]

					# Finds the weight of the backtrack subflow. It is equal to the sum of the weights of the remaining hops in the subflow
					backtrack_weight = this_subflow.remainingRouteLength() * this_subflow.weight()

					# Creates the backtrack_subflow and adds it to the system.
					backtrack_subflow = SubFlow(subflow = this_subflow, override_route = backtrack_route, override_subflow_id = this_subflow.subflowID(), override_weight = backtrack_weight)

			Profiler.end('computeSchedule-backtrack')

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
			# Get set of unique alphas to consider
			alphas = getUniqueAlphas(subflows_by_next_hop)

			# Truncate alphas that go over the duration
			alphas = filterAlphas(alphas, schedule.totalDuration(reconfig_delta), schedule.numMatchings(), window_size, reconfig_delta)

			# Track the best alpha and matching
			objective_value, matching_weight, alpha, matching = findBestMatching(subflows_by_next_hop, alphas, num_nodes, reconfig_delta)

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
		for edge in matching:
			packets_sent, unused_alpha, updated_subflows, finished_subflows, new_subflows = updateSubFlows(subflows_by_next_hop[edge], alpha, track_updated_subflows = track_updated_subflows)
			
			time_slots_used += packets_sent
			time_slots_not_used += unused_alpha

			# Removes subflows of other routes once one of the subflows was chosen.
			if consider_all_routes:
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

					# Remove all other related subflows, by simply replacing the list with a list with just this subflow
					current_subflows_by_subflow_id[updated_subflow.subflowID()] = [updated_subflow]

			# Removes any subflows that have finished from current_subflows_by_subflow_id and places adds them to completed_subflows
			for fin_subflow in finished_subflows:
				current_subflows_by_subflow_id[fin_subflow.subflowID()].remove(fin_subflow)

				if len(current_subflows_by_subflow_id[fin_subflow.subflowID()]) == 0:
					del current_subflows_by_subflow_id[fin_subflow.subflowID()]

				completed_subflows.append(fin_subflow)

			# Adds any sublfows that were split to current_subflows_by_subflow_id
			for new_subflow in new_subflows:
				current_subflows_by_subflow_id[new_subflow.subflowID()].append(new_subflow)
			

		Profiler.end('computeSchedule-updateAllSubflows')
		Profiler.end('computeSchedule-iteration')

	# Compute result metrics
	total_objective_value = schedule.getTotalMatchingWeight()

	packets_delivered = sum(subflow.getSize() for subflow in completed_subflows)

	# All subflows with the same subflow ID represent one undelivered subflow. All of these subflows should have the same size, and there should be no empty lists.
	for _, subflows in current_subflows_by_subflow_id.iteritems():
		if len(subflows) == 0:
			raise Exception('Still have a subflow list with no subflows')

		if any(subflow.getSize() != subflows[0].getSize() for subflow in subflows):
			raise Exception('Some subflows with the same ID don\'t have the same size')

	packets_not_delivered = sum(subflows[0].getSize() for _, subflows in current_subflows_by_subflow_id.iteritems())

	# Add in reconfiguration delay to unused time slots
	time_slots_not_used += reconfig_delta * num_nodes * schedule.numReconfigs()

	result_metric = ResultMetric(total_objective_value, packets_delivered, packets_not_delivered, window_size, time_slots_used, time_slots_not_used)

	rvs = [schedule, result_metric]

	# If requested, returns the set of flow_ids that were completed
	if return_completed_flow_ids:
		finished_flow_ids = [subflow.flowID() for subflow in completed_subflows]
		rvs.append(finished_flow_ids)

	Profiler.end('computeSchedule')
	return tuple(rvs)
