
from collections import defaultdict

# Holds a multi-hop schedule
class Schedule:
	def __init__(self):
		self.matchings = []
		self.durations = []

	def totalDuration(self, reconfig_delta):
		if len(self.durations) <= 0:
			return 0
		else:
			return sum(self.durations) + reconfig_delta * (len(self.durations) - 1)

# Holds a subflow of a input_utils.Flow
class SubFlow:
	# Input must a input_utils.flow
	def __init__(self, flow):
		# Reference to the parent flow
		self.flow = flow

		# Current Start of the subflow
		self.cur_node = self.flow.src

		# Remaining route of the subflow (starting at cur_node and ending at self.flow.dst)
		self.rem_route = list(self.flow.route)

		if self.cur_node != self.rem_route[0]:
			raise Exception('Current node is not the first node in the remaining_route')

		# The next node this flow will visit
		self.next_node = self.rem_route[1]

		# This subflow's size.
		self.size = self.flow.size


	def weight(self):
		return self.flow.weight()

	def invweight(self):
		return self.flow.invweight()

	def flowID(self):
		return self.flow.id

	def getSize(self):
		return self.size

# Sorts subflows in place
def sortSubFlows(subflows):
	# TODO sort by subflow id as well incase there are two subflows of the same size from the same flow

	# Sorts flows by shortest route to longest route. Uses flow.id as a tiebreaker in order to keep things deterministic.
	subflows.sort(key = lambda x: (x.invweight(), x.flowID()))

# Given the next hop traffic, find the set of alphas to consider
# Input:
#   subflows_by_next_hop --> All subflows grouped by their next hop. SubFlows in each group are sorted from shortest to longest with flow.id as a tiebreak. Must be a dict with keys of (cur_node, next_node) and value of list of SubFlow.
# Output:
#   Returns a set of alphas to be considered.
def getUniqueAlphas(subflows_by_next_hop):
	alphas = set()

	this_alpha = 0
	for _, subflows in subflows_by_next_hop.iteritems():
		# Ignores length zero lists
		if len(subflows) == 0:
			continue

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

	return alphas

# Computes the multi-hop schedule for the given flows
# Input:
#   flows --> dict with keys (src_node_id, dst_node_id) and values of input_utils.Flow
def computeSchedule(num_nodes, flows, window_size, reconfig_delta):
	# Initialzie subflows (same as the remaining traffic)
	# Key is (cur_node, flow.dst) and values are a list of SubFlows
	subflows_by_src_dst = {k:[SubFlow(flows[k])] for k in flows}

	# Initialize the schedule, which will hold the result
	schedule = Schedule()

	while schedule.totalDuration(reconfig_delta) < window_size:
		# Gets the subflows_by_next_hop information from remaining_flows
		# Keys are (cur_node, next_node) and values are a list of NextHopFlows
		subflows_by_next_hop = defaultdict(list)
		for _, subflows in subflows_by_src_dst.iteritems():
			for subflow in subflows:
				subflows_by_next_hop[(subflow.cur_node, subflow.next_node)].append(subflow)

		# Sorts each list in subflows_by_next_hop
		for k in subflows_by_next_hop:
			sortSubFlows(subflows_by_next_hop[k])

		# Get set of unique alphas to consider
		alphas = getUniqueAlphas(subflows_by_next_hop)

		for alpha in alphas:
			pass
			# Compute the weight of each edge in G'

			# Find the maximum weight matching of G'

			# Track the G' that maximizes value(G') / (alpha + reconfig_delta)

		# Add best G' and its associated alpha to Schedule

		# Update remaining

	return schedule
