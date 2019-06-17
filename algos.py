
class Schedule:
	def __init__(self):
		self.matchings = []
		self.durations = []

	def totalDuration(self, reconfig_delta):
		if len(self.durations) <= 0:
			return 0
		else:
			return sum(self.durations) + reconfig_delta * (len(self.durations) - 1)

def getUniqueAlphas(trem):
	return set()

# Computes the multi-hop schedule for the given flows
# Input:
#   flows --> dict with keys (src_node_id, dst_node_id) and values of input_utils.Flow
def computeSchedule(num_nodes, flows, window_size, reconfig_delta):
	# Initialzie Trem and other values
	remaining = None

	# Initialize the schedule, which will hold the result
	schedule = Schedule()

	while schedule.totalDuration(reconfig_delta) < window_size:
		next_hop = None

		# Get set of unique alphas to consider
		alphas = getUniqueAlphas(next_hop)

		for alpha in alphas:
			pass
			# Compute the weight of each edge in G'

			# Find the maximum weight matching of G'

			# Track the G' that maximizes value(G') / (alpha + reconfig_delta)

		# Add best G' and its associated alpha to Schedule

		# Update remaining

	return schedule


