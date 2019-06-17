
import random

class Flow:
	# id --> ID of the flow. Must be an int greater than or equal to 0
	# src, dst --> ID of the source and destination nodes. Must be between 0 and N-1 (where N is the number of nodes)
	# size --> size of the flow in numbers of packets. Must be an int greater than 0.
	# route --> The route the flow must take. Must be a list of ints, where the ints are all between 0 and N-1, and there are no repeats
	def __init__(self, id, src, dst, size, route):
		self.id  = id
		self.src = src
		self.dst = dst

		self.size  = size
		self.route = route

		self.check()

	# Checks that the values of the flow match the expections
	def check(self, num_nodes = None):
		if not isinstance(self.id, int) or self.id < 0:
			raise Exception('self.id has invalid value: ' + str(self.id))

		# Lambda function that checks if a value is a valid node_id
		is_node_id = lambda v: isinstance(v, int) and v >= 0 and (num_nodes is None or v < num_nodes)

		if not is_node_id(self.src):
			raise Exception('self.src has invalid value: ' + str(self.src))

		if not is_node_id(self.dst):
			raise Exception('self.dst has invalid value: ' + str(self.dst))

		if self.src == self.dst:
			raise Exception('self.src and self.dst must be different: ' + str(self.src) + ', ' + str(self.dst))

		if not isinstance(self.size, int) or self.size <= 0:
			raise Exception('self.size has invalid value: ' + str(self.size))

		if any(not is_node_id(v) for v in self.route) or \
				any(self.route.count(v) != 1 for v in self.route) or \
				self.route[0] != self.src or self.route[-1] != self.dst or \
				len(self.route) <= 1:
			raise Exception('self.route has invalid value: ' + str(self.route))

	# Returns the weight of this flow. Returns a float.
	def weight(self):
		return 1.0 / float(len(self.route) - 1)

	# Returns the inverse weight of this flow. Guarenteed to return an int.
	def invweight(self):
		return len(self.route) - 1

	def __str__(self):
		return '(Flow ' + str(self.id) + ', src:' + str(self.src) + ' --> dst:' + str(self.dst) + ', # of Packets: ' + str(self.size) + ', route: [' + ', '.join(map(str, self.route)) + '])'

# Generates a random flow size between min_size and max_size
def simpleFlowSizeGenerator(min_size = 10, max_size = 100):
	return random.randint(min_size, max_size)

# Generates synthetic data
# Input:
#   num_nodes --> Number of nodes in the network. Generates 1 flow for each pair of unique nodes. Must be an int greater than zero.
#   max_route_length --> Maximum length of a route. Generates routes between 1 and max_route_length. Must be an int greater than zero and less than num_nodes.
#   flow_size_generator --> A zero arg function that returns a flow size
# Output:
#   flows --> a dictionary with keys of (src_node_id, dst_node_id) and value of Flow.
def generateTestFlows(num_nodes, max_route_length, flow_size_generator = simpleFlowSizeGenerator):
	flows = {}

	next_flow_id = 0
	for i in range(num_nodes):
		for j in range(num_nodes):
			# Skip cases where src_node == dst_node
			if i == j:
				continue

			# Generate data for this flow
			this_size = flow_size_generator()

			this_route_length = random.randint(1, max_route_length)
			this_route = [i, j]
			for x in range(this_route_length - 1):
				# Note: this will be inefficient if max_route_length is close to num_nodes
				next_node = random.randint(0, num_nodes - 1)
				while next_node in this_route:
					next_node = random.randint(0, num_nodes - 1)

				this_route.insert(x + 1, next_node)

			flows[(i, j)] = Flow(next_flow_id, i, j, this_size, this_route)

			next_flow_id += 1

	return flows
