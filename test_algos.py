
# This file cntains simple unit tests for helper functions of algos.

import algos

from collections import defaultdict
import random

# Fakes a algos.SubFlow
class FakeSubFlow:
	def __init__(self, flow_id, invweight, size):
		self.flow_id = flow_id
		self.invweight_val = invweight
		self.size = size

	def weight(self):
		return 1.0 / float(invweight_val)

	def invweight(self):
		return self.invweight_val

	def flowID(self):
		return self.flow_id

	def getSize(self):
		return self.size

def testSortSubFlows(n_iters = 100, num_subflows = 1000, max_flow_id = 100000, max_invweight = 10):
	# Perform the specified number of tests
	for n_iter in range(n_iters):

		# Generate fake subflows
		subflows = [None] * num_subflows
		for i in range(num_subflows):
			subflows[i] = FakeSubFlow(random.randint(0, max_flow_id), random.randint(1, max_invweight), 0)

		# Run the fake data through the function being tested
		algos.sortSubFlows(subflows)

		# Check that the subflows are sorted
		# Should go from lowest invweight to highest invweight with flow_id as a tiebreaker
		for i in range(num_subflows - 1):
			low_sf  = subflows[i]
			high_sf = subflows[i + 1]

			# Checks the constraint
			if (low_sf.invweight() < high_sf.invweight()) or (low_sf.invweight() == high_sf.invweight() and low_sf.flowID() <= high_sf.flowID()):
				pass
			else:
				raise Exception('Sorting constraint failed: got (' + str(low_sf.invweight()) + ', ' + str(low_sf.flowID()) + ') followed by (' + str(high_sf.invweight())  + ', ' + str(high_sf.flowID()) + ')')
	
	print 'testSortSubFlow passed'

def testGetUniqueAlphas(n_iters = 100, num_subflow_groups = 100, max_subflows_per_invweight = 10, min_packets = 10, max_packets = 1000, max_invweight = 5):
	for n_iter in range(n_iters):
		# Generate fake subflows and calculate the expected output
		subflows = defaultdict(list)
		expected_alphas = set()
		for x in range(num_subflow_groups):
			sum_packets = 0

			# Generates packets starting with invweight 1 going up to max_invweight
			for invw in xrange(1, max_invweight + 1):

				# Randomly chooses the number of subflows to create
				num_subflows = random.randint(0, max_subflows_per_invweight)

				for _ in range(num_subflows):
					# Randomly chooses the size of the subflow
					this_size = random.randint(min_packets, max_packets)
					sum_packets += this_size

					this_subflow = FakeSubFlow(0, invw, this_size)
					subflows[x].append(this_subflow)
				
				# Once all packets of a certain invweight are created, we add the current value of sum_packets to the set of expected_alphas
				if sum_packets > 0:
					expected_alphas.add(sum_packets)

		# Call the function to calculate the unique alphas
		calculated_alphas = algos.getUniqueAlphas(subflows)

		# Compare expected_alphas and calculated_alphas
		if len(expected_alphas) != len(calculated_alphas) or \
				any(v not in calculated_alphas for v in expected_alphas) or \
				any(v not in expected_alphas for v in calculated_alphas):

			raise Exception('Unexpected result from getUnqiueAlphas: got (' + ', '.join(map(str, sorted(calculated_alphas))) + '); expected (' + ', '.join(map(str, sorted(expected_alphas))) + ')')

	print 'testGetUnqiueAlphas passed'

def testCreateBipartiteGraph(n_iters = 100, min_num_nodes = 50, max_num_nodes = 150, edge_prob = 1.0):
	for n_iter in range(n_iters):
		# Create fake data
		# Choose a random value for the number of nodes in the graph
		num_nodes = random.randint(min_num_nodes, max_num_nodes)

		# Build up the fake values for subflows
		fake_subflows = dict()
		fake_weights  = dict()
		x = 1
		for i in range(num_nodes):
			for j in range(num_nodes):
				# Randomly choose if this edge should be added
				rv = random.random()
				if i == j or rv > edge_prob:
					fake_weights[(i, j)]  = 0
				else:
					# Add the fake subflow. The value of the subflow isn't directly used by the funciton being tested.
					fake_subflows[(i, j)] = (i, j)

					# Define the weight for this fake subflow
					fake_weights[(i, j)]  = x
					x += 1

		# The value of alpha isn't directly used by the function being tested
		alpha = None

		# Fake function that returns the weight of the given subflows
		def fake_calc_weight_func(subflows, alpha):
			# Checks input
			if alpha is not None:
				raise Exception('Unexpected alpha value: ' + str(alpha))

			if subflows not in fake_weights:
				raise Exception('Unexpected subflows value: ' + str(subflows))

			# Returns the fake weight that was defined above
			return fake_weights[subflows]

		# Runs the function
		graph = algos.createBipartiteGraph(fake_subflows, alpha, num_nodes, calc_weight_func = fake_calc_weight_func)

		# Check the graph to make sure that the weights match what we defined.
		for x, y, d in graph.edges(data = True):
			i = x
			j = y - num_nodes
			this_weight = d['weight']

			if fake_weights[(i, j)] != this_weight:
				raise Exception('Unexpected weight for edge (' + str(i) + ', ' + str(j) + '): expected ' + str(fake_weights[(i, j)]) + '; got ' + str(this_weight))

	print 'testCreateBipartiteGraph passed'


# TODO create tests for the following functions
#   findBestMatching
#   createBipartiteGraph
#   calculateTotalWeight
#   convertMatching

if __name__ == '__main__':
	testSortSubFlows()
	testGetUniqueAlphas()
	testCreateBipartiteGraph()

