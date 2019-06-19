
# This file cntains simple unit tests for helper functions of algos.

import algos

from collections import defaultdict
import random

EPS = 10e-6

# Fakes a algos.SubFlow
class FakeSubFlow:
	def __init__(self, flow_id, invweight, size):
		self.flow_id = flow_id
		self.invweight_val = invweight
		self.size = size

	def weight(self):
		return 1.0 / float(self.invweight_val)

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

def testGetUniqueAlphas(n_iters = 100, num_subflow_groups = 100, max_subflows_per_invweight = 100, min_packets = 10, max_packets = 1000, max_invweight = 5):
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

def testCreateBipartiteGraph(n_iters = 100, min_num_nodes = 50, max_num_nodes = 150, flow_prob = 0.5):
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
				if i == j or rv > flow_prob:
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
		for i, j in fake_weights:
			x = i
			y = j + num_nodes

			if fake_weights[(i, j)] == 0 and graph.has_edge(x, y):
				raise Exception('Unexpected edge: (' + str(i) + ', ' + str(j) + ')' )

			if fake_weights[(i, j)] > 0 and not graph.has_edge(x, y):
				raise Exception('Expected edge but none found: (' + str(i) + ', ' + str(j) + ')')

			if fake_weights[(i, j)] > 0 and graph.has_edge(x, y) and fake_weights[(i, j)] != graph[x][y]['weight']:
				raise Exception('Unexpected weight for edge (' + str(i) + ', ' + str(j) + '): expected ' + str(fake_weights[(i, j)]) + '; got ' + str(graph[x][y]['weight']))

	print 'testCreateBipartiteGraph passed'

def testCalculateTotalWeight(n_iters = 100, n_alphas = 10, max_subflows_per_invweight = 100, min_packets = 10, max_packets = 1000, max_invweight = 5):
	for n_iter in range(n_iters):
		flow_id = 0

		num_packets_by_inv_weight = {invw: 0 for invw in xrange(1, max_invweight + 1)}
		total_packets = 0

		# Generates packets starting with invweight 1 going up to max_invweight
		subflows = []
		for invw in xrange(1, max_invweight + 1):
			# Randomly chooses the number of subflows to create
			num_subflows = random.randint(0, max_subflows_per_invweight)

			for _ in range(num_subflows):
				this_size = random.randint(min_packets, max_packets)
				num_packets_by_inv_weight[invw] += this_size
				total_packets += this_size

				this_subflow = FakeSubFlow(0, invw, this_size)
				subflows.append(this_subflow)

		cumulative_packets_by_inv_weight = {invw: sum(num_packets_by_inv_weight[i] for i in xrange(1, invw + 1)) for invw in xrange(1, max_invweight + 1)}

		for invw_test in xrange(1, max_invweight + 2):
			# Skips values with no packets
			if invw_test <= max_invweight and num_packets_by_inv_weight[invw_test] == 0:
				continue

			# Figure out the range of alpha such that the cutoff should be at subflows with an inverse weight of invw_test.
			if invw_test == 1:
				min_alpha = 1
			else:
				min_alpha = cumulative_packets_by_inv_weight[invw_test - 1] + 1

			if invw_test == max_invweight + 1:
				max_alpha = cumulative_packets_by_inv_weight[max_invweight] + 1000
			else:
				max_alpha = cumulative_packets_by_inv_weight[invw_test]

			# Check that the range of alphas is correct
			if invw_test <= max_invweight and min_alpha + num_packets_by_inv_weight[invw_test] - 1 != max_alpha:
					raise Exception('Invalid values for min and max alpha: min --> ' + str(min_alpha) + ', max --> ' + str(max_alpha))

			# Generates a random alpha
			alpha = random.randint(min_alpha, max_alpha)

			# Calculate the expected total weight for the chosen alpha value
			expected_total_weight = 0.0
			unaccounted_alpha = alpha

			# Adds in the weighted packet size for all subflows with an inverse weight strictly less than invw_test
			for i in xrange(1, invw_test):
				expected_total_weight += num_packets_by_inv_weight[i] * 1.0 / float(i)
				unaccounted_alpha -= num_packets_by_inv_weight[i]

			if unaccounted_alpha <= 0 or (invw_test <= max_invweight and unaccounted_alpha > num_packets_by_inv_weight[invw_test]):
				raise Exception('Invalid calculation of unnacounted_alpha')

			# Adds in the weighted packet size for subflows with an inverse weight of exactly invw_test that are not cutoff with this value of alpha
			if invw_test <= max_invweight:
				expected_total_weight += unaccounted_alpha * 1.0 / float(invw_test)

			calculated_total_weight = algos.calculateTotalWeight(subflows, alpha)

			if abs(expected_total_weight - calculated_total_weight) > EPS:
				raise Exception('Mismatch in total weight: expected ' + str(expected_total_weight) + '; got ' + str(calculated_total_weight))

	print 'testCalculateTotalWeight passed'

# TODO create tests for the following functions
#   findBestMatching (this function doesn't do much ouside of calling subfunctions)
#   convertMatching (pretty straightforward)
#   updateSubFlows
#   SubFlow.send

if __name__ == '__main__':
	testSortSubFlows()
	testGetUniqueAlphas()
	testCreateBipartiteGraph()
	testCalculateTotalWeight()
