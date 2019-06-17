
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

# tests algos.sortSubFlows
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

def testGetUniqueAlphas(n_iters = 100, num_subflow_groups = 1, max_subflows_per_invweight = 5, min_packets = 10, max_packets = 20, max_invweight = 5):
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

if __name__ == '__main__':
	random.seed(10)

	testSortSubFlows()
	testGetUniqueAlphas()

