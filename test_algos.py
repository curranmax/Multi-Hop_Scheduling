
# This file cntains simple unit tests for helper functions of algos.

import algos
import random

# Fakes a algos.SubFlow
class FakeSubFlow:
	def __init__(self, flow_id, invweight):
		self.flow_id = flow_id
		self.invweight_val = invweight

	def weight(self):
		return 1.0 / float(invweight_val)

	def invweight(self):
		return self.invweight_val

	def flowID(self):
		return self.flow_id

# tests algos.sortSubFlows
def testSortSubFlows(n_iters = 100, num_subflows = 1000, max_flow_id = 100000, max_invweight = 10):
	# Perform the specified number of tests
	for n_iter in range(n_iters):

		# Generate fake subflows
		subflows = [None] * num_subflows
		for i in range(num_subflows):
			subflows[i] = FakeSubFlow(random.randint(0, max_flow_id), random.randint(1, max_invweight))

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

if __name__ == '__main__':
	testSortSubFlows()

