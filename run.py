
import algos
import input_utils
from input_utils import Traffic

if __name__ == '__main__':
	# Get command line args
	num_nodes = 10
	max_route_length = 3
	window_size = 1000
	reconfig_delta = 10

	# Get input
	# flows = input_utils.generateTestFlows(num_nodes, max_route_length)
	traffic = Traffic(num_nodes=num_nodes, max_hop=max_route_length, window_size=window_size, random_seed=0)
	
	flows = traffic.microsoft(1)  # cluster 1 is somewhat dense
	for k in flows:
		print(flows[k])

	flows = traffic.microsoft(2)  # cluster 2 is between 1 and 3
	for k in flows:
		print(flows[k])

	flows = traffic.microsoft(3)  # cluster 3 is sparse (many zeros)
	for k in flows:
		print(flows[k])

	flows = traffic.sigmetrics()
	# flows = traffic.facebook( ... )
	# flows = traffic.university( ... )

	# Run test
	#algos.computeSchedule(num_nodes, flows, window_size, reconfig_delta)

	# Output result
