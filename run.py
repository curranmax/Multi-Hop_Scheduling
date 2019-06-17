
import input_utils

if __name__ == '__main__':
	# Get command line args
	num_nodes = 10
	max_route_length = 3
	window_size = 1000
	reconfig_delta = 10

	# Get input
	flows = input_utils.generateTestFlows(num_nodes, max_route_length)

	for k in flows:
		print flows[k]

	# Run test
	computeSchedule(num_nodes, flows, window_size, reconfig_delta)

	# Output result
