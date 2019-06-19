
import input_utils
from profiler import Profiler

# Creates a copies of flows, but all routes only have one hop
# Input:
#   flows --> dictionary of flows (indexed by (src node, dst node)) to be reduced
# Output:
#   single_hop_flows --> Essentially a deep copy of flows, but all routes are reduced to one hop.
def reduceToOneHop(flows):
	Profiler.start('reduceToOneHop')
	single_hop_flows = {}

	for k, flow in flows.iteritems():
		single_hop_flows[k] = input_utils.Flow(flow.id, flow.src, flow.dst, flow.size, [flow.src, flow.dst])

	Profiler.end('reduceToOneHop')

	return single_hop_flows
