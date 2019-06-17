import numpy as np
import random

class Flow:
	# id --> ID of the flow. Must be an int greater than or equal to 0
	# src, dst --> ID of the source and destination nodes. Must be between 0 and N-1 (where N is the number of nodes)
	# size --> size of the flow in numbers of packets. Must be an int greater than 0.
	# route --> The route the flow must take. Must be a list of ints, where the ints are all between 0 and N-1, and there are no repeats
	def __init__(self, id, src, dst, size, route, num_nodes):
		self.id  = id
		self.src = src
		self.dst = dst

		self.size  = size
		self.route = route

		self.check(num_nodes)

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

		# TODO: some bug here ?
		# if any(not is_node_id(v) for v in self.route) or \
		# 		any(self.route.count(v) != 1 for v in self.route) or \
		# 		self.route[0] != self.src or self.route[-1] != self.dst or \
		# 		len(self.route) <= 1:
		# 	raise Exception('self.route has invalid value: ' + str(self.route))
			

	# Returns the weight of this flow. Returns a float.
	def weight(self):
		return 1.0 / float(len(self.route) - 1)

	# Returns the inverse weight of this flow. Guarenteed to return an int.
	def invweight(self):
		return len(self.route) - 1

	def __str__(self):
		return '(Flow ' + str(self.id) + ', src:' + str(self.src) + ' --> dst:' + str(self.dst) + ', # of Packets: ' + str(self.size) + ', route: [' + ', '.join(map(str, self.route)) + '])'


class Traffic:
    '''Traffic matrix
    '''
    def __init__(self, num_nodes=64, max_hop=4, random_seed=1):
        '''
        Args:
            num_nodes (int): |V|
        '''
        self.num_nodes = num_nodes    # the nodes are [0, 1, ... , num_nodes-1]
        self.max_hop   = max_hop      # max hop is typically 4
        self.matrix    = []           # np.ndrray, n=2
        self.flows     = {}           # list<Flow>
        self.random_seed = random_seed


    def random_route(self, source, dest):
        '''Random route
        Args:
            source (int)
            dest   (int)
        Return:
            (list<int>)
        '''
        if source < 0 or source >= self.num_nodes or dest < 0 or dest >= self.num_nodes or source == dest:
            raise Exception('Wrong paramaters! {}, {}'.format(source, dest))
        route = [source]
        all_nodes = list(range(self.num_nodes))
        all_nodes.remove(source)
        all_nodes.remove(dest)
        middle_hops = random.randint(1, self.max_hop) - 1
        middle_nodes = np.random.choice(all_nodes, middle_hops)
        route.extend(list(middle_nodes))
        route.append(dest)
        return route


    def sigmetrics(self):
        pass


    def microsoft(self, cluster):
        ''' Website: https://www.microsoft.com/en-us/research/project/projector-agile-reconfigurable-data-center-interconnect/
        Args:
            cluster (int): options -- 1, 2, 3. Cluster 1 has ~100 nodes, 2 has ~450 nodes, 3 has ~1500 nodes
        '''
        random.seed(self.random_seed)                                       # replicable
        np.random.seed(self.random_seed)
        ID = 0
        filename = 'microsoft/cluster-{}.txt'.format(cluster)
        self.matrix = np.loadtxt(filename, delimiter=',')
        num_node = len(self.matrix[0])                       # num_node is typically larger than self.max_nodes, so randomly truncate a subset
        if self.num_nodes > num_node:
            raise Exception('self.num_nodes ({}) is larger than nodes in microsoft cluster-{} ({})'.format(self.num_nodes, cluster, num_node))
        subset_nodes = random.sample(range(num_node), self.num_nodes)
        self.matrix = self.matrix[np.ix_(subset_nodes, subset_nodes)]
        self.flows  = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                size = self.matrix[i][j]
                if size == 0.0 or i == j:
                    continue
                rand = random.uniform(0.99, 1.01)   # add some randomness
                size = int(size*rand)
                route = self.random_route(i, j)
                self.flows[(i, j)] = Flow(ID, i, j, size, route, self.num_nodes)
                ID += 1
            print()
        print('Init succuess! Microsoft cluster {}.'.format(cluster), self)
        return self.flows


    def facebook(self):
        pass

    def university(self):
        pass

    def __str__(self):
        return '# of nodes = {}, max hop = {}, # of flows = {}'.format(self.num_nodes, self.max_hop, len(self.flows))


if __name__ == '__main__':
    t = Traffic(num_nodes=64, max_hop=4, random_seed=1)
    flow = t.microsoft(1)
    #t.microsoft(2)
    #t.microsoft(3)
    #t.microsoft(4)



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


