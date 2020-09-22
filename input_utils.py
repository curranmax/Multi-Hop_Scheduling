import numpy as np
import random
from scipy.stats import norm

class Flow:
	# id --> ID of the flow. Must be an int greater than or equal to 0
	# src, dst --> ID of the source and destination nodes. Must be between 0 and N-1 (where N is the number of nodes)
	# size --> size of the flow in numbers of packets. Must be an int greater than 0.
	# route --> The route the flow must take. Must be a list of ints, where the ints are all between 0 and N-1, and there are no repeats
	def __init__(self, id, src, dst, size, route, all_routes, num_nodes = None, do_check = True):
		self.id  = id
		self.src = src
		self.dst = dst

		self.size       = size
		self.all_routes = all_routes
		self.route      = route

		if do_check:
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

		if not (isinstance(self.size, int) or isinstance(self.size, np.int64)) or self.size <= 0:
			raise Exception('self.size has invalid value: ' + str(self.size))

		# Checks self.route
		if any(not is_node_id(v) for v in self.route) or \
				any(self.route.count(v) != 1 for v in self.route) or \
				self.route[0] != self.src or self.route[-1] != self.dst or \
				len(self.route) <= 1:
			raise Exception('self.route has invalid value: ' + str(self.route))

		# Checks all routes in self.all_routes are valid routes
		for route in self.all_routes:
			if any(not is_node_id(v) for v in route) or \
					any(route.count(v) != 1 for v in route) or \
					route[0] != self.src or route[-1] != self.dst or \
					len(route) <= 1:
				raise Exception('A route in self.all_routes has invalid value: ' + str(self.route))

		# Checks that no two routes in self.all_routes share the same first hop
		for i in range(len(self.all_routes)):
			for j in range(len(self.all_routes)):
				if i < j and self.all_routes[i][0] == self.all_routes[j][0] and self.all_routes[i][1] == self.all_routes[j][1]:
					raise Exception('Two routes share the same first hop: route 1: ' + str(self.all_routes[i]) + ', and route 2: ' + str(self.all_routes[j]))

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
    def __init__(self, num_nodes = 64, max_hop = 4, window_size = 1000, num_routes = 1, random_seed = 1, min_route_length = 1, debug = False):
        '''
        Args:
            num_nodes (int): |V|
        '''
        self.num_nodes   = num_nodes    # the nodes are [0, 1, ... , num_nodes-1]
        self.max_hop     = max_hop      # max hop is typically 4
        self.matrix      = []           # np.ndrray, n=2
        self.flows       = {}           # {(int, int) -> Flow}
        self.window_size = window_size  # for a traffic matrix, sum of row and sum of column should be bounded by window size
        self.random_seed = random_seed

        self.min_route_length = min_route_length

        self.num_routes = num_routes


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
        middle_nodes = random.sample(all_nodes, middle_hops)
        route.extend(list(middle_nodes))
        route.append(dest)
        return route


    def check_permutation(self, permutation):
        '''Assume that node i do not send traffic to node i itself
        Args:
            permutation (np.array)
        Return:
            (bool)
        '''
        for i, p in enumerate(permutation):
            if i == p:
                return False
        return True


    def random_permutation(self):
        '''Return a random permutation matrix
        '''
        index = list(range(self.num_nodes))
        random.shuffle(index)
        while self.check_permutation(index) == False:
            random.shuffle(index)
        permutation_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        for i in range(self.num_nodes):
            permutation_matrix[i][index[i]] = 1
        return permutation_matrix


    def bound_traffic(self, multiply=1):
        '''Bound the traffic so that the sum of column and row is smaller than window size. Sigmetrics'16 Hybrid section 2.2
        '''
        colum_sum = np.sum(self.matrix, 0)
        row_sum   = np.sum(self.matrix, 1)
        max1 = max(max(colum_sum), max(row_sum))
        ratio = max1/1.
        if ratio > 1:
            self.matrix /= ratio               # bounded to 1

        self.matrix *= self.window_size        # scale to window size
        self.matrix *= multiply                # if multiply = 2, then it is doubling the traffic. It will lead to some column/row summation larger than window_size
        self.matrix = np.array(self.matrix, dtype=int)


    def sigmetrics(self, c_l=0.7, n_l=4, c_s=0.3, n_s=12):
        '''
        Args:
            c_l: capacity of summation of large flows
            n_l: number of large flows
            c_s: capacity of summation of small flows
            n_s: number of small flows
        Return:
            {(int, int) -> Flow}
        '''
        random.seed(self.random_seed)                                       # replicable
        np.random.seed(self.random_seed)
        self.matrix = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(n_l):                  # large flows
            permutation = self.random_permutation()
            self.matrix += c_l/n_l * permutation
        
        for _ in range(n_s):                  # small flows
            permutation = self.random_permutation()
            self.matrix += c_s/n_s * permutation

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                if self.matrix[i][j] != 0:    # add noise to the non-zero entries
                    self.matrix[i][j] += np.random.normal(0, 0.003)
                    self.matrix[i][j] = 0 if self.matrix[i][j] < 0 else self.matrix[i][j]
        
        self.bound_traffic()

        ID = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                size = self.matrix[i][j]
                if size == 0 or i == j:
                    continue
                # route = self.random_route(i, j)

                all_routes = generateRandomRoutes(i, j, self.num_nodes, self.max_hop, self.num_routes, min_route_length = self.min_route_length)

                self.flows[(i, j)] = Flow(ID, i, j, size, all_routes[0], all_routes, self.num_nodes)
                ID += 1
        # print('\nInit succuess! Sigmetrics c-l={}, n-l={}, c-s={}, n-s={}'.format(c_l, n_l, c_s, n_s))
        return self.flows


    def microsoft(self, cluster, multiply=1):
        ''' Website: https://www.microsoft.com/en-us/research/project/projector-agile-reconfigurable-data-center-interconnect/
        Args:
            cluster (int): options -- 1, 2, 3. Cluster 1 has ~100 nodes, 2 has ~450 nodes, 3 has ~1500 nodes
            multiply (int): it multiplies the entire traffic matrix
        Return:
            {(int, int) -> Flow}
        '''
        random.seed(self.random_seed)                                       # replicable
        np.random.seed(self.random_seed)
        self.flows  = {}
        filename = 'microsoft/cluster-{}.txt'.format(cluster)
        self.matrix = np.loadtxt(filename, delimiter=',')
        num_node = len(self.matrix[0])                       # num_node is typically larger than self.max_nodes, so randomly truncate a subset
        if self.num_nodes > num_node:
            raise Exception('self.num_nodes ({}) is larger than nodes in microsoft cluster-{} ({})'.format(self.num_nodes, cluster, num_node))
        subset_nodes = random.sample(range(num_node), self.num_nodes)
        self.matrix = self.matrix[np.ix_(subset_nodes, subset_nodes)]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                size = self.matrix[i][j]
                if size == 0.0 or i == j:
                    continue
                if size > self.window_size:
                    size = self.window_size
                rand = random.uniform(0.99, 1.01)   # add some randomness
                size = int(size*rand)
                self.matrix[i][j] = size

        self.bound_traffic(multiply)

        ID = 0
        self.flows  = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                size = self.matrix[i][j]
                if size == 0.0 or i == j:
                    continue
                # route = self.random_route(i, j)

                all_routes = generateRandomRoutes(i, j, self.num_nodes, self.max_hop, self.num_routes)

                self.flows[(i, j)] = Flow(ID, i, j, size, all_routes[0], all_routes, self.num_nodes)
                ID += 1
        # print('\nInit succuess! Microsoft cluster {}.'.format(cluster))
        return self.flows


    def facebook(self, cluster='A', multiply=1):
        '''
        Args:
            cluster (str): options -- 'A', 'C'. A is database cluster, C is hadoop cluster
            multiply (int): it multiplies the entire traffic matrix
        Return:
            {(int, int) -> Flow}
        '''
        random.seed(self.random_seed)             # reproducable
        np.random.seed(self.random_seed)
        filename = 'facebook/facebook_cluster_{}.demand_matrix'.format(cluster)
        ID = 0       # give each rack an ID
        map_id = {}  # maps a hash number to ID
        with open(filename, 'r') as f:            # pass 1: count how many racks, give each rack an ID
            for line in f:
                line = line.replace('\n', '')
                if line == 'Pod_Demand_Matrix':
                    break
                line = line.split(' ')
                if len(line) != 3:                # must be three elements: src dest size
                    continue
                src = line[0]
                if map_id.get(src) == None:
                    map_id[src] = ID
                    ID += 1
            # print 'number of racks', len(map_id)

        self.matrix = np.zeros((len(map_id), len(map_id)))

        with open(filename, 'r') as f:            # pass 2: initialize the traffic matrix
            for line in f:
                line = line.replace('\n', '')
                if line == 'Pod_Demand_Matrix':
                    break
                line = line.split(' ')
                if len(line) != 3:                # must be three elements: src dest size
                    continue
                src, dest, size = line[0], line[1], line[2]
                if src == dest:                   # assume a rack don't send traffic to itself
                    continue
                src_id  = map_id.get(src)
                dest_id = map_id.get(dest)
                self.matrix[src_id][dest_id] = size
        
        num_node = len(self.matrix[0])            # num_node (hundreds) is typically larger than self.max_nodes(no more than a hundred)
        if self.num_nodes > num_node:
            raise Exception('self.num_nodes ({}) is larger than nodes in facebook cluster-{} ({})'.format(self.num_nodes, cluster, num_node))
        subset_nodes = random.sample(range(num_node), self.num_nodes)  # randomly select a subset of nodes
        self.matrix = self.matrix[np.ix_(subset_nodes, subset_nodes)]

        self.bound_traffic(multiply)

        ID = 0
        self.flows  = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                size = self.matrix[i][j]
                if size == 0 or i == j:
                    continue
                # route = self.random_route(i, j)

                all_routes = generateRandomRoutes(i, j, self.num_nodes, self.max_hop, self.num_routes)

                self.flows[(i, j)] = Flow(ID, i, j, size, all_routes[0], all_routes, self.num_nodes)
                ID += 1
        # print('\nInit succuess! Facebook cluster {}.'.format(cluster))
        return self.flows


    def __str__(self):
        c_sum = np.sum(self.matrix, 0)
        r_sum = np.sum(self.matrix, 1)
        c_sum = np.sort(c_sum)        # accending order
        r_sum = np.sort(r_sum)
        m     = int(self.num_nodes/2) # approximately medium
        total = np.sum(self.matrix)
        return '\n# of nodes = {}, max hop = {}, # of flows = {}, window_size = {}, total packets = {}\ncolum-sum min|medium|mean|max = {}|{}|{}|{}, row-sum min|medium|mean|max = {}|{}|{}|{}'.format(\
               self.num_nodes, self.max_hop, len(self.flows), self.window_size, total, c_sum[0], c_sum[m], int(c_sum.mean()), c_sum[self.num_nodes-1], r_sum[0], r_sum[m], int(r_sum.mean()), r_sum[self.num_nodes-1])


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
def generateTestFlows(num_nodes, max_route_length, num_routes, flow_size_generator = simpleFlowSizeGenerator, flow_prob = 1.0):
	flows = {}

	next_flow_id = 0
	for i in range(num_nodes):
		for j in range(num_nodes):
			# Skip cases where src_node == dst_node
			if i == j:
				continue

			# Random decides whether to create flow from i to j, based on flow_prob
			rv = random.random()
			if rv > flow_prob:
				continue

			# Generate data for this flow
			this_size = flow_size_generator()

			all_routes = generateRandomRoutes(i, j, num_nodes, max_route_length, num_routes)

			flows[(i, j)] = Flow(next_flow_id, i, j, this_size, all_routes[0], all_routes)

			next_flow_id += 1

	return flows

# Creates a set of random routes between src and dst. Guarenteed that there are no duplicate routes, and that no routes share the same first hop.
# Input:
#   src --> source node (as an int)
#   dst --> destination node (as an int)
#   num_nodes --> Total number of nodes in the network
#   max_route_length --> The maximum length of the route
#   num_routes --> The number of routes to generate
# Ouput:
#   routes --> List of routes, guarenteed to have no duplicates, and that no routes share the same first hop
def generateRandomRoutes(src, dst, num_nodes, max_route_length, num_routes, min_route_length = 1):
	routes = []

	while len(routes) < num_routes:
		this_route_length = random.randint(min_route_length, max_route_length)
		this_route = [src, dst]
		for x in range(this_route_length - 1):
			# Note: this will be inefficient if max_route_length is close to num_nodes
			next_node = random.randint(0, num_nodes - 1)
			while next_node in this_route:
				next_node = random.randint(0, num_nodes - 1)

			this_route.insert(x + 1, next_node)

		# Check that the route is unique and that it doesn't share the same first hop
		valid = True
		for other_route in routes:
			# Check if same first hop
			if this_route[0] == other_route[0] and this_route[1] == other_route[1]:
				valid = False
				break

			# Check if the route are the same (not really necessary, because the same routes will share the same first hop)
			if all(nr == no for nr, no in zip(this_route, other_route)):
				valid = False
				break

		if valid:
			routes.append(this_route)

	return routes

if __name__ == '__main__':
    t = Traffic(num_nodes=64, window_size=1000, max_hop=4, random_seed=1)
    # flows = t.microsoft(cluster=1)
    # print(t)
    # flows = t.sigmetrics(c_l=0.7, n_l=4, c_s=0.3, n_s=12)
    # print(t)
    # flows = t.facebook(cluster='A')
    # for k in flows:
    #     print(flows[k])
    # print(t)
    flows = t.facebook(cluster='A', multiply=2)
    print(t)
    print('\n****\n')
    flows = t.facebook(cluster='B', multiply=2)
    print(t)
    print('\n****\n')
    flows = t.facebook(cluster='C', multiply=2)
    print(t)
    # for k in flows:
    #     print(flows[k])
    #t.microsoft(2)
    #t.microsoft(3)