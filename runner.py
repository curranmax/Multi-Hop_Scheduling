
import argparse
from collections import defaultdict
from datetime import datetime
import os
import subprocess
import time


# Defines default values for parameters
DEFAULT_NUM_NODES        = 100
DEFAULT_MIN_ROUTE_LENGTH = 1
DEFAULT_MAX_ROUTE_LENGTH = 3
DEFAULT_WINDOW_SIZE      = 10000
DEFAULT_RECONFIG_DELTA   = 20
DEFAULT_NUM_ROUTES       = 10
DEFAULT_USE_EPS          = False
DEFAULT_INPUT_SOURCE     = 'sigmetrics'
DEFAULT_METHODS          = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse', 'octopus+', 'octopus-e', 'octopus-b', 'octopus-greedy']
DEFAULT_NL               = 4    # number of large flows   (for sigmetrics only)
DEFAULT_NS               = 12   # number of small flows   (for sigmetrics only)
DEFAULT_CL               = 0.7  # capacity of large flows (for sigmetrics only)
DEFAULT_CS               = 0.3  # capacity of small flows (for sigmetrics only)
DEFAULT_CLUSTER          = 1    # cluster number (for Facebook and Microsoft only)

class Input:
	def __init__(self, num_nodes         = DEFAULT_NUM_NODES,
						min_route_length = DEFAULT_MIN_ROUTE_LENGTH,
						max_route_length = DEFAULT_MAX_ROUTE_LENGTH,
						window_size      = DEFAULT_WINDOW_SIZE,
						reconfig_delta   = DEFAULT_RECONFIG_DELTA,
						num_routes       = DEFAULT_NUM_ROUTES,
						use_eps          = DEFAULT_USE_EPS,
						input_source     = DEFAULT_INPUT_SOURCE,
						methods          = DEFAULT_METHODS,
						nl               = DEFAULT_NL,
						ns               = DEFAULT_NS,
						cl               = DEFAULT_CL,
						cs               = DEFAULT_CS,
						cluster          = DEFAULT_CLUSTER,
	    				
	    				# Only used when running experiments
	    				out_file = None,
	    				key_value = None):

		self.num_nodes        = int(num_nodes)
		self.min_route_length = int(min_route_length)
		self.max_route_length = int(max_route_length)
		self.window_size      = int(window_size)
		self.reconfig_delta   = int(reconfig_delta)
		self.num_routes       = int(num_routes)
		self.use_eps          = convertToBool(use_eps)
		self.input_source     = str(input_source)
		self.nl               = int(nl)
		self.ns               = int(ns)
		self.cl               = float(cl)
		self.cs               = float(cs)
		self.cluster          = int(cluster)

		if isinstance(methods, list):
			self.methods = methods
		elif isinstance(methods, str):
			self.methods = methods.split(',')
			
		self.out_file  = out_file
		self.key_value = key_value

	def getKeyValue(self):
		if self.key_value is None:
			return None

		return getattr(self, self.key_value)

	def niceOutput(self):
		vals = []
		if self.num_nodes != DEFAULT_NUM_NODES:
			vals.append(('Num nodes', self.num_nodes))

		if self.min_route_length != DEFAULT_MIN_ROUTE_LENGTH:
			vals.append(('Min Route Length', self.min_route_length))

		if self.max_route_length != DEFAULT_MAX_ROUTE_LENGTH:
			vals.append(('Max Route Length', self.max_route_length))

		if self.window_size != DEFAULT_WINDOW_SIZE:
			vals.append(('Window Size', self.window_size))

		if self.reconfig_delta != DEFAULT_RECONFIG_DELTA:
			vals.append(('Reconfig Delta', self.reconfig_delta))

		if self.num_routes != DEFAULT_NUM_ROUTES:
			vals.append(('Num Routes', self.num_routes))

		if self.use_eps != DEFAULT_USE_EPS:
			vals.append(('Use EPS', self.use_eps))

		if self.input_source != DEFAULT_INPUT_SOURCE:
			vals.append(('Input Source', self.input_source))
		
		if self.nl != DEFAULT_NL:
			vals.append(('Num Large flows', self.nl))
		
		if self.ns != DEFAULT_NS:
			vals.append(('Num Small flows', self.ns))
		
		if self.cl != DEFAULT_CL:
			vals.append(('Capacity of Large flows', self.cl))
		
		if self.cs != DEFAULT_CS:
			vals.append(('Capacity of Small flows', self.cs))
		
		if self.cluster != DEFAULT_CLUSTER:
			vals.append(('Default cluster', self.cluster))

		# TODO include methods

		if len(vals) == 0:
			return 'all default values'

		else:
			return '--> ' + ', '.join(map(lambda x: x[0] + ' = ' + str(x[1]), vals))

	def getArgs(self):
		args = ['-nn', self.num_nodes] + \
				['-min_rl', self.min_route_length] + \
				['-rl', self.max_route_length] + \
				['-ws', self.window_size] + \
				['-rd', self.reconfig_delta] + \
				['-nr', self.num_routes] + \
				['-eps', self.use_eps] + \
				['-is',  self.input_source] + \
				['-nl',  self.nl] + \
				['-ns',  self.ns] + \
				['-cl',  self.cl] + \
				['-cs',  self.cs] + \
				['-clus', self.cluster] + \
				['-m'] + self.methods + \
				['-runner']

		return map(str, args)

	def strVals(self):
		return [('num_nodes',        self.num_nodes),
				('min_route_length', self.min_route_length),
				('max_route_length', self.max_route_length),
				('window_size',      self.window_size),
				('reconfig_delta',   self.reconfig_delta),
				('num_routes',       self.num_routes),
				('use_eps',          self.use_eps),
				('input_source',     self.input_source),
				('nl',               self.nl),
				('ns',               self.ns),
				('cl',               self.cl),
				('cs',               self.cs),
				('cluster',          self.cluster),
				('methods',          ','.join(map(str, self.methods)))]

	def __str__(self):
		str_vals = self.strVals()

		return ', '.join(map(lambda x: x[0] + ' --> ' + str(x[1]), str_vals))

	def equals(self, inpt):
		if not isinstance(inpt, Input):
			return False

		if len(self.methods) != len(inpt.methods):
			return False

		if any(self.methods.count(m) != inpt.methods.count(m) for m in self.methods):
			return False

		return self.num_nodes         == inpt.num_nodes and \
				self.min_route_length == inpt.min_route_length and \
				self.max_route_length == inpt.max_route_length and \
				self.window_size      == inpt.window_size and \
				self.reconfig_delta   == inpt.reconfig_delta and \
				self.num_routes       == inpt.num_routes and \
				self.use_eps          == inpt.use_eps and \
				self.input_source     == inpt.input_source and \
				self.nl         	  == inpt.nl and \
				self.ns         	  == inpt.ns and \
				self.cl         	  == inpt.cl and \
				self.cs         	  == inpt.cs and \
				self.cluster          == inpt.cluster

class Output:
	def __init__(self, total_objective_value = None,
						packets_delivered = None,
						packets_not_delivered = None,
						time_slots_used = None,
						time_slots_not_used = None,
						packets_by_tag = None,
						computation_duration = None):

		# Note that values are require for parameters being directly converted to floats and ints
		self.total_objective_value = float(total_objective_value)
		self.packets_delivered     = int(float(packets_delivered))
		self.packets_not_delivered = int(float(packets_not_delivered))
		self.time_slots_used       = int(float(time_slots_used))
		self.time_slots_not_used   = int(float(time_slots_not_used))
		
		if computation_duration is None:
			self.computation_duration = 0
		else:
			self.computation_duration = float(computation_duration)

		if isinstance(packets_by_tag, dict):
			self.packets_by_tag = packets_by_tag
		elif isinstance(packets_by_tag, str):
			self.packets_by_tag = {val.split('=')[0]: int(val.split('=')[1]) for val in packets_by_tag.split(',')}
		else:
			self.packets_by_tag = None

	def strVals(self):
		str_vals = [('total_objective_value', self.total_objective_value),
					('packets_delivered',     self.packets_delivered),
					('packets_not_delivered', self.packets_not_delivered),
					('time_slots_used',       self.time_slots_used),
					('time_slots_not_used',   self.time_slots_not_used),
					('computation_duration',  self.computation_duration)]

		if self.packets_by_tag is not None:
			str_vals.append(('packets_by_tag',        ','.join(map(lambda x: str(x[0]) + '=' + str(x[1]), self.packets_by_tag.iteritems()))))

		return str_vals

	def __str__(self):
		str_vals = self.strVals()

		return ', '.join(map(lambda x: x[0] + ' --> ' + str(x[1]), str_vals))

def convertToBool(val):
	if isinstance(val, bool):
		return val
	if isinstance(val, str):
		return val in ['True', 'true', 'T', 't']
	raise Exception('Invalid bool value: ' + str(val))

def getInputAndOutput(vals):
	input_vals  = vals[:vals.index('OUTPUT')]
	output_vals = vals[vals.index('OUTPUT') + 1:]

	# TODO If any input_vals don't have the proper format, ignore them
	input_dict = {val.split('|')[0]: val.split('|')[1] for val in input_vals}
	inpt = Input(**input_dict)

	cur_method = None
	output_vals_by_method = defaultdict(list)
	for val in output_vals:
		if val[:7] == 'method|':
			cur_method = val[7:]
		elif cur_method is not None:
			output_vals_by_method[cur_method].append(val)
		else:
			raise Exception('No method declared')

	output_by_method = {method: Output(**{v.split('|')[0]: v.split('|')[1] for v in vals}) for method, vals in output_vals_by_method.iteritems()}

	return inpt, output_by_method

def appendToFile(out_file, inpt, output_by_method):
	print 'Adding results to:', out_file
	
	directory = os.path.dirname(out_file)
	if not os.path.exists(directory):
		os.makedirs(directory)

	f = open(out_file, 'a')

	vals = inpt.strVals()
	vals.append('OUTPUT')
	for method, output in output_by_method.iteritems():
		vals.append(('method', method))
		vals += output.strVals()

	def formatVals(val):
		if val == 'OUTPUT':
			return val
		else:
			return str(val[0]) + '|' + str(val[1])

	out_str = ' '.join(map(formatVals, vals))
	f.write(out_str + '\n')

	f.close()

def printStatus(status):
	max_fname_len = max(len(fname) for fname in status)
	print '-' * 100
	for fname, fstatus in sorted(status.iteritems()):
		print str(fname) + ' ' * (max_fname_len - len(fname)), '-->', ', '.join(map(lambda x: str(x[0]) + ': ' + str(x[1]), sorted(fstatus.iteritems())))
	print '-' * 100

def getDataFromProcess(proc, inpt, out_file, status = None):
	vals = []
	for line in proc.stdout:
		vals.append(line.strip())

	for line in proc.stderr:
		print 'STDERR:', line.strip()

	try:
		check_inpt, output_by_method = getInputAndOutput(vals)
	except Exception as e:
		print e

		print 'Error parsing values:', vals
		return

	if not check_inpt.equals(inpt):
		print 'Input doesn\'t match:'
		print 'Given: ', inpt
		print 'Parsed:', check_inpt
		return

	if status is not None:
		status[inpt.out_file][inpt.getKeyValue()] += 1
		printStatus(status)
	
	this_out_file = out_file
	if inpt.out_file is not None:
		this_out_file = inpt.out_file
		
	appendToFile(this_out_file, inpt, output_by_method)

def runAllTests(inputs, num_tests, out_file):
	for nt in range(num_tests):
		for inpt in inputs:
			args = ['python', 'run.py'] + inpt.getArgs()

			print 'Running test with params', inpt.niceOutput()
			print 'Start time:                 ', datetime.now().strftime('%A %I:%M %p')

			p = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
			p.wait()

			getDataFromProcess(p, inpt, out_file)

def runAllTestsInParallel(inputs, num_tests, out_file, num_cores = 1, wait_time = 10):
	if all(inpt.out_file is not None and inpt.key_value is not None for inpt in inputs):
		status = defaultdict(dict)
		for inpt in inputs:
			status[inpt.out_file][inpt.getKeyValue()] = 0

		printStatus(status)
	else:
		status = None

	ps = []
	for nt in range(num_tests):
		for inpt in inputs:
			# Starts a test
			args = ['python', 'run.py'] + inpt.getArgs()

			print 'Running test with params', inpt.niceOutput()
			print 'Start time:                 ', datetime.now().strftime('%A %I:%M %p')

			p = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
			ps.append((p, inpt))

			# Waits until there are free cores available to run more tests
			while len(ps) >= num_cores:
				new_ps = []
				for p, inpt in ps:
					if p.poll() is not None:
						getDataFromProcess(p, inpt, out_file, status = status)
					else:
						new_ps.append((p, inpt))
				ps = new_ps
				time.sleep(wait_time)

	while len(ps) > 0:
		new_ps = []
		for p, inpt in ps:
			if p.poll() is not None:
				getDataFromProcess(p, inpt, out_file, status = status)
			else:
				new_ps.append((p, inpt))
		ps = new_ps
		time.sleep(wait_time)

# Gets all test inputs and outputs saved to a file
# Input:
#   filename --> Name of the file holding the data
# Output:
#   tests --> List of 2-tuples (Input, {method_str: Output}). Each value is the input given to run.py and the output for each method
def readDataFromFile(filename):
	try:
		f = open(filename, 'r')
	except IOError:
		return []

	tests = []
	for line in f:
		vals = line.split()

		inpt, output_by_method = getInputAndOutput(vals)

		tests.append((inpt, output_by_method))

	return tests

# Experiment strings
NUM_NODES      = 'num_nodes'       # 1.1
RECONFIG_DELTA = 'reconfig_delta'  # 1.2
SKEWNESS       = 'skewness'        # 1.3
SPARSITY       = 'sparsity'        # 1.4
REAL_TRAFFIC   = 'real_traffic'    # 2
OCTOPUS        = 'octopus'         # 4 (the second 3 in the document)
EPS_TEST       = 'eps'             # 5 (the 4 in the document)

PROJECTOR = 'projector'
GREEDY =    'greedy'    # comparing the octopus-greedy agains octopus-fast


TEST = 'test'

EXPERIMENTS = [NUM_NODES, RECONFIG_DELTA, SPARSITY, SKEWNESS, EPS_TEST, REAL_TRAFFIC, OCTOPUS, PROJECTOR, TEST]

# python runner.py -exp reconfig_delta -nt NUM_TEST -out FILE

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Runs multiple iterations of run.py and saves results to a file')

	parser.add_argument('-exp', '--experiments', metavar = 'EXPERIMENT', type = str, nargs = '+', default = [], help = 'List of experiments to run. Must be one of (' + ', '.join(EXPERIMENTS) + ')')
	parser.add_argument('-nt', '--num_tests', metavar = 'NUM_TESTS', type = int, nargs = 1, default = [1], help = 'Number of times to run each set of input')

	parser.add_argument('-nc', '--num_cores', metavar = 'NUM_CORES', type = int, nargs = 1, default = [None], help = 'Number of cores to run tests on')
	parser.add_argument('-wt', '--wait_time', metavar = 'WAIT_TIME', type = int, nargs = 1, default = [30], help = 'Number of seconds to wait between polling the status of parallel processes')

	parser.add_argument('-out', '--out_file', metavar = 'OUT_FILE', type = str, nargs = 1, default = [''], help = 'File to output the results to')


	args = parser.parse_args()

	experiments = args.experiments
	num_tests   = args.num_tests[0]

	num_cores = args.num_cores[0]
	wait_time = args.wait_time[0]

	out_file    = args.out_file[0]

	if len(experiments) == 0:
		raise Exception('Must specify at least one experiment to run')

	if out_file == '':
		# TODO create default filename based on current time
		out_file = 'data/default.txt'

	
	inputs = []
	for experiment in experiments:
		if experiment == TEST:
			inputs.append(Input(window_size = 100))

		elif experiment == NUM_NODES:
			num_nodes = [25, 50, 75, 100, 125, 150, 200, 300]
			num_large = [1,  2,  3,  4,   5,   6,   8,   12]
			num_small = [3,  6,  9,  12,  15,  18,  24,  36]
			# methods   = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-e', 'octopus-b']
			methods = ['octopus-b']
			
			for nn, nl, ns in zip(num_nodes, num_large, num_small):
				inputs.append(Input(num_nodes = nn, nl = nl, ns = ns, methods = methods, out_file = out_file.format('num_nodes'), key_value = 'num_nodes'))

		elif experiment == RECONFIG_DELTA:
			reconfig_deltas = [2, 5, 10, 20, 50, 100, 200, 500]
			# methods         = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-e', 'octopus-b']
			methods = ['octopus-b']

			for rd in reconfig_deltas:
				inputs.append(Input(reconfig_delta = rd, methods = methods, out_file = out_file.format('reconfig_delta'), key_value = 'reconfig_delta'))

		elif experiment == SPARSITY:
			num_large = [1, 2, 3, 4,  5,  6,  7,  8]
			num_small = [3, 6, 9, 12, 15, 18, 21, 24]

			# methods   = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-e', 'octopus-b']
			methods = ['octopus-b']

			for nl, ns in zip(num_large, num_small):
				inputs.append(Input(nl = nl, ns = ns, methods = methods, out_file = out_file.format('sparsity'), key_value = 'nl'))
	
		elif experiment == SKEWNESS:
			capa_large = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
			capa_small = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
			# methods    = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-e', 'octopus-b']
			methods = ['octopus-b']

			for cl, cs in zip(capa_large, capa_small):
				inputs.append(Input(cl = cl, cs = cs, methods = methods, out_file = out_file.format('skewness'), key_value = 'cl'))

		elif experiment == EPS_TEST:
			# methods       = ['upper-bound', 'octopus-r', 'octopus-e']
			methods       = ['octopus-b']
			route_lengths = [1, 2, 3]

			for route_length in route_lengths:
				inputs.append(Input(min_route_length = route_length, max_route_length = route_length, num_routes = 1, methods = methods, out_file = out_file.format('eps'), key_value = 'min_route_length'))

		elif experiment == REAL_TRAFFIC:
			# methods = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-e']
			input_source = ['microsoft', 'facebook']
			cluster = ['1', '2', '3']

			methods = ['octopus-b']

			for source in input_source:
				for clus in cluster:
					inputs.append(Input(input_source = source, cluster = clus, methods = methods, out_file = out_file.format('real_traffic'), key_value = 'input_source'))

		elif experiment == OCTOPUS:
			# methods = ['octopus+', 'octopus-r']
			methods = ['octopus-b']
			reconfig_deltas = [2, 5, 10, 20, 50, 100, 200, 500]

			for rd in reconfig_deltas:
				inputs.append(Input(reconfig_delta = rd, methods = methods, out_file = out_file.format('octopus'), key_value = 'reconfig_delta'))
		
		elif experiment == PROJECTOR:
			# reconfig_deltas = [2, 5, 10, 20, 50, 100, 200, 500]
			reconfig_deltas = [2, 5, 10, 20, 50, 100, 200, 500]
			# reconfig_deltas = [500]
			# methods = ['optopus-greedy']
			methods = ['octopus-r', 'projector']

			for rd in reconfig_deltas:
				inputs.append(Input(reconfig_delta=rd, methods = methods, out_file = out_file.format('reconfig_delta'), key_value = 'reconfig_delta'))

		elif experiment == GREEDY:
			reconfig_deltas = [2, 5, 10, 20, 50, 100, 200, 500]
			methods = ['octopus-greedy', 'octopus-r']

			for rd in reconfig_deltas:
				inputs.append(Input(reconfig_delta=rd, methods = methods, out_file = out_file.format('reconfig_delta'), key_value = 'reconfig_delta'))

		else:
			raise Exception('Unexpected experiment: ' + str(experiment))

	if num_cores is None or num_cores <= 0:
		runAllTests(inputs, num_tests, out_file)
	else:
		runAllTestsInParallel(inputs, num_tests, out_file, num_cores = num_cores, wait_time = wait_time)


# python runner.py -exp projector -nt 2 -nc 11 -out data/9-22-2/{}.txt