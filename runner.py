
import argparse
from collections import defaultdict
from datetime import datetime
import os
import subprocess

# Defines default values for parameters
DEFAULT_NUM_NODES        = 64
DEFAULT_MAX_ROUTE_LENGTH = 3
DEFAULT_WINDOW_SIZE      = 10000
DEFAULT_RECONFIG_DELTA   = 20
DEFAULT_NUM_ROUTES       = 10
DEFAULT_INPUT_SOURCE     = 'sigmetrics'
DEFAULT_METHODS          = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse', 'octopus+']

class Input:
	def __init__(self, num_nodes         = DEFAULT_NUM_NODES,
						max_route_length = DEFAULT_MAX_ROUTE_LENGTH,
						window_size      = DEFAULT_WINDOW_SIZE,
						reconfig_delta   = DEFAULT_RECONFIG_DELTA,
						num_routes       = DEFAULT_NUM_ROUTES,
						input_source     = DEFAULT_INPUT_SOURCE,
						methods          = DEFAULT_METHODS):

		self.num_nodes        = int(num_nodes)
		self.max_route_length = int(max_route_length)
		self.window_size      = int(window_size)
		self.reconfig_delta   = int(reconfig_delta)
		self.num_routes       = int(num_routes)
		self.input_source     = str(input_source)

		if isinstance(methods, list):
			self.methods = methods
		elif isinstance(methods, str):
			self.methods = methods.split(',')

	def niceOutput(self):
		vals = []
		if self.num_nodes != DEFAULT_NUM_NODES:
			vals.append(('Num nodes', self.num_nodes))

		if self.max_route_length != DEFAULT_MAX_ROUTE_LENGTH:
			vals.append(('Route Length', self.max_route_length))

		if self.window_size != DEFAULT_WINDOW_SIZE:
			vals.append(('Window Size', self.window_size))

		if self.reconfig_delta != DEFAULT_RECONFIG_DELTA:
			vals.append(('Reconfig Delta', self.reconfig_delta))

		if self.num_routes != DEFAULT_NUM_ROUTES:
			vals.append(('Num Routes', self.num_routes))

		if self.input_source != DEFAULT_INPUT_SOURCE:
			vals.append(('Input Source', self.input_source))

		# TODO include methods

		if len(vals) == 0:
			return 'all default values'

		else:
			return '--> ' + ', '.join(map(lambda x: x[0] + ' = ' + str(x[1]), vals))

	def getArgs(self):
		args = ['-nn', self.num_nodes] + \
				['-rl', self.max_route_length] + \
				['-ws', self.window_size] + \
				['-rd', self.reconfig_delta] + \
				['-nr', self.num_routes] + \
				['-is', self.input_source] + \
				['-m'] + self.methods + \
				['-runner']

		return map(str, args)

	def strVals(self):
		return [('num_nodes',        self.num_nodes),
				('max_route_length', self.max_route_length),
				('window_size',      self.window_size),
				('reconfig_delta',   self.reconfig_delta),
				('num_routes',       self.num_routes),
				('input_source',     self.input_source),
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
				self.max_route_length == inpt.max_route_length and \
				self.window_size      == inpt.window_size and \
				self.reconfig_delta   == inpt.reconfig_delta and \
				self.num_routes       == inpt.num_routes and \
				self.input_source     == inpt.input_source

class Output:
	def __init__(self, total_objective_value = None,
						packets_delivered = None,
						packets_not_delivered = None,
						time_slots_used = None,
						time_slots_not_used = None,
						packets_by_tag = None):

		# Note that values are require for parameters being directly converted to floats and ints
		self.total_objective_value = float(total_objective_value)
		self.packets_delivered     = int(packets_delivered)
		self.packets_not_delivered = int(packets_not_delivered)
		self.time_slots_used       = int(time_slots_used)
		self.time_slots_not_used   = int(time_slots_not_used)

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
					('time_slots_not_used',   self.time_slots_not_used)]

		if self.packets_by_tag is not None:
			str_vals.append(('packets_by_tag',        ','.join(map(lambda x: str(x[0]) + '=' + str(x[1]), self.packets_by_tag.iteritems()))))

		return str_vals

	def __str__(self):
		str_vals = self.strVals()

		return ', '.join(map(lambda x: x[0] + ' --> ' + str(x[1]), str_vals))

def getInputAndOutput(vals):
	input_vals  = vals[:vals.index('OUTPUT')]
	output_vals = vals[vals.index('OUTPUT') + 1:]

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
	directory = os.path.dirname(out_file)
	if not os.path.exists(directory):
		os.makedirs(directory)

	f = open(out_file, 'a')

	vals = inpt.strVals()
	vals.append('OUTPUT')
	for method, output in output_by_method.iteritems():
		vals.append(('method', method))
		vals += output.strVals()

	out_str = ' '.join(map(lambda x: str(x[0]) + '|' + str(x[1]), vals))
	f.write(out_str + '\n')

	f.close()

def runAllTests(inputs, num_tests, out_file):
	for nt in range(num_tests):
		for inpt in inputs:
			args = ['python', 'run.py'] + inpt.getArgs()

			print 'Running test with params', inpt.niceOutput()
			print 'Start time:                 ', datetime.now().strftime('%A %I:%M %p')

			p = subprocess.Popen(args, stdout = subprocess.PIPE)
			p.wait()

			vals = []
			for line in p.stdout:
				vals.append(line.strip())

			check_inpt, output_by_method = getInputAndOutput(vals)

			if not check_inpt.equals(inpt):
				raise Exception('Input doesn\'t match')
			
			appendToFile(out_file, inpt, output_by_method)

# Experiment strings
NUM_NODES      = 'num_nodes'
RECONFIG_DELTA = 'reconfig_delta'
SPARSITY       = 'sparsity'
SKEWNESS       = 'skewness'

EXPERIMENTS = [NUM_NODES, RECONFIG_DELTA, SPARSITY, SKEWNESS]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Runs multiple iterations of run.py and saves results to a file')

	parser.add_argument('-exp', '--experiments', metavar = 'EXPERIMENT', type = str, nargs = '+', default = [], help = 'List of experiments to run. Must be one of (' + ', '.join(EXPERIMENTS) + ')')
	parser.add_argument('-nt', '--num_tests', metavar = 'NUM_TESTS', type = int, nargs = 1, default = [1], help = 'Number of times to run each set of input')

	parser.add_argument('-out', '--out_file', metavar = 'OUT_FILE', type = str, nargs = 1, default = [''], help = 'File to output the results to')


	args = parser.parse_args()

	experiments = args.experiments
	num_tests   = args.num_tests[0]
	out_file    = args.out_file[0]

	if len(experiments) == 0:
		raise Exception('Must specify at least one experiment to run')

	if out_file == '':
		# TODO create default filename based on current time
		out_file = 'data/default.txt'

	
	inputs = []
	for experiment in experiments:
		if experiment == NUM_NODES:
			num_nodes = [50, 100, 200, 300, 400, 500]

			# Note that traffic generation parameters are made to match size automatically

			for nn in num_nodes:
				inputs.append(Input(num_nodes = nn))

		if experiment == RECONFIG_DELTA:
			reconfig_deltas = [5, 10, 20, 30, 40, 50]

			for rd in reconfig_deltas:
				inputs.append(Input(reconfig_delta = rd))

		if experiment == SPARSITY:
			raise Exception('Not implemented yet')
	
		if experiment == SKEWNESS:
			raise Exception('Not implemented yet')

	runAllTests(inputs, num_tests, out_file)