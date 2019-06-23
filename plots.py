
import runner

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import tabulate

from runner import DEFAULT_WINDOW_SIZE


def average(vals):
	return float(sum(vals)) / float(len(vals))


def getMetric(inpt, output, metric = 'percent_packets_delivered'):
	if metric == 'percent_packets_delivered':
		return float(output.packets_delivered) / float(output.packets_delivered + output.packets_not_delivered) * 100.0

	if metric == 'link_utilization':
		return float(output.time_slots_used) / float(output.time_slots_used + output.time_slots_not_used) * 100.0

	if metric == 'objective_value':
		return output.total_objective_value

	if metric == 'percent_objective_value':
		return output.total_objective_value / float(output.packets_delivered + output.packets_not_delivered)  * 100.0

	raise Exception('Unexpected metric: ' + str(metric))


def plot_line(table, method, filename=None, x_label=None, x_log=False, y_label=None):
	'''
	Args:
		table: the same table used by function tabulate.tabulate()
	'''
	markers    = ['o', 'h', 's', '^', 'D', 'P']     # markers
	linestyles = ['-', '--', '-', ':', '--', '-.']  # line styles
	
	fig, ax = plt.subplots(figsize=(16, 16))
	fig.subplots_adjust(left=0.16, right=0.96, top=0.85, bottom=0.15)
	X  = [row[0] for row in table]
	if x_log:
		X = np.log10((np.array(X, float)/DEFAULT_WINDOW_SIZE))
	num = len(table[0]) - 1                         # number of methods
	Y = []
	for i in range(1, num+1):
		y = [row[i] for row in table]
		Y.append(y)
	for i in range(0, num):
		plt.plot(X, Y[i], linestyle=linestyles[i], marker=markers[i], label=method[i])

	plt.xticks(X)
	if x_log:
		plt.xticks(np.arange(-4, 0, 1), ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$'], )
		plt.xlim([-4, -0.9])
	y_min = np.min(np.min(table, 0)[1:]) - 10
	y_max = 101
	plt.ylim([y_min, y_max])
	plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2, fontsize=45)
	ax.tick_params(direction='in', length=10, width=3)
	ax.tick_params(axis='x', pad=15)
	if x_label:
		ax.set_xlabel(x_label)
	if y_label:
		ax.set_ylabel(y_label)
	if filename:
		fig.savefig('{}.png'.format(filename))
	else:
		fig.savefig('plot/tmp.png')
	#plt.show()


def plot1_1(path):
	# python runner.py -exp num_nodes -nt 1 -out data/6-22/1.1.txt

	filename = '{}/1.1.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.num_nodes)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[vs] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NUN_NODE'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_num_node'), x_label='# of Nodes', y_label=metric_[i])


def plot1_2(path):
	# python runner.py -exp reconfig_delta -nt 3 -out data/6-22/1.2.txt

	filename = '{}/1.2.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[vs] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i])


def plot1_3(path):
	# python runner.py -exp skewness -nt 3 -out data/6-22/1.3.txt

	filename = '{}/1.3.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.cs)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[int(vs*100)] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['CS'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'skewness'), x_label='% of traffic carry by small flows', y_label=metric_[i])


def plot1_4(path):
	# python runner.py -exp sparsity -nt 3 -out data/6-22/1.4.txt
	
	filename = '{}/1.4.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.nl, inpt.ns)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[int(vs[0])+int(vs[1])] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NL+NS'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'sparsity'), x_label='flows per node', y_label=metric_[i])


def plot2(path):
	# python runner.py -exp real_traffic -nt 3 -out data/6-22/2.txt  (somehow three experiments are getting the same results, random seed. but how the experiments in 1.* worked?)
	filename = '{}/2.txt'
	data = runner.readDataFromFile(filename.format(path))
	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	metrics  = ['percent_packets_delivered']
	metrics_ = ['% of Packets Deliverd']

	for metric, metric_ in zip(metrics, metrics_):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.input_source, inpt.cluster)] = {method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()}
		
		print_table = [list(vs) + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['SOURCE', 'CLUSTER'] + methods)
		
		arr = np.array(print_table)
		oct_r = np.array(arr[:, 2], float)
		ub    = np.array(arr[:, 3], float)
		split = np.array(arr[:, 4], float)
		eclip = np.array(arr[:, 5], float)

		ind   = np.arange(len(oct_r))
		width = 0.16
		
		fig, ax = plt.subplots(figsize=(22.2, 15))
		fig.subplots_adjust(left=0.15, right=0.96, top=0.85, bottom=0.1)
		pos1 = ind - width*1.5
		pos2 = ind - width*0.5
		pos3 = ind + width*0.5
		pos4 = ind + width*1.5
		ax.bar(pos1, eclip, width, edgecolor='black', label='Eclipse')
		ax.bar(pos2, split, width, edgecolor='black', label='Split')
		ax.bar(pos3, oct_r, width, edgecolor='black', label='Oct-r')
		ax.bar(pos4, ub,    width, edgecolor='black', label='UB')
		
		plt.legend(bbox_to_anchor=(-0.02, 1), loc='lower left', ncol=4, fontsize=45)
		ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
		ax.set_ylabel(metric_)
		ax.set_xlabel('Varies Clusters')
		plt.xticks(ind, ['FB-1', 'FB-2', 'FB-3', 'MS-1', 'MS-2', 'MS-3'], fontsize=45)
		plt.savefig('{}/{}-real_traffic'.format(path, metric))


def plot3(path):
	# the data comes from python runner.py -exp reconfig_delta -nt 3 -out data/6-22/1.2.txt
	filename = '{}/1.2.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Eclipse']
	metric   = ['percent_objective_value']
	metric_  = [ '% of Objective Value']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[vs] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i])


def plot4(path):
	# python runner.py -exp octopus -nt 3 -out data/6-22/4.txt (somehow three experiments are getting the same results, random seed. but how the experiments in 1.* worked?)
	filename = '{}/4.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'octopus+']
	methods_ = ['Oct-r',     'Oct+']
	metric   = ['percent_packets_delivered']
	metric_  = ['% of Packets Deliverd']

	for i in range(0, len(metric)):
		table = {}
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)] = {method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()}
		
		print_table = [[vs] + [(vals_by_method[method] if method in vals_by_method else None) for method in methods] for vs, vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, 'Octopus', 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i])



def plot5(path):
	# python runner.py -exp eps -nt 3 -out data/6-22/5.txt
	filenames = ['{}/5.txt', '{}/5_2.txt']
	data = sum((runner.readDataFromFile(filename.format(path)) for filename in filenames), [])

	methods  = ['octopus-e', 'octopus-r', 'upper-bound']
	methods_ = ['Oct-e', 'Oct-r', 'UB']
	metrics   = ['percent_packets_delivered']
	metrics_  = ['% of Packets Deliverd']

	reduce_func = average

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.min_route_length)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, 'Octopus', 'vary_hop_count'), x_label='Varying ave. hop count', y_label=metric_)


if __name__ == '__main__':
	
	plt.rcParams['font.size'] = 55
	plt.rcParams['font.weight'] = 'bold'
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['lines.linewidth'] = 10
	plt.rcParams['lines.markersize'] = 15

	path = 'data/6-22'

	# plot1_1(path)
	# plot1_2(path)
	# plot1_3(path)
	# plot1_4(path)
	# plot2(path)
	# plot3(path)
	# plot4(path)
	plot5(path)
