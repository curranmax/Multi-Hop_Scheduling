
import runner

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
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
		return float(output.packets_delivered) / output.total_objective_value * 100.0

	raise Exception('Unexpected metric: ' + str(metric))


def y_fmt(tick_val, pos):
    if tick_val > 100000:
        val = int(tick_val)/100000
        return '{:d}'.format(val)
    else:
        return tick_val


def plot_line(table, method, filename=None, x_label=None, x_log=False, y_label=None, absolute_ub=False):
	'''
	Args:
		table: the same table used by function tabulate.tabulate()
	'''
	markers    = ['o', 'h', 's', '^', 'D', 'P']     # markers
	linestyles = ['-', '--', '-', ':', '--', '-.']  # line styles
	
	fig, ax = plt.subplots(figsize=(16, 16))
	fig.subplots_adjust(left=0.2, right=0.96, top=0.8, bottom=0.15)
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

	legend_num_column = 2
	bbox_to_anchor = (0, 1)
	if absolute_ub and y_label == '% of Packets Deliverd':
		if x_label != 'Varying ave. hop count':
			y = [66 for _ in X]	
			plt.plot(X, y, label='A-UB')
			legend_num_column = 3
			bbox_to_anchor = (-0.28, 1.05)
		if x_label == 'Varying ave. hop count':
			y = [100, 50, 33]
			plt.plot(X, y, linestyle='--', marker='D', label='A-UB')

	plt.xticks(X)
	if x_log:       # log scale on x axis
		plt.xticks(np.arange(-4, 0, 1), ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$'], )
		plt.xlim([-4, -0.9])

	y_min = np.min(np.min(table, 0)[1:]) - 10
	y_max = 101
	if y_label == 'Objective Value':   # objective value comes in hundreds of thousands
		y_min = np.min(np.min(table, 0)[1:])*0.9
		y_max = np.max(np.max(table, 0)[1:])*1.1
		ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
		y_label = y_label + ' ($10^5$)'

	plt.ylim([y_min, y_max])
	plt.legend(bbox_to_anchor=bbox_to_anchor, loc='lower left', ncol=legend_num_column, fontsize=50)
	ax.tick_params(direction='in', length=15, width=5)
	ax.tick_params(pad=20)

	if x_label:
		ax.set_xlabel(x_label, labelpad=15)
	if y_label:
		ax.set_ylabel(y_label)
	if filename:
		fig.savefig('{}.png'.format(filename))
	else:
		fig.savefig('plot/tmp.png')


def plot1_1(path):
	filename = '{}/num_nodes.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.num_nodes)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NUN_NODE'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_num_node'), x_label='# of Nodes', y_label=metric_[i], absolute_ub=True)


def plot1_2(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i], absolute_ub=True)


def plot1_3(path):
	filename = '{}/skewness.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.cs)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[int(vs*100)] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['CS'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'skewness'), x_label='% of traffic by small flows', y_label=metric_[i], absolute_ub=True)


def plot1_4(path):
	filename = '{}/sparsity.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Split', 'Eclipse']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     '% of Link Utilization']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.nl, inpt.ns)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[int(vs[0])+int(vs[1])] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NL+NS'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'sparsity'), x_label='flows per node', y_label=metric_[i], absolute_ub=True)


def plot2(path):
	filename = '{}/real_traffic.txt'
	data = runner.readDataFromFile(filename.format(path))
	methods  = ['octopus-r', 'upper-bound', 'split', 'eclipse']
	metrics  = ['percent_packets_delivered']
	metrics_ = ['% of Packets Deliverd']

	reduce_func = average

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.input_source, inpt.cluster)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [list(vs) + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
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
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'eclipse']
	methods_ = ['Oct-r',     'UB',          'Eclipse']
	metric   = ['percent_objective_value']
	metric_  = [ '% of Objective Value']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i])


def plot4(path):
	filename = '{}/octopus.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'octopus+']
	methods_ = ['Oct-r',     'Oct+']
	metric   = ['percent_packets_delivered']
	metric_  = ['% of Packets Deliverd']

	reduce_func = average

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, 'octopus', 'vary_delta'), x_label='Delta/WindowSize', x_log=True, y_label=metric_[i])


def plot5(path):
	filenames = ['{}/eps.txt']
	data = sum((runner.readDataFromFile(filename.format(path)) for filename in filenames), [])

	methods  = ['octopus-e', 'octopus-r', 'upper-bound']
	methods_ = ['Oct-e', 'Oct-r', 'UB']
	metrics   = ['percent_packets_delivered', 'objective_value']
	metrics_  = ['% of Packets Deliverd', 'Objective Value']

	reduce_func = average

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.min_route_length)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['MIN_ROUTE'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric, 'vary_hop_count'), x_label='Varying ave. hop count', y_label=metric_, absolute_ub=True)


if __name__ == '__main__':
	
	plt.rcParams['font.size'] = 60
	plt.rcParams['font.weight'] = 'bold'
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['lines.linewidth'] = 10
	plt.rcParams['lines.markersize'] = 15

	path = 'data/6-23'

	plot1_1(path)  # num of nodes
	plot1_2(path)  # reconfig delta
	plot1_3(path)  # skewness
	plot1_4(path)  # sparsity
	plot2(path)    # real traffic
	plot3(path)    # reconfig delta + percentage objective value
	plot4(path)    # reconfig delta + octopus+/R
	plot5(path)    # average hop count
