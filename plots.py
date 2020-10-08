
import runner

from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import tabulate
from runner import DEFAULT_WINDOW_SIZE
import shutil


def average(vals):
	vals = list(filter(None, vals))  # filter out the None values
	return float(sum(vals)) / float(len(vals))

def min_max(vals):
	vals = list(filter(None, vals))  # filter out the None values
	vals = np.array(vals)
	return (round(vals.min(), 4), round(vals.max(), 4))

reduce_func  = average
reduce_func2 = min_max

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

# _MARKER = ['o',       'h',        's',         '^',             'D',       'P']

METHOD  = ['Octopus-Random', 'Octopus', 'Octopus+', 'Octopus-e', 'Octopus-B', 'Eclipse-Based', 'UB',      'Absolute-UB', 'Projector', 'Octopus-G', 'Rotornet']
_COLOR  = ['b',              'b',       'c',        'gray',      'k',         'g',             'orange',  'r',           'purple',    'cyan',      'magenta']
_MARKER = ['o',              'o',       'o',        'o',         'o',         'o',             'o',       'o',           'o',         'o',         'o']
_LINE   = ['-',              '-',       ':',        ':',         ':',         ':',             ':',       '-',           '-',         '-',         ':']
COLOR   = dict(zip(METHOD, _COLOR))
MARKER  = dict(zip(METHOD, _MARKER))
LINE    = dict(zip(METHOD, _LINE))



def plot_line(table, methods, filename=None, x_label=None, x_log=False, y_label=None, absolute_ub=False, yerr_table=None):
	'''
	Args:
		table: the same table used by function tabulate.tabulate()
		  NUN_NODE    octopus-r    upper-bound    eclipse
		----------  -----------  -------------  ---------
		        25      48.1263        47.4182    38.6041
		        50      47.3271        47.0469    29.7168
		        75      51.8396        51.7936    33.1816
		       100      52.2284        52.0966    32.3763
		       125      53.7332        53.7551    33.0334
		       150      53.8109        53.9773    32.2842
		
		yerr_table: the same table used by function tabulate.tabulate()
		  NUN_NODE  octopus-r           upper-bound         eclipse
		----------  ------------------  ------------------  ------------------
		       25   (38.8811, 56.3964)  (35.3078, 55.9665)  (25.491, 51.4077)
		       50   (45.577, 48.3904)   (46.1241, 48.3337)  (27.5965, 32.0417)
		       75   (49.7072, 55.2583)  (50.6886, 54.9006)  (30.5833, 37.1484)
		      100   (51.7271, 53.0539)  (51.4198, 52.9464)  (30.8013, 34.1405)
		      125   (52.5892, 54.7168)  (52.4997, 54.6874)  (31.7188, 34.7831)
		      150   (52.2469, 54.7792)  (52.0139, 55.1176)  (29.5608, 33.4454)
	'''
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
	
	# core plotting
	if yerr_table is None:
		for i in range(0, num):
			method = methods[i]
			plt.plot(X, Y[i], linestyle=LINE[method], marker=MARKER[method], color=COLOR[method], label=method)
	else:
		for i in range(0, num):
			method = methods[i]
			col = i + 1
			min_, max_ = [], []
			for row in yerr_table:
				min_.append(row[col][0])
				max_.append(row[col][1])
			yerr = np.stack((min_, max_))
			yerr[0] = Y[i] - yerr[0]
			yerr[1] = yerr[1] - Y[i]
			plt.errorbar(X, Y[i], yerr=yerr, capsize=15, capthick=3, elinewidth=5, linestyle=LINE[method], marker=MARKER[method], color=COLOR[method], label=method)

	if absolute_ub and y_label == '% of Packets Deliverd':
		method = 'Absolute-UB'
		if x_label != 'Average # of hops' and ('Octopus-B' not in methods):
			y = [66 for _ in X]	
			plt.plot(X, y, linestyle=LINE[method], marker=MARKER[method], color=COLOR[method], label=method)
		if x_label == 'Average # of hops':
			y = [100, 50, 33]
			plt.plot(X, y, linestyle=LINE[method], marker=MARKER[method], color=COLOR[method], label=method)
		
	plt.xticks(X)
	if x_log:       # log scale on x axis
		plt.xticks(np.arange(-4, 0, 1), ['1', '10', '100', '1000'])
		plt.xlim([-4, -0.9])

	y_min = np.min(np.min(table, 0)[1:]) - 10
	y_max = 101
	if y_label == 'Objective Value':   # objective value comes in hundreds of thousands
		y_min_ = np.min(np.min(table, 0)[1:])*0.9
		y_max_ = np.max(np.max(table, 0)[1:])*1.1
		y_min = int(100000*(y_min_//100000))
		y_max = int(100000*(y_max_//100000 + 1))
		yticks = list(range(y_min, y_max, 100000))
		plt.yticks(yticks)
		ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
		y_label = y_label + ' ($10^5$)'
		y_max = int(100000*(y_max_//100000))

	elif y_label == 'Link Utilization (%)' and x_label != 'Various Facebook and Microsoft Clusters':
		y_min_ = np.min(np.min(table, 0)[1:])*0.85
		y_max_ = np.max(np.max(table, 0)[1:])*1.1
		y_min = int(10*(y_min_//10))
		y_max = int(10*(y_max_//10 + 1))
		yticks = list(range(y_min, y_max, 10))
		plt.yticks(yticks)
		y_max = int(10*(y_max_//10))

	elif y_label == '% of Packets Deliverd' and absolute_ub:
		y_min_ = np.min(np.min(table, 0)[1:])*0.9
		y_min = int(10*(y_min_//10))
		if x_label != 'Average # of hops':
			y_max = 80
		if x_label == 'Average # of hops':
			y_max = 110
		yticks = list(range(y_min, y_max, 10))
		plt.yticks(yticks)
		y_max = 100 if x_label == 'Average # of hops' else 70
	
	box_to_anchor = (-0.1, 1.04)
	if 'Octopus-Random' in methods or 'Octopus' in methods:
		box_to_anchor = (-0.15, 1.04)
	
	plt.ylim([y_min, y_max])
	plt.legend(bbox_to_anchor=box_to_anchor, loc='lower left', ncol=2, fontsize=50)
	ax.tick_params(direction='in', length=15, width=5)
	ax.tick_params(pad=20)

	if x_label == '# of Nodes':
		plt.xticks([50, 100, 150, 200, 250, 300])
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

	methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.num_nodes)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NUN_NODE'] + methods)
		print tabulate.tabulate(yerr_table, headers = ['NUN_NODE'] + methods)
		print ''
		
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_num_node'), x_label='# of Nodes', y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def plot1_2(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method]  if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table, headers = ['DELTA'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def plot1_3(path):
	filename = '{}/skewness.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.cs)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[int(vs*100)] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['CS'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'skewness'), x_label='% of traffic by small flows', y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def plot1_4(path):
	filename = '{}/sparsity.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.nl, inpt.ns)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[int(vs[0])+int(vs[1])] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['NL+NS'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'sparsity'), x_label='flows per node', y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def plot2_1(path):
	filename = '{}/real_traffic.txt'
	data = runner.readDataFromFile(filename.format(path))
	methods  = ['octopus-r', 'upper-bound', 'eclipse']
	metrics  = ['percent_packets_delivered']
	metrics_ = ['% of Packets Deliverd']

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
		eclip = np.array(arr[:, 4], float)
		a_ub  = [100] * 6

		ind   = np.arange(len(oct_r))
		width = 0.16
		
		# fig, ax = plt.subplots(figsize=(23.5, 16))                        # SIGMETRICS
		fig, ax = plt.subplots(figsize=(16, 16))                            # Mobihoc
		# fig.subplots_adjust(left=0.15, right=0.96, top=0.85, bottom=0.1)  # SIGMETRICS
		fig.subplots_adjust(left=0.2, right=0.96, top=0.8, bottom=0.15)     # Mobihoc
		pos1 = ind - width*1.5-0.03
		pos2 = ind - width*0.5-0.01
		pos3 = ind + width*0.5+0.01
		pos4 = ind + width*1.5+0.03
		ax.bar(pos1, eclip, width, edgecolor='black', label='Eclipse-Based', color=COLOR['Eclipse-Based'])
		ax.bar(pos2, oct_r, width, edgecolor='black', label='Octopus', color=COLOR['Octopus'])
		ax.bar(pos3, ub, width, edgecolor='black', label='UB', color=COLOR['UB'])
		ax.bar(pos4, a_ub,    width, edgecolor='black', label='Absolute-UB', color=COLOR['Absolute-UB'])
		
		box_to_anchor = (-0.22, 1.04)
		plt.legend(bbox_to_anchor=box_to_anchor, loc='lower left', ncol=2, fontsize=50)
		# ax.tick_params(direction='in', length=15, width=5)
		ax.tick_params(pad=20)

		# plt.legend(bbox_to_anchor=(-0.18, 1.04), loc='lower left', ncol=4, fontsize=45)
		# ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
		ax.set_ylabel(metric_)
		ax.set_xlabel('Various FB and MS Clusters', labelpad=25)
		plt.xticks(ind, ['FB-1', 'FB-2', 'FB-3', 'MS-1', 'MS-2', 'MS-3'], fontsize=50)
		plt.savefig('{}/{}-real_traffic'.format(path, metric))
		try:
			src = '/home/caitao/Project/Multi-Hop_Scheduling'
			dest = '/home/caitao/Project/latex/hybrid-circuit-switch-lcn-2019/figures/'
			shutil.copy(src + '/{}/{}-real_traffic.png'.format(path, metric), dest)
		except Exception as e:
			print(e)


def plot2_2(path):
	filename = '{}/real_traffic.txt'
	data = runner.readDataFromFile(filename.format(path))
	methods  = ['octopus-r', 'upper-bound', 'eclipse']
	metrics  = ['link_utilization']
	metrics_ = ['Link Utilization (%)']

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
		eclip = np.array(arr[:, 4], float)

		ind   = np.arange(len(oct_r))
		width = 0.21
		
		fig, ax = plt.subplots(figsize=(23.5, 16))
		fig.subplots_adjust(left=0.15, right=0.96, top=0.85, bottom=0.1)
		pos1 = ind - width*1.5-0.02
		pos2 = ind - width*0.5
		pos3 = ind + width*0.5+0.02
		ax.bar(pos1, eclip, width, edgecolor='black', label='Eclipse-Based', color=COLOR['Eclipse-Based'])
		ax.bar(pos2, oct_r, width, edgecolor='black', label='Octopus', color=COLOR['Octopus'])
		ax.bar(pos3, ub, width, edgecolor='black', label='UB', color=COLOR['UB'])
		
		plt.legend(bbox_to_anchor=(0, 1.04), loc='lower left', ncol=4, fontsize=45)
		ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
		ax.set_ylabel(metric_)
		ax.set_ylim([0, 80])
		ax.set_xlabel('Various Facebook and Microsoft Clusters')
		plt.xticks(ind, ['FB-1', 'FB-2', 'FB-3', 'MS-1', 'MS-2', 'MS-3'], fontsize=45)
		plt.savefig('{}/{}-real_traffic'.format(path, metric))


def plot2_(path, file):
	filename = '{}/{}.txt'
	data = runner.readDataFromFile(filename.format(path, file))
	methods  = ['octopus-r', 'split']
	metrics  = ['percent_packets_delivered']
	metrics_ = ['% of Packets Deliverd']

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.input_source, inpt.cluster)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [list(vs) + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['SOURCE', 'CLUSTER'] + methods)
		
		arr = np.array(print_table)
		oct_r = np.array(arr[:, 2], float)
		split = np.array(arr[:, 3], float)

		ind   = np.arange(len(oct_r))
		width = 0.3
		
		fig, ax = plt.subplots(figsize=(22.2, 15))
		fig.subplots_adjust(left=0.15, right=0.96, top=0.85, bottom=0.1)
		pos1 = ind - width*0.5
		pos2 = ind + width*0.5
		ax.bar(pos1, split, width, edgecolor='black', label='Split')
		ax.bar(pos2, oct_r, width, edgecolor='black', label='Octopus')
		
		plt.legend(bbox_to_anchor=(-0.02, 1), loc='lower left', ncol=4, fontsize=45)
		ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
		ax.set_ylabel(metric_)
		ax.set_xlabel('Varies Clusters')
		plt.xticks(ind, ['FB-1', 'MS-1'], fontsize=45)
		plt.savefig('{}/{}-{}'.format(path, metric, file))


def plot3(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r', 'upper-bound', 'eclipse']
	methods_ = ['Octopus',     'UB',          'Eclipse-Based']
	metric   = ['percent_objective_value']
	metric_  = [ 'Packets Del. as a % of $\\psi$']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method]  if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table,  headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], yerr_table=yerr_table)


def plot4(path):
	filename = '{}/octopus.txt'
	data = runner.readDataFromFile(filename.format(path))

	methods  = ['octopus-r',      'octopus+']
	methods_ = ['Octopus-Random', 'Octopus+']
	metric   = ['percent_packets_delivered']
	metric_  = ['% of Packets Deliverd']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table,  headers = ['DELTA'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, 'octopus', 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], yerr_table=yerr_table)


def plot5(path):
	filenames = ['{}/eps.txt']
	data = sum((runner.readDataFromFile(filename.format(path)) for filename in filenames), [])

	methods  = ['octopus-r', 'upper-bound', 'octopus-e']
	methods_ = ['Octopus',   'UB',          'Octopus-e']
	metrics   = ['percent_packets_delivered', 'objective_value']
	metrics_  = ['% of Packets Deliverd', 'Objective Value']

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.min_route_length)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['MIN_ROUTE'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric, 'vary_hop_count'), x_label='Average # of hops', y_label=metric_, absolute_ub=True)


def plot_7(path):
	filenames = ['{}/reconfig_delta.txt']
	data = sum((runner.readDataFromFile(filename.format(path)) for filename in filenames), [])

	methods  = ['octopus-r', 'octopus-b']
	methods_ = ['Octopus',   'Octopus-B']
	metrics   = ['percent_packets_delivered']
	metrics_  = ['% of Packets Deliverd']

	for metric, metric_ in zip(metrics, metrics_):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric
		print tabulate.tabulate(print_table, headers = ['MIN_ROUTE'] + methods)
		print tabulate.tabulate(yerr_table,  headers = ['MIN_ROUTE'] + methods)
		print ''
		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, 'octopus', 'binary'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_, absolute_ub=True, yerr_table=yerr_table)


def oneshot_revision_projector(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	# methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	# methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	methods  = ['octopus-r', 'projector']
	methods_ = ['Octopus',   'Projector']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method]  if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table, headers = ['DELTA'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def oneshot_revision_rotornet(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	# methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	# methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	methods  = ['octopus-r', 'rotornet']
	methods_ = ['Octopus',   'Rotornet']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method]  if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table, headers = ['DELTA'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


def oneshot_revision_greedy(path):
	filename = '{}/reconfig_delta.txt'
	data = runner.readDataFromFile(filename.format(path))

	# methods  = ['eclipse'      , 'octopus-r', 'upper-bound']
	# methods_ = ['Eclipse-Based', 'Octopus',   'UB']
	methods  = ['octopus-r', 'octopus-greedy']
	methods_ = ['Octopus', 'Octopus-G']
	metric   = ['percent_packets_delivered', 'link_utilization']
	metric_  = ['% of Packets Deliverd',     'Link Utilization (%)']

	for i in range(0, len(metric)):
		table = defaultdict(list)
		for inpt, output_by_method in data:
			table[(inpt.reconfig_delta)].append({method: getMetric(inpt, output, metric=metric[i]) for method, output in output_by_method.iteritems()})
		
		print_table = [[vs] + [reduce_func([(vals_by_method[method]  if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		yerr_table  = [[vs] + [reduce_func2([(vals_by_method[method] if method in vals_by_method else None) for vals_by_method in list_of_vals_by_method]) for method in methods] for vs, list_of_vals_by_method in sorted(table.iteritems())]
		print metric[i]
		print tabulate.tabulate(print_table, headers = ['DELTA'] + methods)
		print tabulate.tabulate(yerr_table, headers = ['DELTA'] + methods)
		print ''

		plot_line(print_table, methods_, filename='{}/{}-{}'.format(path, metric[i], 'vary_delta'), x_label='Reconfig. Delay (# of slots)', x_log=True, y_label=metric_[i], absolute_ub=True, yerr_table=yerr_table)


if __name__ == '__main__':
	
	plt.rcParams['font.size'] = 60
	# plt.rcParams['font.weight'] = 'bold'
	# plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['lines.linewidth'] = 10
	plt.rcParams['lines.markersize'] = 15

	# path = 'data/6-23'

	# plot1_1(path)  # num of nodes
	# plot1_2(path)  # reconfig delta
	# plot1_3(path)  # skewness
	# plot1_4(path)  # sparsity
	# plot2_1(path)    # real traffic
	# plot2_2(path)    # real traffic
	# plot3(path)    # reconfig delta + objective value
	# plot4(path)    # reconfig delta + octopus+/R
	# plot5(path)    # average hop count
	# plot_7(path)     # Octopus-B

	# plot2_(path, 'real_traffic-10-merge')    # real traffic

	# path = 'data/9-22-slow'
	# oneshot_revision_projector(path)

	# path = 'data/9-23-slow-rotor'
	# oneshot_revision_rotornet(path)

	path = 'data/10-8-greedy'
	oneshot_revision_greedy(path)
