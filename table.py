
import runner
import numpy as np
import matplotlib.pyplot as plt
import tabulate

METHODS  = ['octopus-r', 'upper-bound', 'split', 'eclipse', 'octopus-s', 'octopus+']
METHODS_ = ['Oct-r',     'UB',          'Split', 'Eclipse', 'Oct-s',     'Oct+']
METRIC   = ['percent_packets_delivered', 'link_utilization', 'percent_objective_value']
METRIC_  = ['% of Packets Deliverd',     '% of Link Utilization', '% of Objective Value']

def getMetric(inpt, output, metric = 'percent_packets_delivered'):
	if metric == 'percent_packets_delivered':
		return float(output.packets_delivered) / float(output.packets_delivered + output.packets_not_delivered) * 100.0

	if metric == 'link_utilization':
		return float(output.time_slots_used) / float(output.time_slots_used + output.time_slots_not_used) * 100.0

	if metric == 'objective_value':
		return output.total_objective_value

	if metric == 'percent_objective_value':
		return output.total_objective_value / float(inpt.window_size) / float(inpt.num_nodes) * 100.0

	raise Exception('Unexpected metric: ' + str(metric))


def plot_line(table, method, filename=None, x_label=None, y_label=None):
	'''
	Args:
		table: the same table used by function tabulate.tabulate()
	'''
	markers    = ['o', 'h', 's', '^', 'D', 'P']     # markers
	linestyles = ['-', '--', '-', ':', '--', '-.']  # line styles
	plt.rcParams['font.size'] = 60
	plt.rcParams['font.weight'] = 'bold'
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['lines.linewidth'] = 7
	fig, ax = plt.subplots(figsize=(16, 16))
	fig.subplots_adjust(left=0.2, right=0.99, top=0.8, bottom=0.15)
	X  = [row[0] for row in table]
	num = len(table[0]) - 1                         # number of methods
	Y = []
	for i in range(1, num+1):
		y = [row[i] for row in table]
		Y.append(y)
	for i in range(0, num):
		plt.plot(X, Y[i], linestyle=linestyles[i], marker=markers[i], markersize=10, label=method[i])

	plt.xticks(X)
	y_min = np.min(np.min(table, 0)[1:]) - 5
	y_max = 101
	plt.ylim([y_min, y_max])
	plt.legend(bbox_to_anchor=(-0.1, 1), loc='lower left', ncol=3, fontsize=40)

	if x_label:
		ax.set_xlabel(x_label)
	if y_label:
		ax.set_ylabel(y_label)
	if filename:
		fig.savefig('plot/{}.png'.format(filename))
	else:
		fig.savefig('plot/tmp.png')
	#plt.show()


if __name__ == '__main__':
	data = runner.readDataFromFile('data/6-21/max_cp.txt')

	for i in range(0, len(METRIC)):
		# Reconfig Delta test
		rd_table = {}
		for inpt, output_by_method in data:
			if inpt.num_nodes == 64:
				rd_table[(inpt.reconfig_delta, inpt.use_eps, inpt.input_source)] = {method: getMetric(inpt, output, metric=METRIC[i]) for method, output in output_by_method.iteritems()}

		print_table = [list(vs) + [(vals_by_method[method] if method in vals_by_method else None) for method in METHODS] for vs, vals_by_method in sorted(rd_table.iteritems(), key = lambda x: (x[0][2], x[0][1], x[0][0]))]
		print METRIC[i]
		print tabulate.tabulate(print_table, headers = ['RD', 'EPS', 'Input Source'] + METHODS)
		print ''
		#plot_line(print_table, METHODS_, filename='{}-{}'.format(METRIC[i], 'vary_reconfig'), x_label='Reconfiguration Delay', y_label=METRIC_[i])
