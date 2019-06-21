
import runner
import numpy as np
import matplotlib.pyplot as plt
import tabulate

METHODS = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse', 'octopus+']
METRIC  = ['percent_packets_delivered', 'link_utilization',      'objective_value', 'percent_objective_value']
METRIC_ = ['% of Packets Deliverd',     '% of Link Utilization', 'Objective Value', '% of Objectetive Value']

def getMetric(inpt, output, metric = 'percent_objective_value'):
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
	plt.rcParams['font.size'] = 40
	fig, ax = plt.subplots(figsize=(18, 15))
	X  = [row[0] for row in table]
	num = len(table[0]) - 1                         # number of methods
	Y = []
	for i in range(1, num+1):
		y = [row[i] for row in table]
		Y.append(y)
	for i in range(0, num):
		plt.plot(X, Y[i], linestyle=linestyles[i], linewidth=4, marker=markers[i], markersize=8, label=method[i])

	plt.xticks(X)
	y_min = np.min(np.min(table, 0)[1:])*0.9
	y_max = np.max(np.max(table, 0)[1:])
	y_max = 101 if y_max <= 100 else y_max*1.02
	plt.ylim([y_min, y_max])
	plt.legend(bbox_to_anchor=(0., 1.), loc='lower left', ncol=3, fontsize=30)

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

	data = runner.readDataFromFile('data/6-20/first_run.txt')

	for i in range(0, len(METRIC)):
		# Reconfig Delta test
		rd_table = {}
		for inpt, output_by_method in data:
			if inpt.num_nodes == 64:
				rd_table[inpt.reconfig_delta] = {method: getMetric(inpt, output, metric=METRIC[i]) for method, output in output_by_method.iteritems()}

		print_table = [[rd] + [(vals_by_method[method] if method in vals_by_method else None) for method in METHODS] for rd, vals_by_method in sorted(rd_table.iteritems())]
		print tabulate.tabulate(print_table, headers = ['RD'] + METHODS)
		plot_line(print_table, METHODS, filename='{}-{}'.format(METRIC[i], 'vary_reconfig'), x_label='Reconfiguration Delay', y_label=METRIC_[i])
