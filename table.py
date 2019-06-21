
import runner

import tabulate

METHODS = ['octopus-r', 'octopus-s', 'upper-bound', 'split', 'eclipse', 'octopus+']

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

if __name__ == '__main__':
	data = runner.readDataFromFile('data/6-20/first_run.txt')

	# Reconfig Delta test
	rd_table = {}
	for inpt, output_by_method in data:
		if inpt.num_nodes == 64:
			rd_table[inpt.reconfig_delta] = {method: getMetric(inpt, output) for method, output in output_by_method.iteritems()}

	print_table = [[rd] + [(vals_by_method[method] if method in vals_by_method else None) for method in METHODS] for rd, vals_by_method in sorted(rd_table.iteritems())]

	print tabulate.tabulate(print_table, headers = ['RD'] + METHODS)
