
import time
from collections import defaultdict

import tabulate

class Profiler:
	event_stack = []
	prev_t = 0.0
	in_time_per_event = defaultdict(float)
	total_time_per_event = defaultdict(float)
	num_event = defaultdict(int)

	use_profiler = False

	@staticmethod
	def turnOn():
		Profiler.use_profiler = True

	@staticmethod
	def start(id_str):
		if Profiler.use_profiler:
			cur_t = time.time()
			delta_t = cur_t - Profiler.prev_t

			# Update total_time (this can be done only at the end?)
			for e_id, x in Profiler.event_stack:
				Profiler.total_time_per_event[e_id] += delta_t

			# Update in_time
			if len(Profiler.event_stack) > 0:
				Profiler.in_time_per_event[Profiler.event_stack[-1][0]] += delta_t

			Profiler.event_stack.append((id_str, cur_t))

			Profiler.prev_t = cur_t

	@staticmethod
	def end(id_str):
		if Profiler.use_profiler:
			cur_t = time.time()
			delta_t = cur_t - Profiler.prev_t

			if len(Profiler.event_stack) == 0:
				raise Exception('Problem with Profiler!!!! New Event: ' + id_str)

			last_event, last_time = Profiler.event_stack[-1]
			if last_event != id_str:
				raise Exception('Problem with Profiler!!! Last Event: ' + last_event + ', New Event: ' + id_str)

			# Update total_time (this can be done only at the end?)
			for e_id, x in Profiler.event_stack:
				Profiler.total_time_per_event[e_id] += delta_t

			# Update in_time
			Profiler.in_time_per_event[Profiler.event_stack[-1][0]] += delta_t

			Profiler.event_stack.pop()
			Profiler.num_event[id_str] += 1

			Profiler.prev_t = cur_t

	@staticmethod
	def stats():
		if Profiler.use_profiler:
			stats_per_event = []
			overall_time = 0.0
			for event_id in Profiler.total_time_per_event:
				if Profiler.num_event[event_id] == 0:
					continue
				total_time = Profiler.total_time_per_event[event_id]
				num_times = Profiler.num_event[event_id]
				avg_total = total_time / num_times
				total_time_in = Profiler.in_time_per_event[event_id]
				avg_time_in = total_time_in / num_times

				overall_time += total_time_in

				stats_per_event.append((total_time_in, total_time, avg_total, avg_time_in, num_times, event_id))
			stats_per_event.sort()
			stats_per_event.reverse()
			print '\n\n' + '-' * 100
			headers = ['Total', 'Total_in', 'Percent_in', 'Avg', 'Avg_in', 'Num_called', 'Function']

			values = [[total_time, total_time_in, total_time_in / overall_time * 100.0, avg_total, avg_time_in, num_times, event_id] for total_time_in, total_time, avg_total, avg_time_in, num_times, event_id in stats_per_event]

			print tabulate.tabulate(values, headers = headers)
				