
import urllib2
import urllib
import subprocess
from collections import defaultdict

import re

def getListOfZipFiles(filename):
	f = open(filename)
	zips = []
	for line in f:
		zips.append(line[:-1])
	return zips

def downloadZipFiles(zip_url, i):
	filename = 'raw_data/tmp_facebook_data_' + str(i) + '.bz2'
	try:
		urllib.urlretrieve(zip_url, filename)
	except Exception, e:
		return None
	return filename

def parseFacebookFile(in_filename, pods, racks, pod_demand_matrix, rack_demand_matrix):
	fin = open(in_filename, 'r')

	prev_pod_len = len(pods)
	prev_rack_len = len(racks)
	for line in fin:
		time, packet_len, \
			src_ip, dst_ip, \
			src_port, dst_port, \
			protocol, \
			src_host_prefix, dst_host_prefix, \
			src_rack, dst_rack, \
			src_pod, dst_pod, \
			intercluster, interdatacenter = line.split()

		time = int(time)	
		packet_len = int(packet_len)

		if intercluster == '0' and interdatacenter == '0':
			racks.add(src_rack)
			racks.add(dst_rack)

			pods.add(src_pod)
			pods.add(dst_pod)

			rack_demand_matrix[(src_rack, dst_rack)] += packet_len
			pod_demand_matrix[(src_pod, dst_pod)] += packet_len

	print 'Added', len(racks) - prev_rack_len, 'racks'
	print 'Added', len(pods) - prev_pod_len, 'pods'

	fin.close()

def outputDemandMatrix(out_filename, pods, racks, pod_demand_matrix, rack_demand_matrix):
	f = open(out_filename, 'w')

	f.write('Rack_Demand_Matrix\n')
	used_rack_pairs = set()
	for r1 in racks:
		for r2 in racks:
			if (r1, r2) in used_rack_pairs:
				continue
			used_rack_pairs.add((r1, r2))
			used_rack_pairs.add((r2, r1))

			f.write(r1 + ' ' + r2 + ' ' + str(rack_demand_matrix[(r1, r2)] + rack_demand_matrix[(r2, r1)]) + '\n')

	f.write('\nPod_Demand_Matrix\n')
	used_pod_pairs = set()
	for p1 in pods:
		for p2 in pods:
			if (p1, p2) in used_pod_pairs:
				continue
			used_pod_pairs.add((p1, p2))
			used_pod_pairs.add((p2, p1))

			f.write(p1 + ' ' + p2 + ' ' + str(pod_demand_matrix[(p1, p2)] + pod_demand_matrix[(p2, p1)]) + '\n')

class SimpFlow:
	def __init__(self):
		self.size = 0
		self.num_packets = 0
		self.start_time = None
		self.end_time = None

	def adjustTime(self, t):
		if self.start_time == None or self.start_time > t:
			self.start_time = t
		if self.end_time == None or self.end_time < t:
			self.end_time = t

	def __str__(self):
		return '(' + str(self.size) + ', ' + str(self.num_packets) + ')'

def parseFacebookFileForFlows(in_filename, new_flow_file, old_flow_file, only_intracluster = True):
	fin = open(in_filename, 'r')

	# Key is (a_ip, b_ip, a_port, b_port) [where a_ip is min(src_ip, dst_ip) and b_ip is the other], value is (sum of packet lens, number of packets)
	flows = defaultdict(SimpFlow)
	for line in fin:
		time, packet_len, \
			src_ip, dst_ip, \
			src_port, dst_port, \
			protocol, \
			src_host_prefix, dst_host_prefix, \
			src_rack, dst_rack, \
			src_pod, dst_pod, \
			intercluster, interdatacenter = line.split()

		time = int(time)	
		packet_len = int(packet_len)

		if (not only_intracluster or (intercluster == '0' and interdatacenter == '0')) and protocol == '6':
			a_ip, a_port = ((src_ip, src_port) if src_ip <= dst_ip else (dst_ip, dst_port))
			b_ip, b_port = ((dst_ip, dst_port) if src_ip <= dst_ip else (src_ip, src_port))

			sf = flows[(a_ip, b_ip, a_port, b_port)]
			sf.size += packet_len
			sf.num_packets += 1
			sf.adjustTime(time)

	fin.close()
	print 'Found', len(flows), 'flows'

	fold = open(old_flow_file, 'r')
	fnew = open(new_flow_file, 'w')

	flow_re = '(?P<a_ip>\w+) (?P<b_ip>\w+) (?P<a_port>\w+) (?P<b_port>\w+) (?P<flow_size>\d+) (?P<num_packets>\d+) (?P<start_time>\d+) (?P<end_time>\d+)\n'

	for line in fold:
		flow_match = re.match(flow_re, line)
		if flow_match == None:
			raise Exception('Invalid "old_flow_file"')

		flow_id = tuple(flow_match.group(k) for k in ('a_ip', 'b_ip', 'a_port', 'b_port'))
		flow_size, num_packets, start_time, end_time = map(int, (flow_match.group(k) for k in ('flow_size', 'num_packets', 'start_time', 'end_time')))

		sf = flows.pop(flow_id, None)
		if sf != None:
			fnew.write(' '.join(flow_id) + ' ' + str(sf.size + flow_size) + ' ' + str(sf.num_packets + num_packets) + ' ' + str(min(sf.start_time, start_time)) + ' ' + str(max(sf.end_time, end_time)) + '\n')
		else:
			fnew.write(line)

	print 'Found', len(flows), 'new flows'
	for flow_id, sf in flows.iteritems():
		fnew.write(' '.join(flow_id) + ' ' + str(sf.size) + ' ' + str(sf.num_packets) + ' ' + str(sf.start_time) + ' ' + str(sf.end_time) + '\n')

def downloadAllData(in_filenames, out_filename, output_type = 'demand_matrix'):
	facebook_sources = in_filenames
	
	all_data = []
	for filename in facebook_sources:
		zip_file_urls = getListOfZipFiles(filename)
		for zip_url in zip_file_urls:
			all_data.append(zip_url)

	zip_files = []

	# Set of unique entities
	if output_type == 'demand_matrix':
		pods = set()
		racks = set()

		# Key is is (src_entity, dst_entity) and value is the total amount of data transfered between them
		pod_demand_matrix = defaultdict(int)
		rack_demand_matrix = defaultdict(int)
	elif output_type == 'flow_size':
		flow_file1 = 'facebook_flows1.txt'
		flow_file2 = 'facebook_flows2.txt'

		# Clears the files
		open(flow_file1, 'w').close()
		open(flow_file2, 'w').close()

	failed_files = []
	last_file = ''
	for i, zip_file_url in enumerate(all_data):
		if output_type == 'flow_size':
			if i % 2:
				new_flow_file = flow_file1
				old_flow_file = flow_file2
			else:
				new_flow_file = flow_file2
				old_flow_file = flow_file1

		print 'Downloading Zip', i + 1, 'of', len(all_data)
		zip_file = downloadZipFiles(zip_file_url, i + 1)

		if zip_file == None:
			failed_files.append(zip_file_url)
			continue
		
		print 'Decompressing Zip', i + 1, 'of', len(all_data)
		decomp = subprocess.Popen(['bunzip2', zip_file])
		decomp.wait()

		new_filename = zip_file[:zip_file.rfind('.')]
		try:
			print 'Parsing File', i + 1, 'of', len(all_data)
			if output_type == 'demand_matrix':
				parseFacebookFile(new_filename, pods, racks, pod_demand_matrix, rack_demand_matrix)
			if output_type == 'flow_size':
				flows = parseFacebookFileForFlows(new_filename, new_flow_file, old_flow_file)
				last_file = new_flow_file
		except IOError:
			failed_files.append(zip_file_url)

		print 'Deleting File', i + 1, 'of', len(all_data)
		delete = subprocess.Popen(['rm', new_filename])
		delete.wait()

	print 'Failed Files:', ' '.join(failed_files)

	if output_type == 'demand_matrix':
		outputDemandMatrix(out_filename, pods, racks, pod_demand_matrix, rack_demand_matrix)
	elif output_type == 'flow_size':
		move = subprocess.Popen(['mv', last_file, out_filename])
		move.wait()
		if last_file == flow_file1:
			delete = subprocess.Popen(['rm', flow_file2])
			delete.wait()
		elif last_file == flow_file2:
			delete = subprocess.Popen(['rm', flow_file1])
			delete.wait()

if __name__ == '__main__':
	# downloadAllData(['raw_data/facebook_cluster_A.txt'], 'facebook_cluster_A.raw_flow_size', output_type = 'flow_size')
	# downloadAllData(['raw_data/facebook_cluster_C.txt'], 'facebook_cluster_C.raw_flow_size', output_type = 'flow_size')
	# downloadAllData(['raw_data/facebook_cluster_C.txt'], 'facebook_cluster_C.demand_matrix', output_type = 'demand_matrix')
	# downloadAllData(['raw_data/facebook_cluster_B.txt'], 'facebook_cluster_B.demand_matrix')

	downloadAllData(['facebook_cluster_B.txt'], 'facebook_cluster_B.demand_matrix', output_type = 'demand_matrix')