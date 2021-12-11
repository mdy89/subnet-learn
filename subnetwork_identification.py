import os
import sys
try: import commands
except: pass
import numpy as np
import time
import math
import multiprocessing as mp
import itertools

nn = "\n"
tt = "\t"
ss = "/"
cc = ","

def main():
	#[1]
	subentwork_identifier_obj = SubnetworkIdentifier()	

	##End main

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
class SubnetworkIdentifier(object):
	def __init__(self):
		#[1]
		self.geneset_path = sys.argv[1]
		self.ppi_path = sys.argv[2]
		self.output_dir = sys.argv[3]

		#[2]
		self.retrieve_interactions()

		#[3]
		self.identify_subnetworks()
		##End init
	
	def retrieve_interactions(self):
		#[1]
		self.ppi_set = set([])

		#[2]
		for ppi_line in open(self.ppi_path):
			a, b = ppi_line.split()
			key = tuple(sorted([a, b]))
			self.ppi_set.add(key)
			##End for
		##End retrieve_interactions
	
	def identify_subnetworks(self):
		#[1]
		self.subnetworks_dir = self.output_dir + ss + "subnetworks.dir"
		make_path(self.subnetworks_dir)
		self.output_path = self.output_dir + ss + "subnetwork_list.txt"
		self.output_file = open(self.output_path, 'w')

		#[2]
		for geneset_line in open(self.geneset_path):
			geneset_id = geneset_line.strip().split(tt)[0]			
			gene_id_list = geneset_line.strip().split(tt)[1:]
			geneset_obj = Geneset()
			geneset_obj.id = geneset_id
			geneset_obj.gene_id_list = sorted(gene_id_list)
			self.identify_subnetworks_sub(geneset_obj)
			##End for

		#[3]
		self.output_file.close()
		##End identify_subnetworks
		
	def identify_subnetworks_sub(self, geneset_obj):
		#[1]
		edge_path = self.subnetworks_dir + ss + "%s_edges.txt"%geneset_obj.id
		edge_file = open(edge_path, 'w')

		#[2]
		self.gene_dic = {}
		for i, gene_id in enumerate(geneset_obj.gene_id_list):
			gene_obj = Gene()
			gene_obj.id = gene_id
			gene_obj.index = i+1
			self.gene_dic[gene_id] = gene_obj
			self.gene_dic[gene_obj.index] = gene_obj
			##End for

		#[2]
		for a, b in itertools.combinations(geneset_obj.gene_id_list, 2):
			#[2-1]
			key = a, b
			if key not in self.ppi_set:
				continue
				##End if
			#[2-2]
			c, d = [self.gene_dic[x] for x in [a, b]]
			edge_line = make_line([c.index, d.index], tt)
			edge_file.write(edge_line + nn)
			##End for

		#[3]
		edge_file.close()
		embedding_path = self.subnetworks_dir + ss + "%s_embeddings.txt"%geneset_obj.id
		log_path = self.subnetworks_dir + ss + "%s_log.txt"%geneset_obj.id
		cmd = "nohup deepwalk --input %s --output %s --representation-size 8 --seed 0 > %s"%(edge_path, embedding_path, log_path)
		os.system(cmd)

		#[4]
		embedding_file = open(embedding_path, 'r')
		embedding_file.readline()
		gene_id_list = []
		gene_embedding_arr = []
		for embedding_line in embedding_file:
			index = int(embedding_line.split()[0])
			gene_id = self.gene_dic[index].id
			embedding_vec = [float(x) for x in embedding_line.split()[1:]]
			gene_id_list.append(gene_id)
			gene_embedding_arr.append(embedding_vec)
			##End for

		#[5]
		gene_embedding_arr = np.array(gene_embedding_arr)
		clusterer_list = []
		max_clusters = int(len(gene_id_list)**0.5)+1
		for n_clusters in range(2, max_clusters):
			clusterer_obj = GaussianMixture(n_components=n_clusters, random_state=0)
			clusterer_obj.fit(gene_embedding_arr)
			clusterer_obj.score_ = -clusterer_obj.bic(gene_embedding_arr)
			clusterer_obj.n_clusters_ = n_clusters
			clusterer_obj.labels_ = clusterer_obj.predict(gene_embedding_arr)
			clusterer_list.append(clusterer_obj)
			##End for

		#[6]
		clusterer_obj = sorted(clusterer_list, key=lambda x:x.score_, reverse=True)[0]
		subnet_dic = {}
		for gene_id, label in zip(gene_id_list, clusterer_obj.labels_):
			#[6-1]
			if label not in subnet_dic:
				subnet_obj = Subnet()
				subnet_obj.gene_id_list = []
				subnet_dic[label] = subnet_obj
				##End if
			#[6-2]
			subnet_obj = subnet_dic[label]
			subnet_obj.gene_id_list.append(gene_id)
			##End for

		#[7]
		subnet_list = sorted(subnet_dic.values(), key=lambda x:len(x.gene_id_list), reverse=True)
		subnet_list = list(filter(lambda x:len(x.gene_id_list)>2, subnet_list))
		for i, subnet_obj in enumerate(subnet_list):
			subnet_obj.id = "%s_%s"%(geneset_obj.id, i+1)
			gene_id_line = make_line(sorted(subnet_obj.gene_id_list), cc)
			subnet_line = make_line([subnet_obj.id, gene_id_line], tt)
			self.output_file.write(subnet_line + nn)
			##End for
		##End identify_subnetworks_sub
	##End SubnetworkIdentifier

class Subnet(object):
	def __init__(self):
		pass
		##End init
	##End Subnet

class Gene(object):
	def __init__(self):
		pass
		##End init
	##End Gene

class Geneset(object):
	def __init__(self):
		pass
		##End init
	##End Geneset

def remove_path(path):
	if os.path.exists(path):
		try: return commands.getoutput("rm -r %s"%path)
		except: return os.system("rm -r %s"%path)
		##End if
	##End makePath

def make_path(path):
	if not os.path.exists(path):
		try: return commands.getoutput("mkdir %s"%apth)
		except: return os.system("mkdir -p %s"%path)
		##End if
	##End makePath

def make_line(token_list, sep):
	return sep.join(map(str, token_list))	
	##End makeLine

if __name__ ==  "__main__" :
	main()
	sys.exit()
	##End if
