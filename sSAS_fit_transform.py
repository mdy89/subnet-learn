"""
Title: Subnetwork representation learning for discovering network biomarkers in predicting lymph node metastasis in early oral cancer	
Author:	Minsu Kim, Sangseon Lee, Sangsoo Lim, Doh Young Lee∗, and SunKim*				

v2.0.0 (released at 04/22/2021)															
Description: a script for calculating sSAS from a given dataset
Usage: python sSAS_fit_transform.py [subnetworks_path] [train_tpm_list_path] [test_tpm_list_path] [output_dir]
"""

import os
import sys
import numpy as np
import time
import math
import multiprocessing as mp
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

nn = "\n"
tt = "\t"
ss = "/"
cc = ","

def main():
	#[1]
	sSAS_calculator_obj = sSAS_Calculator()

	##End main

import pickle, copy
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
class sSAS_Calculator(object):
	def __init__(self):
		#[1]
		self.subnetwork_path = sys.argv[1]
		self.train_path = sys.argv[2]	
		self.test_path = sys.argv[3]	
		self.output_dir = sys.argv[4]

		#[2]
		self.process_subnetworks()

		#[3]
		self.export_sSASs()
		
		##End init
	
	def export_sSASs(self):
		#[1]
		self.model_dir = self.output_dir + ss + "subnetworks.dir"

		#[2]
		train_list, test_list = [], []; index = 0
		for subnetwork_obj in self.enumerate_subnetworks():
			#[2-1]
			subnetwork_obj.index = index; index += 1
			subnetwork_obj.path = self.model_dir + ss + "subnetwork_%s.pickle"%subnetwork_obj.index
			#[2-2]
			train_obj = copy.copy(subnetwork_obj)
			test_obj = copy.copy(subnetwork_obj)
			train_obj.sample_list, test_obj.sample_list, score = pickle.load(open(subnetwork_obj.path, 'rb'))
			#[2-3]
			train_obj.score = score
			test_obj.score = score
			train_list.append(train_obj); test_list.append(test_obj)
			##End for

		#[3]
		cutoff = np.mean([x.score for x in train_list])
		subnetwork_path = self.output_dir + ss + "subnetwork_evaluated.txt"
		subnetwork_file = open(subnetwork_path, 'w')
		head_line = make_line(["#subnetwork", "score(3-fold_cv_accuracy)", "selected", "gene_components"], tt)
		subnetwork_file.write(head_line + nn)	
	
		#[4]
		for subnetwork_obj in sorted(train_list, key=lambda x:x.score, reverse=True):
			genes = make_line(sorted(subnetwork_obj.components_set), cc)
			subnetwork_line = make_line([subnetwork_obj.name, subnetwork_obj.score, subnetwork_obj.score>=cutoff, genes], tt)
			subnetwork_file.write(subnetwork_line + nn)
			##End for

		#[5]
		subnetwork_file.close()

		#[4]
		train_list = list(filter(lambda x:x.score>=cutoff, train_list))
		test_list = list(filter(lambda x:x.score>=cutoff, test_list))

		#[3]
		train_path = self.output_dir + ss + "train_sSAS.pickle"
		train_file = open(train_path, 'wb')
		pickle.dump(train_list, train_file, protocol=pickle.HIGHEST_PROTOCOL)
		train_file.close()

		#[4]
		test_path = self.output_dir + ss + "test_sSAS.pickle"
		test_file = open(test_path, 'wb')
		pickle.dump(test_list, test_file, protocol=pickle.HIGHEST_PROTOCOL)
		test_file.close()

		##End export_sSASs
	
	def process_subnetworks(self):
		#[1]
		self.model_dir = self.output_dir + ss + "subnetworks.dir"
		remove_path(self.model_dir); make_path(self.model_dir)

		#[2]
		index = 0; pool_obj = mp.Pool(processes=mp.cpu_count())
		for subnetwork_obj in self.enumerate_subnetworks():
			subnetwork_obj.index = index; index += 1
			r=pool_obj.apply_async(self.process_subnetworks_sub, args=tuple([subnetwork_obj]))
			##End for

		#[3]
		pool_obj.close(); pool_obj.join()
		##End process_subnetworks
	
	def process_subnetworks_sub(self, subnetwork_obj):
		#[1]
		self.retrieve_tpms(subnetwork_obj)

		#[2]
		self.calculate_sSASs(subnetwork_obj)

		#[3]
		self.fit_sSASs(subnetwork_obj)

		#[3]
		self.trasnsform_sSAS(subnetwork_obj)
		##End process_subnetworks_sub
	
	def trasnsform_sSAS(self, subnetwork_obj):
		#[1]
		train_x_arr = subnetwork_obj.imputer_obj.fit_transform(np.array([x.feature_vec for x in self.train_list]))
		train_y_arr = self.extract_sample_labels(self.train_list)
		test_x_arr = subnetwork_obj.imputer_obj.fit_transform(np.array([x.feature_vec for x in self.test_list]))
		test_y_arr = self.extract_sample_labels(self.test_list)

		#[2]
		sSAS_path = self.model_dir + ss + "subnetwork_%s.pickle"%subnetwork_obj.index
		train_z_arr = subnetwork_obj.predictor_obj.predict_proba(train_x_arr)
		test_z_arr = subnetwork_obj.predictor_obj.predict_proba(test_x_arr)

		#[3]
		for index, sample_obj in enumerate(self.train_list):
			sample_obj.index = index
			sample_obj.feature_vec = train_z_arr[index]
			##End for

		#[4]
		for index, sample_obj in enumerate(self.test_list):
			sample_obj.index = index
			sample_obj.feature_vec = test_z_arr[index]
			##End for
		
		#[5]	
		sSAS_file = open(sSAS_path, 'wb')	
		pickle.dump([self.train_list, self.test_list, subnetwork_obj.score], sSAS_file, protocol=pickle.HIGHEST_PROTOCOL)
		sSAS_file.close()
		##End trasnsform_sSAS
	
	def fit_sSASs(self, subnetwork_obj):
		#[1]
		subnetwork_obj.imputer_obj = SimpleImputer()
		train_x_arr = subnetwork_obj.imputer_obj.fit_transform(np.array([x.feature_vec for x in self.train_list]))
		train_y_arr = self.extract_sample_labels(self.train_list)
		subnetwork_obj.score = 0

		#[2]
		subnetwork_obj.predictor_obj = LogisticRegression(random_state=0,class_weight="balanced",fit_intercept=False,multi_class="ovr")
		subnetwork_obj.predictor_obj.fit(train_x_arr, train_y_arr)
		subnetwork_obj.score = []

		#[3]
		evaluator_obj = LogisticRegression(random_state=0,class_weight="balanced",fit_intercept=False,multi_class="ovr")
		for i, j in StratifiedKFold(n_splits=3).split(train_x_arr, train_y_arr):
			xi, yi = train_x_arr[i], train_y_arr[i]
			xj, yj = train_x_arr[j], train_y_arr[j]
			evaluator_obj.fit(xi, yi)
			subnetwork_obj.score.append(evaluator_obj.score(xj, yj))
			##End for

		#[4]
		subnetwork_obj.score = np.mean(subnetwork_obj.score)
		##End fit_sSASs
	
	def calculate_sSASs(self, subnetwork_obj):
		#[1]
		gene_name_list = sorted(subnetwork_obj.expressed_set)

		#[2]
		for sample_obj in self.sample_list:
			#[2-1]
			tpm_list = [sample_obj.tpm_dic[x] for x in gene_name_list]
			sample_obj.feature_vec = []
			#[2-2]
			for i, j in itertools.combinations(tpm_list, 2):
				divider = i+j+np.finfo(float).eps
				ii = (i**2)/divider
				jj = (j**2)/divider
				ij = (i*j)/divider
				sample_obj.feature_vec += [ii, ij, jj]	
				##End for
			##End for

		#[3]
		self.train_list = list(filter(lambda x:x.dataset=="train", self.sample_list))
		self.test_list = list(filter(lambda x:x.dataset=="test", self.sample_list))
		##End calculate_sSASs

	def extract_sample_labels(self, sample_list):
		#[1]
		sample_label_list = [x.label for x in sample_list]
		label_list = sorted(set(sample_label_list)); label_size = len(label_list)
		sample_label_list = [label_list.index(x) for x in sample_label_list]
		
		#[2]
		return np.array(sample_label_list)
		##End extract_sample_labels

	def retrieve_tpms(self, subnetwork_obj):
		#[1]
		self.sample_dic = {}

		#[2]
		for dataset in ["train", "test"]:
			for input_line in open(self.__dict__["%s_path"%dataset]):
				#[2-1]
				if input_line[0]=="#":
					continue
					##End if
				#[2-2]
				sample_name, label, gene, tpm = input_line.split()
				if gene not in subnetwork_obj.components_set:
					continue
					##End if
				#[2-3]
				try: sample_obj = self.sample_dic[sample_name]
				except:
					#[2-2-1]
					sample_obj = Sample()
					sample_obj.name = sample_name
					sample_obj.label = label
					sample_obj.tpm_dic = {}
					sample_obj.dataset = dataset
					for gene_name in subnetwork_obj.components_set:
						sample_obj.tpm_dic[gene_name] = float("nan")
						##End for
					#[2-2-2]
					self.sample_dic[sample_name] = sample_obj
					##End try-except
				#[2-3]
				subnetwork_obj.expressed_set.add(gene)
				sample_obj.tpm_dic[gene] = float(tpm)
				##End for		
			##End for

		#[3]
		self.sample_list = sorted(self.sample_dic.values(), key=lambda x:x.name)
		##End retrieve_tpms
	
	def enumerate_subnetworks(self):
		for subnetwork_line in open(self.subnetwork_path):
			name, components = subnetwork_line.split()
			subnetwork_obj = Subnetwork()
			subnetwork_obj.name = name
			subnetwork_obj.components_set = set(components.strip().split(cc))
			subnetwork_obj.expressed_set = set([])
			yield subnetwork_obj
			##End for
		##End enumerate_subnetworks
	##End sSAS_Calculator

class Sample(object):
	def __init__(self):
		pass
		##End init
	##End Sample

class Subnetwork(object):
	def __init__(self):
		pass
		##End init
	##End Subnetwork

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
