import os
import sys
import commands
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
	ARGUMENT_handler = Gen_ARGUMENT_handler()		

	#[2]
	TUMOR_encoder = Gen_TUMOR_encoder(ARGUMENT_handler)		

	##End main

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from scipy.stats import ranksums
from scipy.optimize import nnls
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
class Gen_TUMOR_encoder(object):
	def __init__(self, ARGUMENT_handler):
		#[1]
		self.ARGUMENT_assign_method(ARGUMENT_handler)

		#[2]
		self.NETWORK_register_method()

		#[3]
		self.SAMPLE_register_method()

		#[4]
		self.NETWORK_embedding_method()

		#[5]
		self.TSNE_embedding_method()

		#[6]
		self.MODEL_search_method()

		##End init

	def TSNE_embedding_method(self):
		#[1]
		EMBEDDING_path = self.OUT_dir + ss + "EMBEDDING.txt"
		EMBEDDING_file = open(EMBEDDING_path, 'r'); EMBEDDING_file.readline()
		
		#[2]
		for EMBEDDING_line in EMBEDDING_file:
			GENE_index = int(EMBEDDING_line.split()[0])-1
			GENE_obj = self.GENE_list[GENE_index]
			GENE_obj.EMBEDDING_vec = map(float, EMBEDDING_line.split()[1:])
			##End for

		#[3]
		self.GENE_EMBEDDING_arr = np.array(map(lambda x:x.EMBEDDING_vec, self.GENE_list))

		#[4]
		ENCODER_obj = TSNE(n_components=2, random_state=0)
		ENCODER_obj.fit(self.GENE_EMBEDDING_arr)

		#[5]
		RAW_scatter_path = self.OUT_dir + ss + "RAW_SCATTER.png"
		X_list = map(lambda x:x[0], ENCODER_obj.embedding_)
		Y_list = map(lambda x:x[1], ENCODER_obj.embedding_)
		plt.scatter(X_list, Y_list, marker='.', s=25, linewidths=0)
		plt.savefig(RAW_scatter_path); plt.close()

		#[6]
		EMBEDDING_path = self.OUT_dir + ss + "TSNE_EMBEDDING.txt"
		EMBEDDING_file = open(EMBEDDING_path, 'w')
		for GENE_obj, EMBEDDING_vec in zip(self.GENE_list, ENCODER_obj.embedding_):
			EMBEDDING_line = makeLine([GENE_obj.INDEX]+list(EMBEDDING_vec), tt)
			EMBEDDING_file.write(EMBEDDING_line + nn)	
			##End for

		#[7]
		EMBEDDING_file.close()
		##End TSNE_embedding_method
	
	def MODEL_search_method(self):
		#[1]
		self.EMBEDDING_register_method()

		#[2]
		self.MODEL_dir = self.OUT_dir + ss + "MODEL.Dir"
		removePath(self.MODEL_dir); makePath(self.MODEL_dir)
		K_ranges = range(10,201)
		M_ranges = range(2)
#		C_ranges = ["KMeans", "Hierarchical"]
		C_ranges = ["KMeans"]

		#[3]
		self.MODEL_path = self.OUT_dir + ss + "MODEL_RESULT.txt"
		removePath(self.MODEL_path)

		#[4]
		self.BEST = -1
		for K, M, C in itertools.product(K_ranges, M_ranges, C_ranges):
			MODEL_obj = Gen_MODEL_object(); MODEL_obj.K = K; MODEL_obj.M = M; MODEL_obj.C = C
			self.MODEL_evaluate_method(MODEL_obj)
			##End for
		##End MODEL_search_method
	
	def DISTANCE_calculate_method(self, MODEL_obj):
		#[1]
		self.DISTANCE_dir = self.OUT_dir + ss + "DISTANCE.Dir"
		removePath(self.DISTANCE_dir); makePath(self.DISTANCE_dir)

		#[2]
		self.PROC_list = []; self.SEM_obj = mp.Semaphore(self.PN_int)
		for A, B in itertools.combinations(range(self.SAMPLE_GENE_arr.shape[0]),2):
			self.SEM_obj.acquire(); ARGV = tuple([A, B, MODEL_obj])
			PROC_obj = mp.Process(target=self.DISTANCE_calculate_sub, args=ARGV)
			PROC_obj.start(); self.PROC_list.append(PROC_obj)
			##End for

		#[3]
		for PROC_obj in self.PROC_list:
			PROC_obj.join()
			##End for

		#[4]
		self.DISTANCE_dic = {}
		for KEY in itertools.combinations(range(self.SAMPLE_GENE_arr.shape[0]), 2):
			A, B = KEY
			DISTANCE_path = self.DISTANCE_dir + ss + "DISTANCE_%s_%s.txt"%(A, B)
			DISTANCE_vec = map(float, open(DISTANCE_path).read().split())
			self.DISTANCE_dic[KEY] = DISTANCE_vec
			##End for
		##End DISTANCE_calculate_method

	def DISTANCE_calculate_sub(self, A, B, MODEL_obj):
		#[1]
		A_raw_vec = self.SAMPLE_GENE_arr[A]
		B_raw_vec = self.SAMPLE_GENE_arr[B]

		#[2]
		DISTANCE_path = self.DISTANCE_dir + ss + "DISTANCE_%s_%s.txt"%(A, B)
		DISTANCE_file = open(DISTANCE_path, 'w')
		for CLUSTER_obj in self.CLUSTER_list:
			INDEX_list = map(lambda x:x.INDEX, CLUSTER_obj.GENE_list)
			A_vec = A_raw_vec[INDEX_list]
			B_vec = B_raw_vec[INDEX_list]
			if MODEL_obj.M == 0:
#				DIST = euclidean(A_vec, B_vec)
				DIST = ranksums(A_vec, B_vec)[0]
			else:
				A_rank_vec = list(rankdata(list(A_vec)+list(B_vec)))[:len(INDEX_list)]
				B_rank_vec = list(rankdata(list(B_vec)+list(A_vec)))[:len(INDEX_list)]
				DIST = abs(np.mean(A_rank_vec) - np.mean(B_rank_vec))/len(INDEX_list)
				##End if-else
			DISTANCE_file.write(str(DIST)+nn)
			##End for
				
		#[2]
		DISTANCE_file.close(); self.SEM_obj.release()
		##End DISTANCE_calculate_method

	def MODEL_evaluate_method(self, MODEL_obj):
		#[1]
		self.CLUSTER_generate_method(MODEL_obj)

		#[2]
		self.DISTANCE_calculate_method(MODEL_obj)

		#[3]
#		A_ranges = ["relu","logistic","tanh","identity"]
		A_ranges = ["logistic"]
		L_ranges = self.LAYER_enumerate_method(MODEL_obj) 
		L_ranges = [[2]]
		T_ranges = map(lambda x:"".join(x), itertools.product(["0"],["2"]))

		#[4]
		for A, L, T in itertools.product(A_ranges, L_ranges, T_ranges):
			MODEL_obj.A = A; MODEL_obj.L = L; MODEL_obj.T = T
			self.MODEL_evaluate_sub(MODEL_obj)
			##End for
		##End MODEL_evaluate_method
	
	def LAYER_enumerate_method(self, MODEL_obj):
		#[1]
		self.LABEL_arr = np.array(open(self.LABEL_path).read().split())
		K = len(self.LABEL_arr)
		POOL = filter(lambda y:y<K, map(lambda x:2**x, range(2,11)))
	
		#[2]
		L_list = []
		for DEPTH in range(len(POOL)+1):
			for TAIL in itertools.combinations(POOL, DEPTH):
				TAIL = list(TAIL); L = TAIL[::-1] + [2] + TAIL 
				L_list.append(L)
				##End for
			##End for 
		
		#[3]
		return L_list
		##End LAYER_enumerate_method 
	
	def MODEL_evaluate_sub(self, MODEL_obj):
		#[1]
		self.LABEL_arr = np.array(open(self.LABEL_path).read().split())
		self.LABEL_arr = np.array(map(lambda x:{"N0":"N0","N1":"N1","N2":"N1"}[x], self.LABEL_arr))
		MODEL_obj.PREFIX = makeLine([MODEL_obj.C, MODEL_obj.K, MODEL_obj.A, MODEL_obj.T, MODEL_obj.M]+MODEL_obj.L, "_")
		
		#[2]
		BATCH_list = []
		for TRAIN, TEST in StratifiedKFold(n_splits=self.CV_int).split(self.LABEL_arr, self.LABEL_arr):
			BATCH_obj = Gen_BATCH_object(); BATCH_obj.MODEL_obj = MODEL_obj
			BATCH_obj.TRAIN = TRAIN; BATCH_obj.TEST = TEST
			self.BATCH_process_method(BATCH_obj); BATCH_list.append(BATCH_obj)
			##End for

		#[3]
		BATCH_obj = Gen_BATCH_object(); BATCH_obj.MODEL_obj = MODEL_obj
		BATCH_obj.TRAIN = range(len(self.LABEL_arr)); BATCH_obj.TEST = []
		self.KERNEL_train_method(BATCH_obj)
		self.ENCODER_train_method(BATCH_obj)
		self.CLASSIFIER_train_method(BATCH_obj)
		TRAIN_SCORE = BATCH_obj.TRAIN_SCORE
		TEST_SCORE = np.mean(map(lambda x:x.TEST_SCORE, BATCH_list))
		try: self.BEST
		except: self.BEST = TEST_SCORE

		#[4]
		if TRAIN_SCORE >= TEST_SCORE >= self.BEST:
			#[4-1]
			MODEL_SCATTER_path = self.MODEL_dir + ss + "MODEL_%s.CLUSTER.png"%(MODEL_obj.K)
			COLOR_list = cm.rainbow(map(lambda x:x.INDEX/float(len(self.CLUSTER_list)-1), self.CLUSTER_list))
			X_list = map(lambda x:x.TSNE_EMBEDDING_vec[0], self.GENE_list)
			Y_list = map(lambda x:x.TSNE_EMBEDDING_vec[1], self.GENE_list)
			C_list = map(lambda x:COLOR_list[x.CLUSTER], self.GENE_list)
			plt.scatter(X_list, Y_list, c=C_list, marker='.', s=25, linewidths=0)
			plt.savefig(MODEL_SCATTER_path); plt.close()
			#[4-2]
			X_list = map(lambda x:x[0], self.TRAIN_EMBEDDING_arr)
			Y_list = map(lambda x:x[1], self.TRAIN_EMBEDDING_arr)
			C_list = map(lambda x:["skyblue","orange","pink"][int(x[-1])], self.LABEL_arr)
			plt.scatter(X_list, Y_list, c=C_list, marker='.', s=300, linewidths=0)
			plt.title("OVERALL(%.2f%%)"%(BATCH_obj.TRAIN_SCORE*100))
			plt.savefig(self.MODEL_dir+ss+"MODEL_%s.OVERALL.png"%MODEL_obj.PREFIX); plt.close()
			#[4-3]
			try: self.KERNEL_obj.coef_[0][0]; W_arr = self.KERNEL_obj.coef_[0]
			except: W_arr = self.KERNEL_obj.coef_
			for CLUSTER_obj, CLUSTER_weight, in zip(self.CLUSTER_list, W_arr):
				CLUSTER_obj.W = CLUSTER_weight
				##End for
			#[4-4]
			MODEL_SCATTER_path = self.MODEL_dir + ss + "MODEL_%s.WEIGHTED.png"%MODEL_obj.PREFIX
			X_list = map(lambda x:x.TSNE_EMBEDDING_vec[0], self.GENE_list)
			Y_list = map(lambda x:x.TSNE_EMBEDDING_vec[1], self.GENE_list)
			C_list = map(lambda x:self.CLUSTER_dic[x.CLUSTER].W, self.GENE_list)
			plt.scatter(X_list, Y_list, c=C_list, marker='.', s=25, linewidths=0, cmap=plt.cm.coolwarm)
			plt.title("PIN Subnetwork Feature Map (%s Clusters)"%MODEL_obj.K); plt.colorbar(); plt.savefig(MODEL_SCATTER_path); plt.close()
			#[4-5]
			CLUSTER_path = self.MODEL_dir + ss + "MODEL_%s.CLUSTER_RANKED.txt"%MODEL_obj.PREFIX
			CLUSTER_file = open(CLUSTER_path, 'w')
			for CLUSTER_obj in sorted(self.CLUSTER_list, key=lambda x:x.W, reverse=True):
				G = makeLine(sorted(map(lambda x:x.NAME, CLUSTER_obj.GENE_list)), cc)
				CLUSTER_line = makeLine(["CLUSTER_%s"%CLUSTER_obj.INDEX, CLUSTER_obj.W, G], tt)
				CLUSTER_file.write(CLUSTER_line + nn)
				##End for
			#[4-6]
			CLUSTER_file.close()
			#[4-7]
			FEATURE_path = self.MODEL_dir + ss + "MODEL_%s.FEATURE.txt"%MODEL_obj.PREFIX
			FEATURE_file = open(FEATURE_path, 'w'); W_max = max(map(lambda x:x.W, self.CLUSTER_list)); W_mid = W_max/2
			for CLUSTER_obj in filter(lambda x:x.W>W_mid, self.CLUSTER_list):
				for GENE_obj in CLUSTER_obj.GENE_list:
					FEATURE_file.write(GENE_obj.NAME+nn)
					##End for
				##End for
			#[4-8]
			FEATURE_file.close(); self.BEST = TEST_SCORE
			##End if

		#[5]
		self.MODEL_file = open(self.MODEL_path, 'a')
		MODEL_line = makeLine([MODEL_obj.PREFIX, TRAIN_SCORE, TEST_SCORE], tt)
		self.MODEL_file.write(MODEL_line + nn); self.MODEL_file.close()
		##End MODEL_evaluate_sub
	
	def BATCH_process_method(self, BATCH_obj):
		#[1]
		self.KERNEL_train_method(BATCH_obj)

		#[2]
		self.ENCODER_train_method(BATCH_obj)

		#[3]
		self.CLASSIFIER_train_method(BATCH_obj)

		#[4]
		self.BATCH_test_method(BATCH_obj)

		#[5]
		A = map(lambda x:x[0],self.TRAIN_EMBEDDING_arr)
		B = map(lambda x:x[0],self.TEST_EMBEDDING_arr)
		L, H = min(A+B), max(A+B)
		BATCH_obj.X = L-abs(L-H)/10., H+abs(L-H)/10.

		#[6]
		A = map(lambda x:x[1],self.TRAIN_EMBEDDING_arr)
		B = map(lambda x:x[1],self.TEST_EMBEDDING_arr)
		L, H = min(A+B), max(A+B)
		BATCH_obj.Y = L-abs(L-H)/10., H+abs(L-H)/10.
		##End BATCH_process_method
	
	def BATCH_test_method(self, BATCH_obj):
		#[1]
		SAMPLE_SAMPLE_arr = []

		#[2]
		for A in BATCH_obj.TEST:
			#[2-1]
			A_DISTANCE_arr = []
			#[2-2]
			for B in BATCH_obj.TRAIN:
				A_DISTANCE_vec = self.DISTANCE_measure_method(A, B)			
				A_DISTANCE_arr.append(A_DISTANCE_vec)
				##End for
			#[2-3]
			A_DISTANCE_arr = np.array(A_DISTANCE_arr)
			SAMPLE_SAMPLE_vec = self.KERNEL_obj.predict(A_DISTANCE_arr)
			SAMPLE_SAMPLE_arr.append(SAMPLE_SAMPLE_vec)	
			##End for

		#[3]
		W_list = self.ENCODER_obj.coefs_; B_list = self.ENCODER_obj.intercepts_ 
		self.TEST_EMBEDDING_arr = []; self.I = len(BATCH_obj.MODEL_obj.L)/2
		for SAMPLE_GENE_vec in SAMPLE_SAMPLE_arr:
			for W_arr, B_arr in zip(W_list[:self.I+1], B_list[:self.I+1]): 
				SAMPLE_GENE_vec = self.FUNC(np.dot(SAMPLE_GENE_vec, W_arr) + B_arr)
				##End for
			self.TEST_EMBEDDING_arr.append(SAMPLE_GENE_vec)
			##End for

		#[6]
		self.TEST_EMBEDDING_arr = np.array(self.TEST_EMBEDDING_arr)
		self.TEST_LABEL_arr = self.LABEL_arr[BATCH_obj.TEST]
		
		#[7]
		BATCH_obj.TEST_SCORE = self.CLASSIFIER_obj.score(self.TEST_EMBEDDING_arr, self.TEST_LABEL_arr)
		BATCH_obj.TEST_EMBEDDING_arr = self.TEST_EMBEDDING_arr
		##End BATCH_test_method
	
	def CLASSIFIER_train_method(self, BATCH_obj):
		#[1]
		self.CLASSIFIER_obj = NearestCentroid()
		self.CLASSIFIER_obj.fit(self.TRAIN_EMBEDDING_arr, self.TRAIN_LABEL_arr)

		#[2]
		BATCH_obj.TRAIN_SCORE = self.CLASSIFIER_obj.score(self.TRAIN_EMBEDDING_arr, self.TRAIN_LABEL_arr)
		##End CLASSIFIER_train_method
	
	def ENCODER_train_method(self, BATCH_obj):
		#[1]
		self.ENCODER_obj = MLPRegressor(random_state=0,hidden_layer_sizes=BATCH_obj.MODEL_obj.L,activation=BATCH_obj.MODEL_obj.A,max_iter=10000,early_stopping=True)

		#[2]
		SAMPLE_SAMPLE_arr = []
		for A in BATCH_obj.TRAIN:
			#[2-1]
			A_DISTANCE_arr = []
			#[2-2]
			for B in BATCH_obj.TRAIN:
				A_DISTANCE_vec = self.DISTANCE_measure_method(A, B)			
				A_DISTANCE_arr.append(A_DISTANCE_vec)
				##End for
			#[2-3]
			A_DISTANCE_arr = np.array(A_DISTANCE_arr)
			SAMPLE_SAMPLE_vec = self.KERNEL_obj.predict(A_DISTANCE_arr)
			SAMPLE_SAMPLE_arr.append(SAMPLE_SAMPLE_vec)	
			##End for

		#[3]
		self.SAMPLE_SAMPLE_arr = np.array(SAMPLE_SAMPLE_arr)
		self.ENCODER_obj.fit(self.SAMPLE_SAMPLE_arr, self.SAMPLE_SAMPLE_arr)

		#[4]		
		self.FUNC_dic = {
			"identity": (lambda x:x),
			"logistic": (lambda x:1/(1+np.exp(-x))),
			"relu": (lambda x:map(lambda y:max(0,y),x)),
			"tanh": np.tanh
		}
		self.FUNC = self.FUNC_dic[BATCH_obj.MODEL_obj.A]

		#[5]
		W_list = self.ENCODER_obj.coefs_; B_list = self.ENCODER_obj.intercepts_ 
		self.TRAIN_EMBEDDING_arr = []; self.I = len(BATCH_obj.MODEL_obj.L)/2
		for SAMPLE_GENE_vec in self.SAMPLE_SAMPLE_arr:
			for W_arr, B_arr in zip(W_list[:self.I+1], B_list[:self.I+1]): 
				SAMPLE_GENE_vec = self.FUNC(np.dot(SAMPLE_GENE_vec, W_arr) + B_arr)
				##End for
			self.TRAIN_EMBEDDING_arr.append(SAMPLE_GENE_vec)
			##End for

		#[6]
		self.TRAIN_EMBEDDING_arr = np.array(self.TRAIN_EMBEDDING_arr)
		self.TRAIN_LABEL_arr = self.LABEL_arr[BATCH_obj.TRAIN]
		BATCH_obj.TRAIN_EMBEDDING_arr = self.TRAIN_EMBEDDING_arr
		##End ENCODER_train_method

	def DISTANCE_measure_method(self, A, B):
		if A==B:
			return [0.]*len(self.CLUSTER_list)
		else:
			KEY = tuple(sorted([A, B]))
			return self.DISTANCE_dic[KEY]
			##End if-else
		##End DISTANCE_measure_method
	
	def KERNEL_train_method(self, BATCH_obj):
		#[1]
		INPUT_arr = []; LABEL_arr = []

		#[2]
		for A, B in itertools.combinations(BATCH_obj.TRAIN, 2):
			SAMPLE_DISTANCE_vec = self.DISTANCE_measure_method(A, B)			
			if BATCH_obj.MODEL_obj.T[0]=="0":
				LABEL_DISTANCE = [0,1][self.LABEL_arr[A]!=self.LABEL_arr[B]]
			else:
				LABEL_DISTANCE = abs(int(self.LABEL_arr[A][-1]) - int(self.LABEL_arr[B][-1])) 
				##End if-else
			INPUT_arr.append(SAMPLE_DISTANCE_vec); LABEL_arr.append(LABEL_DISTANCE)
			##End for

		#[3]
		INPUT_arr = np.array(INPUT_arr); LABEL_arr = np.array(LABEL_arr)
		if BATCH_obj.MODEL_obj.T[1]=="0":
			self.KERNEL_obj = LinearRegression()
		elif BATCH_obj.MODEL_obj.T[1] == "1":
			self.KERNEL_obj = LogisticRegression(random_state=0)
		else:
			self.KERNEL_obj = NNLS()
			##End if-else

		#[4]
		self.KERNEL_obj.fit(INPUT_arr, LABEL_arr)
		##End KERNEL_train_method
	
	def CLUSTER_generate_method(self, MODEL_obj):
		#[1]
		GENE_EMBEDDING_arr = map(lambda x:x.EMBEDDING_vec, self.GENE_list)
		if MODEL_obj.C == "KMeans":
			CLUSTERER_obj = KMeans(n_clusters=MODEL_obj.K, random_state=0)
		else:
			CLUSTERER_obj = AgglomerativeClustering(n_clusters=MODEL_obj.K)
			##End if-else
		CLUSTERER_obj.fit(GENE_EMBEDDING_arr)

		#[2]
		self.CLUSTER_dic = {}
		for GENE_obj, GENE_cluster in zip(self.GENE_list, CLUSTERER_obj.labels_):
			#[2-1]
			GENE_obj.CLUSTER = GENE_cluster
			if GENE_obj.CLUSTER not in self.CLUSTER_dic:
				CLUSTER_obj = Gen_CLUSTER_object()
				CLUSTER_obj.INDEX = GENE_obj.CLUSTER
				CLUSTER_obj.GENE_list = []
				self.CLUSTER_dic[GENE_obj.CLUSTER] = CLUSTER_obj
				##End if
			#[2-2]
			CLUSTER_obj = self.CLUSTER_dic[GENE_obj.CLUSTER]
			CLUSTER_obj.GENE_list.append(GENE_obj)
			##End for

		#[3]
		self.CLUSTER_list = sorted(self.CLUSTER_dic.values(), key=lambda x:x.INDEX)
		for CLUSTER_obj in self.CLUSTER_list:
			CLUSTER_obj.GENE_index_list = sorted(map(lambda x:x.INDEX, CLUSTER_obj.GENE_list))
			##End for
		##End CLUSTER_generate_method
	
	def EMBEDDING_register_method(self):
		#[1]
		EMBEDDING_path = self.OUT_dir + ss + "EMBEDDING.txt"
		EMBEDDING_file = open(EMBEDDING_path, 'r')
		EMBEDDING_file.readline()

		#[2]
		for EMBEDDING_line in EMBEDDING_file:
			GENE_index = int(EMBEDDING_line.split()[0])-1
			EMBEDDING_vec = map(float, EMBEDDING_line.split()[1:])
			GENE_obj = self.GENE_list[GENE_index]
			GENE_obj.EMBEDDING_vec = EMBEDDING_vec
			##End for

		#[3]
		TSNE_path = self.OUT_dir + ss + "TSNE_EMBEDDING.txt"
		for TSNE_line in open(TSNE_path):
			GENE_index = int(TSNE_line.split()[0])
			GENE_obj = self.GENE_list[GENE_index]
			GENE_obj.TSNE_EMBEDDING_vec = map(float, TSNE_line.split()[1:])
			##End for
		##End EMBEDDING_register_method

	def NETWORK_embedding_method(self):
		#[1]
		makePath(self.OUT_dir)
		ADJ_path = self.OUT_dir + ss + "ADJ.txt"
		ADJ_file = open(ADJ_path, 'w')

		#[2]
		for GENE_obj in self.GENE_list:
			GENE_number = GENE_obj.INDEX + 1
			NEIGHBOR_list = map(lambda x:self.GENE_dic[x], GENE_obj.NEIGHBOR_set)
			NEIGHBOR_number_list = sorted(map(lambda x:x.INDEX+1, NEIGHBOR_list))
			ADJ_line = makeLine([GENE_number] + NEIGHBOR_number_list, tt)
			ADJ_file.write(ADJ_line + nn)
			##End for

		#[3]
		ADJ_file.close(); EMBEDDING_path = self.OUT_dir + ss + "EMBEDDING.txt"
		CMD = "deepwalk --input %s --output %s --representation-size 8 --workers %s --seed 0"%(ADJ_path, EMBEDDING_path, self.PN_int)
		commands.getoutput(CMD)
		##End NETWORK_embedding_method
	
	def SAMPLE_register_method(self):
		#[1]
		self.GENE_SAMPLE_arr = []

		#[2]
		for INPUT_line in open(self.INPUT_path):
			GENE_SAMPLE_vec = map(float, INPUT_line.split())	
			self.GENE_SAMPLE_arr.append(GENE_SAMPLE_vec)
			##End for

		#[3]
		self.GENE_list = filter(lambda x:len(x.NEIGHBOR_set)>0, self.GENE_dic.values())
		self.GENE_list = sorted(self.GENE_list, key=lambda x:x.INDEX)
		self.GENE_index_list = map(lambda x:x.INDEX, self.GENE_list)

		#[4]
		self.GENE_SAMPLE_arr = np.array(self.GENE_SAMPLE_arr)[self.GENE_index_list]
		self.SAMPLE_GENE_arr = np.transpose(self.GENE_SAMPLE_arr)

		#[5]
		self.SAMPLE_name_list = open(self.SAMPLE_path).read().split()
		self.GENE_name_list = map(lambda x:x.NAME, self.GENE_list)

		#[6]
		for GENE_obj in self.GENE_list:
			GENE_obj.INDEX = self.GENE_list.index(GENE_obj) 
			##End for
		##End SAMPLE_register_method
	
	def NETWORK_register_method(self):
		#[1]
		self.GENE_dic = {}; INDEX = 0
		for GENE_name in open(self.GENE_path):
			GENE_obj = Gen_GENE_object()
			GENE_obj.NAME = GENE_name.strip()
			GENE_obj.INDEX = INDEX; INDEX += 1
			GENE_obj.NEIGHBOR_set = set([])
			self.GENE_dic[GENE_obj.NAME] = GENE_obj
			##End for

		#[2]
		GENE_name_set = set(self.GENE_dic.keys())
		for ADJ_line in open(self.ADJ_path):
			#[2-1]
			GENE_name, NEIGHBOR_line = ADJ_line.split()
			try: GENE_obj = self.GENE_dic[GENE_name]
			except: continue
			#[2-2]
			GENE_obj.NEIGHBOR_set = set(NEIGHBOR_line.split(cc)) - set([GENE_name])
			GENE_obj.NEIGHBOR_set = GENE_obj.NEIGHBOR_set & GENE_name_set
			##End for
		##End NETWORK_register_method
	
	def ARGUMENT_assign_method(self, ARGUMENT_handler):
		#[1]
		self.HANDLER_obj = ARGUMENT_handler.HANDLER_obj

		#[2]
		self.INPUT_path = self.HANDLER_obj.input_table
		self.SAMPLE_path = self.HANDLER_obj.sample_list
		self.GENE_path = self.HANDLER_obj.gene_list
		self.LABEL_path = self.HANDLER_obj.label_list
		self.ADJ_path = self.HANDLER_obj.adj_list
		self.OUT_dir = self.HANDLER_obj.out_dir
		self.CV_int= int(self.HANDLER_obj.cross_validation)
		self.PN_int = int(self.HANDLER_obj.process_number)
		##End ARGUMENT_assign_method
	##End Gen_TUMOR_encoder

class NNLS(object):
	def __init__(self):
		pass
		##End init

	def fit(self, INPUT_arr, LABEL_arr):
		self.coef_, self.residual_ = nnls(INPUT_arr, LABEL_arr)
		##End for

	def predict(self, INPUT_arr):
		return np.array(map(lambda x:sum(np.multiply(x,self.coef_)), INPUT_arr))
		##End predict
	##End NNLS

import argparse
class Gen_ARGUMENT_handler(object):
	def __init__(self):
		#[1]
		self.PARSER_generate_method()
			
		#[2]
		self.HANDLER_obj = self.PARSER_obj.parse_args()
		##End init
	
	def PARSER_generate_method(self):
		#[1]
		self.PARSER_obj = argparse.ArgumentParser()

		#[2]Essential
		self.PARSER_obj.add_argument('-it', '--input_table', dest="input_table",help = "")
		self.PARSER_obj.add_argument('-sl', '--sample_list', dest="sample_list",help = "")
		self.PARSER_obj.add_argument('-gl', '--gene_list', dest="gene_list",help = "")
		self.PARSER_obj.add_argument('-ll', '--label_list', dest="label_list",help = "")
		self.PARSER_obj.add_argument('-al', '--adj_list', dest="adj_list",help = "")
		self.PARSER_obj.add_argument('-od', '--out_dir', dest="out_dir",help = "")

		#[3]Optional
		self.PARSER_obj.add_argument('-cv', '--cross_validation', dest="cross_validation",help = "", default="NA")
		self.PARSER_obj.add_argument('-pn', '--process_number', dest="process_number",help = "", default=1)
		##End PARSER_generate_method
	##End Gen_ARGUMENT_handler

class Gen_GENE_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_GENE_object

class Gen_MODEL_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_MODEL_object

class Gen_CLUSTER_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_CLUSTER_object

class Gen_BATCH_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_BATCH_object

def removePath(dirPath):
	return commands.getoutput("rm -r %s"%dirPath)
	##End makePath

def makePath(dirPath):
	return commands.getoutput("mkdir %s"%dirPath)
	##End makePath

def makeLine(tokenList, sepToken):
	return sepToken.join(map(str, tokenList))	
	##End makeLine

if __name__ ==  "__main__" :
	main()
	sys.exit()
	##End if
