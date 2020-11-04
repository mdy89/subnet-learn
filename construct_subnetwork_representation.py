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
	GIN_path = sys.argv[1]
	INPUT_path = sys.argv[2]
	OUT_dir = sys.argv[3]

	#[2]
	GIN_encoder = Gen_GIN_encoder()
	##End main

import community
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ranksums
from scipy.stats import norm
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest
from scipy.spatial.distance import euclidean
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import random
import sklearn
import networkx as nx
class Gen_GIN_encoder(object):
	def __init__(self):
		#[1]
		self.GIN_path = sys.argv[1]
		self.INPUT_path = sys.argv[2]
		self.OUT_dir = sys.argv[3]

		#[2]
		self.GIN_register_method()

		#[3]
		self.INPUT_register_method()

		#[4]
		self.MODEL_evaluate_method()

		#[5]	
		self.FEATURE_extract_method()

		##End init
	
	def FEATURE_extract_method(self):
		#[1]
		#30_0.01_0.98_128_64_32_64_128_logistic_propagation_2
		#26_0.18_0.87_128_64_32_64_128_logistic_propagation_2
		#25_0.17_0.93_128_64_32_64_128_logistic_propagation_2
		#36_0.25_0.84_128_64_32_64_128_logistic_propagation_2
		MODEL_obj = Gen_MODEL_object(); MODEL_obj.K = [128,64,32,64,128]
		MODEL_obj.A = "logistic"; MODEL_obj.LT = 0.25; MODEL_obj.HT = 0.84
		MODEL_obj.S = 36; MODEL_obj.I = len(MODEL_obj.K)/2; MODEL_obj.M = "propgation"
		self.NETWORK_SAMPLE_register_method(MODEL_obj)

		#[2]
		self.UP_SAMPLE_GENE_arr = np.array(map(lambda x:map(lambda y:[0,1][y>MODEL_obj.HT], x), self.SAMPLE_GENE_arr))
		self.DOWN_SAMPLE_GENE_arr = np.array(map(lambda x:map(lambda y:[0,1][y<MODEL_obj.LT], x), self.SAMPLE_GENE_arr))

		#[2]
		ENCODER_obj = MLPRegressor(hidden_layer_sizes=MODEL_obj.K,early_stopping=True,random_state=0,activation=MODEL_obj.A,max_iter=10000)
		ENCODER_obj.fit(self.GENE_GENE_arr, self.TARGET_arr)
		MODEL_obj.ENCODER_obj = ENCODER_obj
		self.GENESET_encode_method(MODEL_obj)

		#[2]
		self.SAMPLE_list = filter(lambda x:x.T in ["T1","T2"], self.SAMPLE_list)
		self.SAMPLE_list = filter(lambda x:x.N in ["N0","N1","N2","N3"], self.SAMPLE_list)
		self.SAMPLE_list= filter(lambda x:x.TISSUE=="TP", self.SAMPLE_list)
		self.NOT_TONGUE_list = filter(lambda x:not x.TONGUE, self.SAMPLE_list)
		self.TONGUE_list = filter(lambda x:x.TONGUE, self.SAMPLE_list)
		self.SNUH_TONGUE_list = filter(lambda x:x.SOURCE=="SNUH", self.TONGUE_list)
		self.TCGA_TONGUE_list = filter(lambda x:x.SOURCE=="TCGA", self.TONGUE_list)

		#[3]
		NOT_TONGUE_index_list = map(lambda x:x.INDEX, self.NOT_TONGUE_list)
		self.NOT_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[NOT_TONGUE_index_list]
		TONGUE_index_list = map(lambda x:x.INDEX, self.TONGUE_list)
		self.TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[TONGUE_index_list]
		SNUH_TONGUE_index_list = map(lambda x:x.INDEX, self.SNUH_TONGUE_list)
		self.SNUH_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[SNUH_TONGUE_index_list]
		TCGA_TONGUE_index_list = map(lambda x:x.INDEX, self.TCGA_TONGUE_list)
		self.TCGA_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[TCGA_TONGUE_index_list]
		REGRESSOR_obj = LinearRegression()

		#[4]
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))
		try: REGRESSOR_obj.fit(INPUT_arr, LABEL_arr)
		except: self.SEM_obj.release(); return False
		INPUT_arr = self.SNUH_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.SNUH_TONGUE_list))
	
		#[5]
		self.BOXPLOT_dir = self.OUT_dir + ss + "BOXPLOT.Dir"
		removePath(self.BOXPLOT_dir); makePath(self.BOXPLOT_dir)

		#[6]
		X_list = LABEL_arr; Y_list = REGRESSOR_obj.predict(INPUT_arr)
		print spearmanr(X_list, Y_list)
		BOX_dic = {}
		for X, Y in zip(X_list, Y_list):
			try: BOX_dic[X].append(Y)
			except: BOX_dic[X] = [Y]
			##End for
		BOX_list = map(lambda x:BOX_dic[x], sorted(BOX_dic))
		BOXPLOT_path = self.BOXPLOT_dir + ss + "TCGA_SNUH.BOXPLOT.png"
		plt.boxplot(BOX_list, showfliers=False, labels=["N0","N1","N2"])	
		plt.savefig(BOXPLOT_path); plt.close()	
	
		#[7]
		INPUT_arr = self.SNUH_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.SNUH_TONGUE_list))
		try: REGRESSOR_obj.fit(INPUT_arr, LABEL_arr)
		except: self.SEM_obj.release(); return False
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))

		#[8]
		X_list = LABEL_arr; Y_list = REGRESSOR_obj.predict(INPUT_arr)
		print spearmanr(X_list, Y_list)
		BOX_dic = {}
		for X, Y in zip(X_list, Y_list):
			try: BOX_dic[X].append(Y)
			except: BOX_dic[X] = [Y]
			##End for
		BOX_list = map(lambda x:BOX_dic[x], sorted(BOX_dic))
		BOXPLOT_path = self.BOXPLOT_dir + ss + "SNUH_TCGA.BOXPLOT.png"
		plt.boxplot(BOX_list, showfliers=False, labels=["N0","N1","N2"])	
		plt.savefig(BOXPLOT_path); plt.close()	
		
		#[9]
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))

		#[10]
		X_list, Y_list = [], []; ACC_list = []
		for TRAIN, TEST in StratifiedKFold(n_splits=10, random_state=0).split(INPUT_arr, LABEL_arr):
			TRAIN_X, TEST_X = INPUT_arr[TRAIN], INPUT_arr[TEST]
			TRAIN_Y, TEST_Y = LABEL_arr[TRAIN], LABEL_arr[TEST]
			REGRESSOR_obj.fit(TRAIN_X, TRAIN_Y)
			TEST_YHAT = REGRESSOR_obj.predict(TEST_X)
			TEST_ACC = spearmanr(TEST_YHAT, TEST_Y)[0]
			ACC_list.append(TEST_ACC)
			X_list += list(TEST_Y); Y_list += list(TEST_YHAT)
			##End for

		#[11]
		print spearmanr(X_list, Y_list)
		BOX_dic = {}
		for X, Y in zip(X_list, Y_list):
			try: BOX_dic[X].append(Y)
			except: BOX_dic[X] = [Y]
			##End for

		#[12]
		BOX_list = map(lambda x:BOX_dic[x], sorted(BOX_dic))
		BOXPLOT_path = self.BOXPLOT_dir + ss + "TONGUE_10CV.BOXPLOT.png"
		plt.boxplot(BOX_list, showfliers=False, labels=["N0","N1","N2"])	
		plt.savefig(BOXPLOT_path); plt.close()	
		##End FEATURE_extract_method
	
	def MODEL_evaluate_method(self):
		#[1]
		self.MODEL_path = self.OUT_dir + ss + "MODEL_RESULT.txt"
		self.MODEL_file = open(self.MODEL_path, 'w'); self.MODEL_file.close()

		#[2]
		self.PROC_list = []
		self.SEM_obj = mp.Semaphore(1)

		#[3]
		#25_0.17_0.93_128_64_32_64_128_logistic_propagation_2
		W_ranges = map(lambda x:2**x, range(4,8))[::-1]
		K_ranges = []
		for D in range(1, len(W_ranges)+1):
			for HALF in itertools.combinations(W_ranges, D):
				HALF = sorted(HALF, reverse=True)
				L = HALF[:-1] + [HALF[-1]] + HALF[:-1][::-1]
				K_ranges.append(L)
				##End for
			##End for
		K_ranges = [[128,64,32,64,128]]
		
		#[4]
		A_ranges = ["logistic"]
		LT_ranges = map(lambda x:x/100., range(1,26))
		HT_ranges = map(lambda x:x/100., range(75,100))[::-1]
		M_ranges = ["propagation"]
		S_ranges = sorted(range(1, 100), key=lambda x:abs(x-25))
		
		#[5]
		for S in S_ranges:
			MODEL_obj = Gen_MODEL_object(); MODEL_obj.S = S
			self.NETWORK_SAMPLE_register_method(MODEL_obj)
			for K, A in itertools.product(K_ranges, A_ranges):
				MODEL_obj.K = K; MODEL_obj.A = A
				ENCODER_obj = MLPRegressor(hidden_layer_sizes=K,early_stopping=True,random_state=0,activation=A,max_iter=10000)
				ENCODER_obj.fit(self.GENE_GENE_arr, self.TARGET_arr)
				MODEL_obj.ENCODER_obj = ENCODER_obj
				ENCODE_idx = len(MODEL_obj.K)/2 ; I_ranges = [ENCODE_idx]
				for LT, HT in itertools.product(LT_ranges, HT_ranges):
					MODEL_obj.LT = LT; MODEL_obj.HT = HT
					self.UP_SAMPLE_GENE_arr = np.array(map(lambda x:map(lambda y:[0,1][y>MODEL_obj.HT], x), self.SAMPLE_GENE_arr))
					self.DOWN_SAMPLE_GENE_arr = np.array(map(lambda x:map(lambda y:[0,1][y<MODEL_obj.LT], x), self.SAMPLE_GENE_arr))
					for I, M in itertools.product(I_ranges, M_ranges):
						MODEL_obj.I = I; MODEL_obj.M = M
						MODEL_obj.METHOD = makeLine([S, LT, HT, makeLine(K,"_"), A, M, I], "_")
						self.SEM_obj.acquire(); ARGV = tuple([MODEL_obj])
						PROC_obj = mp.Process(target=self.MODEL_evaluate_sub, args=ARGV)
						PROC_obj.start(); self.PROC_list.append(PROC_obj)
						##End for		
					##End for
				##End for
			##End for

		#[6]
		for PROC_obj in self.PROC_list:
			PROC_obj.join()
			##End for
		##End MODEL_evaluate_method
		
	def NETWORK_SAMPLE_register_method(self, MODEL_obj):
		#[1]
		random.seed(0)
		self.GENE_list = []
		for COMM_line in open("STRING_GENE.txt"):
			GENE_obj = Gen_GENE_object()
			GENE_name_list = COMM_line.split()[1].split(cc)
			GENE_name_list = filter(lambda x:x in self.GENE_name_index_dic, GENE_name_list)
			GENE_index_list = map(lambda x:self.GENE_name_index_dic[x], GENE_name_list) 
			GENE_obj.EDGE_list = GENE_index_list 
			if not GENE_obj.EDGE_list:
				continue
				##End if
			self.GENE_list.append(GENE_obj); GENE_obj.INDEX = self.GENE_list.index(GENE_obj)
			##End for

		"""
		#[2]
		self.GENE_GENE_arr = []; self.TARGET_arr = []
		for S in range(5, 100, 5):
			MODEL_obj.S = S
			for ITER in range(100):
				SAMPLE_size = max(int(math.ceil(MODEL_obj.S * len(self.GENE_list) / 100.)), 1)
				GENE_list = random.sample(self.GENE_list, SAMPLE_size)
				GENE_GENE_vec = [0]*len(self.GENE_name_index_dic)
				TARGET_vec = [0]*len(self.GENE_list)
				for GENE_obj in GENE_list:
					TARGET_vec[GENE_obj.INDEX] = 1
					for TARGET_index in GENE_obj.EDGE_list:
						GENE_GENE_vec[TARGET_index] = 1
						##End for
					##End for
				self.GENE_GENE_arr.append(GENE_GENE_vec)
				self.TARGET_arr.append(TARGET_vec)
				##End for
			##End for

		"""

		#[2]
		self.GENE_GENE_arr = []; self.TARGET_arr = []
		for ITER in range(1024):
			SAMPLE_size = max(int(math.ceil(MODEL_obj.S * len(self.GENE_list) / 100.)), 1)
			GENE_list = random.sample(self.GENE_list, SAMPLE_size)
			GENE_GENE_vec = [0]*len(self.GENE_name_index_dic)
			TARGET_vec = [0]*len(self.GENE_list)
			for GENE_obj in GENE_list:
				TARGET_vec[GENE_obj.INDEX] = 1
				for TARGET_index in GENE_obj.EDGE_list:
					GENE_GENE_vec[TARGET_index] = 1
					##End for
				##End for
			self.GENE_GENE_arr.append(GENE_GENE_vec)
			self.TARGET_arr.append(TARGET_vec)
			##End for

		#[3]
		self.GENE_GENE_arr = np.array(self.GENE_GENE_arr)
		self.TARGET_arr = np.array(self.TARGET_arr)
		##End NETWORK_SAMPLE_register_method
	
	def GENESET_encode_method(self, MODEL_obj):
		#[1]
		ENCODER_obj = MODEL_obj.ENCODER_obj

		#[2]
		self.FUNC_dic = {
			"identity": (lambda x:x),
			"logistic": (lambda x:1/(1+np.exp(-x))),
			"relu": (lambda x:map(lambda y:max(0,y),x)),
			"tanh": np.tanh
		}
		FUNC = self.FUNC_dic[MODEL_obj.A]

		#[3]
		W_list = ENCODER_obj.coefs_; B_list = ENCODER_obj.intercepts_ + [[0]*self.UP_SAMPLE_GENE_arr.shape[1]]
		self.UP_SAMPLE_ENCODE_arr = []
		for SAMPLE_GENE_vec in self.UP_SAMPLE_GENE_arr:
			for W_arr, B_arr in zip(W_list[:MODEL_obj.I+1], B_list[:MODEL_obj.I+1]): 
				SAMPLE_GENE_vec = FUNC(np.dot(SAMPLE_GENE_vec, W_arr) + B_arr)
				##End for
			self.UP_SAMPLE_ENCODE_arr.append(SAMPLE_GENE_vec)
			##End for
		
		#[4]
		self.DOWN_SAMPLE_ENCODE_arr = []
		for SAMPLE_GENE_vec in self.DOWN_SAMPLE_GENE_arr:
			for W_arr, B_arr in zip(W_list[:MODEL_obj.I+1], B_list[:MODEL_obj.I+1]): 
				SAMPLE_GENE_vec = FUNC(np.dot(SAMPLE_GENE_vec, W_arr) + B_arr)
				##End for
			self.DOWN_SAMPLE_ENCODE_arr.append(SAMPLE_GENE_vec)
			##End for

		#[5]
		self.SAMPLE_GENE_arr = np.array(map(lambda x:list(x[0])+list(x[1]), zip(self.UP_SAMPLE_ENCODE_arr, self.DOWN_SAMPLE_ENCODE_arr)))
		##End GENESET_encode_method

	def SCORE_calculate_method(self, Y_estimate, Y_true):
		#[1]
		Y_dic = {}

		#[2]
		for VALUE, LABEL in zip(Y_estimate, Y_true):
			try: Y_dic[LABEL].append(VALUE)
			except: Y_dic[LABEL] = [VALUE]
			##End for

		#[3]
		Q_list = [25, 50, 75]
		OBSERVED_list = []
		for BIN in sorted(Y_dic):
			BIN_list = Y_dic[BIN]
			OBSERVED_list += map(lambda x:np.percentile(BIN_list, x), Q_list)
			##End for
		CORRECT_list = range(1, len(OBSERVED_list)+1)
		OBSERVED_list = rankdata(OBSERVED_list)
		X = euclidean(CORRECT_list, OBSERVED_list)
		SCORE = 1/(1.+X)

		#[4]
		return SCORE
		##End SCORE_calculate_method
	
	def MODEL_evaluate_sub(self, MODEL_obj):
		#[1]
		self.GENESET_encode_method(MODEL_obj)

		#[2]
		self.SAMPLE_list = filter(lambda x:x.T in ["T1","T2"], self.SAMPLE_list)
		self.SAMPLE_list = filter(lambda x:x.N in ["N0","N1","N2","N3"], self.SAMPLE_list)
		self.SAMPLE_list= filter(lambda x:x.TISSUE=="TP", self.SAMPLE_list)
		self.NOT_TONGUE_list = filter(lambda x:not x.TONGUE, self.SAMPLE_list)
		self.TONGUE_list = filter(lambda x:x.TONGUE, self.SAMPLE_list)
		self.SNUH_TONGUE_list = filter(lambda x:x.SOURCE=="SNUH", self.TONGUE_list)
		self.TCGA_TONGUE_list = filter(lambda x:x.SOURCE=="TCGA", self.TONGUE_list)

		#[3]
		NOT_TONGUE_index_list = map(lambda x:x.INDEX, self.NOT_TONGUE_list)
		self.NOT_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[NOT_TONGUE_index_list]
		TONGUE_index_list = map(lambda x:x.INDEX, self.TONGUE_list)
		self.TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[TONGUE_index_list]
		SNUH_TONGUE_index_list = map(lambda x:x.INDEX, self.SNUH_TONGUE_list)
		self.SNUH_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[SNUH_TONGUE_index_list]
		TCGA_TONGUE_index_list = map(lambda x:x.INDEX, self.TCGA_TONGUE_list)
		self.TCGA_TONGUE_SAMPLE_GENE_arr = self.SAMPLE_GENE_arr[TCGA_TONGUE_index_list]
		D_size = self.SAMPLE_GENE_arr.shape[1]; L = filter(lambda x:x>0, [D_size])
		REGRESSOR_obj = LinearRegression()

		#[4]
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))
		TRAIN_X_list, TRAIN_Y_list, TEST_X_list, TEST_Y_list = [], [], [], []
		for TRAIN, TEST in StratifiedKFold(n_splits=10, random_state=0).split(INPUT_arr, LABEL_arr):
			TRAIN_X, TEST_X = INPUT_arr[TRAIN], INPUT_arr[TEST]
			TRAIN_Y, TEST_Y = LABEL_arr[TRAIN], LABEL_arr[TEST]
			REGRESSOR_obj.fit(TRAIN_X, TRAIN_Y)
			TRAIN_YHAT = REGRESSOR_obj.predict(TRAIN_X)
			TEST_YHAT = REGRESSOR_obj.predict(TEST_X)
			TRAIN_X_list += list(TRAIN_YHAT)
			TRAIN_Y_list += list(TRAIN_Y)
			TEST_X_list += list(TEST_YHAT)
			TEST_Y_list += list(TEST_Y)
			##End for

		#[5]
		A = spearmanr(TRAIN_X_list, TRAIN_Y_list)[0]
		B = spearmanr(TEST_X_list, TEST_Y_list)[0]

		#[6]
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))
		try: REGRESSOR_obj.fit(INPUT_arr, LABEL_arr)
		except: self.SEM_obj.release(); return False
		C = spearmanr(REGRESSOR_obj.predict(INPUT_arr), LABEL_arr)[0]
#		C = REGRESSOR_obj.score(INPUT_arr, LABEL_arr)

		#[7]
		INPUT_arr = self.SNUH_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.SNUH_TONGUE_list))
		D = spearmanr(REGRESSOR_obj.predict(INPUT_arr), LABEL_arr)[0]
#		D = REGRESSOR_obj.score(INPUT_arr, LABEL_arr)
		
		#[8]
		INPUT_arr = self.SNUH_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.SNUH_TONGUE_list))
		try: REGRESSOR_obj.fit(INPUT_arr, LABEL_arr)
		except: self.SEM_obj.release(); return False
		E = spearmanr(REGRESSOR_obj.predict(INPUT_arr), LABEL_arr)[0]
#		E = REGRESSOR_obj.score(INPUT_arr, LABEL_arr)

		#[9]
		INPUT_arr = self.TCGA_TONGUE_SAMPLE_GENE_arr
		LABEL_arr = np.array(map(lambda x:int(x.N[-1]), self.TCGA_TONGUE_list))
		F = spearmanr(REGRESSOR_obj.predict(INPUT_arr), LABEL_arr)[0]
#		F = REGRESSOR_obj.score(INPUT_arr, LABEL_arr)

		#[10]
		self.MODEL_file = open(self.MODEL_path, 'a')
		MODEL_line = makeLine([MODEL_obj.METHOD, A, B, C, D, E, F], tt)
		self.MODEL_file.write(MODEL_line + nn); self.MODEL_file.close(); self.SEM_obj.release()
		##End MODEL_evaluate_sub
	
	def INPUT_register_method(self):
		#[1]
		INPUT_file = open(self.INPUT_path, 'r')
		
		#[2]
		GENE_name_list = INPUT_file.readline().split()[7:]
		GENE_name_index_dic = {}
		for GENE_name, GENE_index in zip(GENE_name_list, range(len(GENE_name_list))):
			GENE_name_index_dic[GENE_name] = GENE_index
			##End for

		#[3]
		INTERSECT_name_list = sorted(set(GENE_name_list) & set(self.GENE_dic.keys()))
		INTERSECT_index_list = map(lambda x:GENE_name_index_dic[x], INTERSECT_name_list)
		
		#[4]
		self.SAMPLE_list = []
		self.SAMPLE_GENE_arr = []
		for INPUT_line in INPUT_file:
			#[4-1]
			SAMPLE, SOURCE, COHORT, TONGUE, TISSUE, T, N = INPUT_line.split()[:7]
			SAMPLE_obj = Gen_SAMPLE_object()
			SAMPLE_obj.NAME = SAMPLE; SAMPLE_obj.COHORT = COHORT
			SAMPLE_obj.TONGUE = TONGUE=="True"; SAMPLE_obj.TISSUE = TISSUE
			SAMPLE_obj.SOURCE = SOURCE; SAMPLE_obj.T = T; SAMPLE_obj.N = N
			#[4-2]
			self.SAMPLE_list.append(SAMPLE_obj)
			SAMPLE_obj.INDEX = self.SAMPLE_list.index(SAMPLE_obj)
			SAMPLE_GENE_vec = np.array(map(float, INPUT_line.split()[7:]))[INTERSECT_index_list]
			#[4-3]
			self.SAMPLE_GENE_arr.append(SAMPLE_GENE_vec)
			##End for
		
		#[5]
		self.SAMPLE_GENE_arr = np.array(self.SAMPLE_GENE_arr)
		self.GENE_SAMPLE_arr = np.transpose(self.SAMPLE_GENE_arr)
		self.GENE_name_list = INTERSECT_name_list
		self.GENE_name_set = set(INTERSECT_name_list)
		
		#[6]
		self.GENE_name_index_dic = {}
		for GENE_name, GENE_index in zip(self.GENE_name_list, range(len(self.GENE_name_list))):
			self.GENE_name_index_dic[GENE_name] = GENE_index
			##End for

		#[7]
		self.GENE_list = []
		for GENE_name in INTERSECT_name_list:
			GENE_obj = self.GENE_dic[GENE_name]
			GENE_obj.NEIGHBOR_set &= self.GENE_name_set
			GENE_obj.EDGE_list = sorted(map(lambda x:self.GENE_name_index_dic[x], GENE_obj.NEIGHBOR_set))
			self.GENE_list.append(GENE_obj)
			GENE_obj.INDEX = self.GENE_list.index(GENE_obj)
			##End for
		##End INPUT_register_method
	
	def GIN_register_method(self):
		#[1]
		self.GENE_dic = {}

		#[2]
		self.CUT = 0
		for GIN_line in open(self.GIN_path):
			#[2-1]
			GENE_name, NEIGHBOR_line = GIN_line.split()
			NEIGHBOR_line =  self.CUT_method(NEIGHBOR_line)
			if not NEIGHBOR_line:
				continue
				##End if
			#[2-2]
			GENE_obj = Gen_GENE_object()
			GENE_obj.NAME = GENE_name
			GENE_obj.NEIGHBOR_set = set([GENE_name])
			for NEIGHBOR in NEIGHBOR_line.split(cc):
				TARGET_name, TARGET_score = NEIGHBOR.split(":")
				GENE_obj.NEIGHBOR_set.add(TARGET_name)
				##End for
			#[2-3]
			self.GENE_dic[GENE_name] = GENE_obj
			##End for
		##End GIN_register_method

	def CUT_method(self, NEIGHBOR_line):
		#[1]
		TARGET_list = []	

		#[2]
		for TARGET in NEIGHBOR_line.split(cc):
			TARGET_score = int(TARGET.split(":")[1])
			if TARGET_score >= self.CUT:
				TARGET_list.append(TARGET)
				##End if
			##End for

		#[3]
		return makeLine(TARGET_list, cc)
		##End CUT_method

	##End Gen_GIN_encoder


class Gen_GENESET_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_GENESET_object

class Gen_GENE_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_GENE_object

class Gen_SAMPLE_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_SAMPLE_object

class Gen_COHORT_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_COHORT_object

class Gen_MODEL_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_COHORT_object

class Gen_ENCODER_object(object):
	def __init__(self):
		pass
		##End init
	##End Gen_ENCODER_object

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
