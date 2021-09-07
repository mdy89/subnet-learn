"""
Title: Subnetwork representation learning for discovering network biomarkers in predicting lymph node metastasis in early oral cancer	
Author:	Minsu Kim, Sangseon Lee, Sangsoo Lim, Doh Young Lee∗, and SunKim*				

v2.0.0 (released at 04/22/2021)															
Description: a script for constructing subnetwork representations for a given dataset
Usage: python SRL_fit_transform.py [sSAS_output_dir] [output_dir]
"""

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
	SRL_model_constructor_obj = SRL_ModelConstructor()

	##End main

import pickle, copy
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, Conv1D, MaxPooling1D, GlobalAveragePooling1D, LocallyConnected1D, Reshape, AveragePooling1D, GlobalMaxPooling1D, Embedding, Activation, Concatenate, Multiply, multiply, average
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
class SRL_ModelConstructor(object):
	def __init__(self):
		#[1]
		self.input_dir = sys.argv[1]
		self.output_dir = sys.argv[2]
	
		#[2]
		self.prepare_inputs()

		#[3]
		self.fit_SRL()

		##End init

	def fit_SRL(self):
		#[1]
		self.set_environments()

		#[2]
		self.create_NN()

		#[3]
		self.fit_NN()

		#[4]
		self.export_embeddings()

		#[5]
		self.export_predictions()	

		#[6]
		self.export_attention_map()

		##End fit_SRL

	def export_attention_map(self):
		#[1]
		self.attention_arr = self.attention_model.predict(self.train_set_obj.sample_subnetwork_arr)	
		self.attention_vec = np.mean(self.attention_arr, axis=0)

		#[2]
		for subnetwork_obj, attention in zip(self.train_set_obj.subnetwork_list, self.attention_vec):
			subnetwork_obj.attention = attention	
			##End for

		#[3]
		attention_path = self.output_dir + ss + "train_set_attention_map.txt"
		attention_file = open(attention_path, 'w')
		head_line = make_line(["#subnetwork", "attention_weight", "gene_components"], tt)
		attention_file.write(head_line + nn)
		for subnetwork_obj in sorted(self.train_set_obj.subnetwork_list, key=lambda x:x.attention, reverse=True):
			components = make_line(sorted(subnetwork_obj.components_set), cc)
			attention_line = make_line([subnetwork_obj.name, subnetwork_obj.attention, components], tt) 
			attention_file.write(attention_line + nn)
			##End for

		#[4]
		attention_file.close()

		#[5]
		self.attention_arr = self.attention_model.predict(self.test_set_obj.sample_subnetwork_arr)	
		self.attention_vec = np.mean(self.attention_arr, axis=0)

		#[6]
		for subnetwork_obj, attention in zip(self.test_set_obj.subnetwork_list, self.attention_vec):
			subnetwork_obj.attention = attention	
			##End for

		#[7]
		attention_path = self.output_dir + ss + "test_set_attention_map.txt"
		attention_file = open(attention_path, 'w')
		head_line = make_line(["#subnetwork", "attention_weight", "gene_components"], tt)
		attention_file.write(head_line + nn)
		for subnetwork_obj in sorted(self.test_set_obj.subnetwork_list, key=lambda x:x.attention, reverse=True):
			components = make_line(sorted(subnetwork_obj.components_set), cc)
			attention_line = make_line([subnetwork_obj.name, subnetwork_obj.attention, components], tt) 
			attention_file.write(attention_line + nn)
			##End for

		#[8]
		attention_file.close()
		##End export_attention_map
	
	def export_predictions(self):
		#[1]
		x = self.train_set_obj.sample_subnetwork_arr
		yhat = self.predictor_model.predict(x)
		ytrue = [self.label_list[np.argmax(x)] for x in self.train_set_obj.sample_label_arr]
		ypred = [self.label_list[np.argmax(x)] for x in yhat]
		report_path = self.output_dir + ss + "classification_report.txt"
		report_file = open(report_path, 'w')
		report_file.write("#[1]Training"+nn)
		report_file.write(str(classification_report(ytrue, ypred, digits=4))+nn)

		#[2]
		train_path = self.output_dir + ss + "train_set_predictions.txt"
		train_file = open(train_path, 'w')
		head_line = make_line(["#sample", "2D_embedding", "actual_label", "predicted_label", "predicted_likelihood(%s)"%make_line(self.label_list,cc)], tt)
		train_file.write(head_line + nn)
		for sample_obj, predicted_likelihood_vec, embedding_vec in zip(self.train_set_obj.sample_list, yhat, self.train_set_obj.e):
			actual_label = sample_obj.label
			predicted_label = self.label_list[np.argmax(predicted_likelihood_vec)]
			predicted_likelihood = make_line(predicted_likelihood_vec, cc)
			_2d_embedding = make_line(embedding_vec, cc)
			sample_line = make_line([sample_obj.name, _2d_embedding, actual_label, predicted_label, predicted_likelihood], tt)
			train_file.write(sample_line + nn)
			##End for

		#[3]
		train_file.close()
		x = self.test_set_obj.sample_subnetwork_arr
		yhat = self.predictor_model.predict(x)
		ytrue = [self.label_list[np.argmax(x)] for x in self.test_set_obj.sample_label_arr]
		ypred = [self.label_list[np.argmax(x)] for x in yhat]
		report_file.write("#[2]Test"+nn)
		report_file.write(str(classification_report(ytrue, ypred, digits=4))+nn)
		report_file.close()

		#[4]
		train_path = self.output_dir + ss + "test_set_predictions.txt"
		train_file = open(train_path, 'w')
		head_line = make_line(["#sample", "2D_embedding", "actual_label", "predicted_label", "predicted_likelihood(%s)"%make_line(self.label_list,cc)], tt)
		train_file.write(head_line + nn)
		for sample_obj, predicted_likelihood_vec, embedding_vec in zip(self.test_set_obj.sample_list, yhat, self.test_set_obj.e):
			actual_label = sample_obj.label
			predicted_label = self.label_list[np.argmax(predicted_likelihood_vec)]
			predicted_likelihood = make_line(predicted_likelihood_vec, cc)
			_2d_embedding = make_line(embedding_vec, cc)
			sample_line = make_line([sample_obj.name, _2d_embedding, actual_label, predicted_label, predicted_likelihood], tt)
			train_file.write(sample_line + nn)
			##End for

		#[5]
		train_file.close()
		##End export_predictions	
	
	def export_embeddings(self):
		#[1]
		self.encoder_model = KernelPCA(kernel="cosine",n_components=2,random_state=0)
		self.encoder_model.fit(self.predictor_model.predict(self.train_set_obj.sample_subnetwork_arr))

		#[2]
		scatter_path = self.output_dir + ss + "train_set_embedding.png"
		try: self.export_scatter_plot(self.train_set_obj, scatter_path)
		except: pass

		#[3]
		scatter_path = self.output_dir + ss + "test_set_embedding.png"
		try: self.export_scatter_plot(self.test_set_obj, scatter_path)
		except: pass
		##End export_embeddings

	def export_scatter_plot(self, dataset_obj, scatter_path):
		#[1]
		dataset_obj.yhat = self.predictor_model.predict(dataset_obj.sample_subnetwork_arr)
		embedding_arr = self.encoder_model.transform(dataset_obj.yhat)
		dataset_obj.e = embedding_arr
		for label in sorted(set(dataset_obj.sample_label_list)):
			#[1-1]
			sample_index_list = []
			for index, target in enumerate(dataset_obj.sample_label_list):
				if label==target:
					sample_index_list.append(index)
					##End if
				##End for
			#[1-2]
			a = [embedding_arr[x][0] for x in sample_index_list]
			b = [embedding_arr[x][1] for x in sample_index_list]
			c = [label]*len(sample_index_list)
			#[1-3]
			plt.scatter(a, b, marker=".", s=50, linewidths=0, label=label)
			##End for

		#[2]
		plt.legend(); plt.savefig(scatter_path); plt.close()
		##End export_scatter_plot

	def fit_NN(self):
		#[1]
		model_path = self.output_dir + ss + "SRL_predictor_model"
		weight_path = self.output_dir + ss + "SRL_predictor_weights"
		early_stopping_obj = EarlyStopping(patience=10,monitor="val_loss")
		model_checkpoint_obj = ModelCheckpoint(weight_path,save_best_only=True,monitor="val_loss",mode="min")

		#[2]
		a = self.train_set_obj.sample_subnetwork_arr
		b = self.train_set_obj.sample_label_arr
		c = early_stopping_obj
		d = model_checkpoint_obj
		history_obj = self.predictor_model.fit(x=a,y=b,epochs=1000,batch_size=128,verbose=1,callbacks=[c,d],validation_split=1/3)

		#[3]
		x, y = self.train_set_obj.sample_subnetwork_arr, self.train_set_obj.sample_label_arr
		a = self.predictor_model.evaluate(x, y)[1]
		x, y = self.test_set_obj.sample_subnetwork_arr, self.test_set_obj.sample_label_arr
		b = self.predictor_model.evaluate(x, y)[1]

		#[4]
		print("Train_ACC: %.4f%%"%(a*100))
		print("Test_ACC: %.4f%%"%(b*100))
		##End fit_NN

	def create_NN(self):
		#[1]
		base_input = Input(shape=(self.representation_dim,))
		attention_input = self.construct_attention_layer(base_input)
		attention_input = Activation("linear")(attention_input)

		#[2]
		attention_layer = Dense(self.feature_dim, use_bias=False, kernel_initializer="uniform")(attention_input)
		attention_layer = BatchNormalization()(attention_layer)
		attention_layer = Dropout(0.5)(attention_layer)
		attention_layer = Activation("softmax")(attention_layer)
		self.attention_model = Model(inputs=base_input, outputs=attention_layer)

		#[3]
		encoder_layer = self.get_encoder_layers(base_input, attention_layer)
		encoder_layer = Activation("softmax")(encoder_layer)
		self.encoder_model = Model(inputs=base_input, outputs=encoder_layer)

		#[4]
		self.predictor_model = Model(inputs=base_input, outputs=encoder_layer)
		self.predictor_model.compile(loss="mean_absolute_error", optimizer="adagrad", metrics=["accuracy"])
		##End create_NN

	def set_environments(self):
		#[1]
		seed_value= 0
		os.environ['PYTHONHASHSEED']=str(seed_value)
		random.seed(seed_value)
		np.random.seed(seed_value)

		#[2]
		self.sample_dim = self.train_set_obj.sample_subnetwork_arr.shape[0]
		self.representation_dim = self.train_set_obj.sample_subnetwork_arr.shape[1]
		self.label_dim = self.train_set_obj.sample_label_arr.shape[1]
		self.feature_dim = int(self.representation_dim/self.label_dim)
		##End set_environments
	
	def retrieve_data(self, dataset_obj):
		#[1]
		subnetwork_list = pickle.load(open(dataset_obj.path, 'rb'))

		#[2]
		dataset_obj.subnetwork_name_list = []
		dataset_obj.subnetwork_list = subnetwork_list

		#[3]
		for subnetwork_obj in subnetwork_list:
			#[3-1]
			sample_subnetwork_arr = np.array([x.feature_vec for x in subnetwork_obj.sample_list])
			try: dataset_obj.sample_subnetwork_arr = np.concatenate((dataset_obj.sample_subnetwork_arr, sample_subnetwork_arr), axis=1)
			except: dataset_obj.sample_subnetwork_arr = sample_subnetwork_arr
			#[2-2]
			dataset_obj.sample_list = subnetwork_obj.sample_list
			#[2-3]
			dataset_obj.subnetwork_name_list += [subnetwork_obj.name]*sample_subnetwork_arr.shape[1]
			##End for

		#[3]
		dataset_obj.sample_label_list = [x.label for x in dataset_obj.sample_list]
		self.label_list = sorted(set(dataset_obj.sample_label_list))
		dataset_obj.sample_label_index_list = [self.label_list.index(x) for x in dataset_obj.sample_label_list]
		dataset_obj.sample_label_arr = to_categorical(dataset_obj.sample_label_index_list, len(self.label_list)) 
		##End retrieve_data

	def prepare_inputs(self):
		#[1]
		self.train_set_obj = Dataset()
		self.train_set_obj.path = self.input_dir + ss + "train_sSAS.pickle"
		self.train_set_obj.name = "train"
		self.test_set_obj = Dataset()
		self.test_set_obj.path = self.input_dir + ss + "test_sSAS.pickle"
		self.test_set_obj.name = "test"

		#[2]
		for dataset_obj in [self.train_set_obj, self.test_set_obj]:
			self.retrieve_data(dataset_obj)
			##End for
		##End prepare_input
	
	def get_encoder_layers(self, base_input, attention_layer):
		#[1]
		encoder_layer_list = []

		#[2]
		for label_index in range(self.label_dim):
			a, b, c = label_index, self.representation_dim, self.label_dim
			label_layer = Lambda(lambda x:x[:,a:b:c])(base_input)
			encoder_layer = multiply([label_layer, attention_layer])
			encoder_layer = Lambda(lambda x:K.sum(x, axis=1, keepdims=True))(encoder_layer)
			encoder_layer_list.append(encoder_layer)
			##End for

		#[3]
		return concatenate(encoder_layer_list)
		##End get_encoder_layers

	def construct_attention_layer(self, base_input):
		#[1]
		feature_layer_list = []

		#[2]
		for feature_index in range(self.feature_dim):
			i = feature_index * self.label_dim
			j = i + self.label_dim
			feature_layer = Lambda(lambda x:x[:,i:j])(base_input)
			feature_layer = Lambda(lambda x:x*K.log(x+K.epsilon()))(feature_layer)
			feature_layer = Lambda(lambda x:K.sum(x, axis=1, keepdims=True))(feature_layer)
			feature_layer_list.append(feature_layer)
			##End for

		#[3]
		return concatenate(feature_layer_list)
		##End construct_attention_layer

	##End SRL_ModelConstructor

from sklearn.decomposition import KernelPCA
class Embedder(object):
	def __init__(self):
		pass
		##End init

	def fit(self, input_arr):
		encoding_arr = self.encoder.predict(input_arr)
		self.embedder.fit(encoding_arr)
		##End fit

	def transform(self, input_arr):
		return self.embedder.transform(self.encoder.predict(input_arr))
		##End transform

	def fit_transform(self, input_arr):
		#[1]
		self.fit(input_arr)

		#[2]
		return self.transform(input_arr)
		##End fit_transform

	##End Sample

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

class Dataset(object):
	def __init__(self):
		pass
		##End init
	##End Dataset

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
