import copy
import os
import random
import warnings

import math
import numpy as np
import sys
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# import the sklearn library
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
from sklearn.utils.multiclass import type_of_target


#google lib
from utils import utils

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_base_AL_mapping
from sampling_methods.constants import get_wrapper_AL_mapping
get_base_AL_mapping()


class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def run(trn_ds, tst_ds, lbr, model, qs, quota, n_labeled):
	acc = []
	auc = []
	f1 = []

	for i_add in range(quota):
		ask_id= qs.make_query()
		X_train, y_train = zip(*trn_ds.data)
		X_train_1 = []
		y_train_1 = []
		for idx in range(len(y_train)):
			if y_train[idx] != None:
				X_train_1.append(X_train[idx])
				y_train_1.append(y_train[idx])
		X_test, y_test = zip(*tst_ds.data)		
		scalar=StandardScaler()
		X_train_tf = scalar.fit_transform(X_train_1)
		X_test_tf=scalar.transform(X_test)
		model.fit(X_train_tf, y_train_1)
		y_pred = model.predict(X_test_tf)
		acc_T = accuracy_score(y_test, y_pred) * 1.0
		auc_T = roc_auc_score(y_test, y_pred) * 1.0
		f1_T = f1_score(y_test, y_pred) * 1.0
		print(str(round(acc_T,4)) +' '+str(round(auc_T,4)) +' ' + str(round(f1_T,4)))

		#print(acc_T)
		acc.append(acc_T)
		auc.append(auc_T)
		f1.append(f1_T)

		lb = lbr.label(X_train[ask_id])
		trn_ds.update(ask_id, lb)

	X_train, y_train = zip(*trn_ds.data)
	#X_train_1 = X_train[:(n_labeled + i_add)]
	#y_train_1 = y_train[:(n_labeled + i_add)]
	X_train_1 = []
	y_train_1 = []
	for idx in range(len(y_train)):
		if y_train[idx] != None:
			X_train_1.append(X_train[idx])
			y_train_1.append(y_train[idx])
	#print(len(y_train_1))
	X_test, y_test = zip(*tst_ds.data)		
	scalar=StandardScaler()
	X_train_tf = scalar.fit_transform(X_train_1)
	X_test_tf=scalar.transform(X_test)
	#model.fit(X_train_tf, y_train_1)
	
	#y_pred = model.predict(X_test_tf)
	model.fit(X_train_1,y_train_1)
	y_pred = model.predict(X_test)
	acc_T = accuracy_score(y_test, y_pred) * 1.0
	auc_T = roc_auc_score(y_test, y_pred) * 1.0
	f1_T = f1_score(y_test, y_pred) * 1.0
	print(str(round(acc_T,4)) +' '+str(round(auc_T,4)) +' ' + str(round(f1_T,4)))

	acc.append(acc_T)
	auc.append(auc_T)
	f1.append(f1_T)
	return acc, auc, f1

def split_data(dataset_filepath, test_size, n_labeled):
	X, y = load_svmlight_file(dataset_filepath)
	X = X.toarray().tolist()
	y = y.tolist()
	num_class = len(set(y))
	#print(num_class)
	dict_data=dict()
	for i in range(0,len(X)):
		dict_data[i+1]=[X[i],y[i]]
	labeled_data=[]
	test_data=[]
	unlabeled_data=[]
	temp=list(range(1,len(X)+1))
	testsize=int(len(X)*test_size)

	trainsize=len(X)-testsize-n_labeled

	temp2=copy.deepcopy(temp)

	test_data_temp=random.sample(range(0,len(temp2)),testsize)
	for i in test_data_temp:
		test_data.append(temp2[i])
		temp.remove(temp2[i])

	temp1=copy.deepcopy(temp)

	Flag = False
	while(Flag == False):
		temp1=copy.deepcopy(temp)
		labeled_data_temp1=random.sample(range(0,len(temp1)),n_labeled)
		labeled_data_1 = []
		for i in labeled_data_temp1:
			labeled_data_1.append(temp1[i])


		X_temp, y_temp = form_dataset(dict_data, labeled_data_1)
		count_temp1 = []
		for s in set(y_temp):
			count_temp1.append(y_temp.count(s))
		#print(y_temp)
		if len(set(y_temp)) == num_class and min(count_temp1)>=2:
			Flag = True
			labeled_data_temp = copy.deepcopy(labeled_data_temp1)
			break

	for i in labeled_data_temp:
		labeled_data.append(temp1[i])
		temp.remove(temp1[i])

	unlabeled_data=copy.deepcopy(temp)

	return dict_data, labeled_data, test_data, unlabeled_data


def form_dataset(dict_data, data_id):
	X=[]
	y=[]
	for i in data_id:
		X.append(dict_data[i][0])
		y.append(dict_data[i][1])
	return X,y

def cal_average(res_total):
	res = []
	for i in range(len(res_total[0])):
		sum = 0.0
		for j in range(len(res_total)):
			sum = sum + res_total[j][i]
		sum = float((sum *1.0) / (len(res_total)*1.0))
		res.append(sum)
	return res


def test_model2(labeled_X, labeled_y,test_X, test_y):
	X_train = labeled_X
	X_test = test_X
	y_train = labeled_y
	y_test = test_y
	scalar=StandardScaler() 
	X_train_tf = scalar.fit_transform(X_train)
	X_test_tf=scalar.transform(X_test)
	clf = SVC(kernel = 'rbf',gamma='auto')
	clf.fit(X_train_tf,y_train) 
	y_pred = clf.predict(X_test_tf)
	acc = accuracy_score(y_test, y_pred) * 1.0

	idx = int(np.max(y_test))
	y_test1=np.zeros((len(y_pred),idx),dtype = 'int')
	y_pred1=np.zeros((len(y_pred),idx),dtype = 'int')

	for i in range(0,len(y_pred)):
		y_test1[i][int(y_test[i])-1] = 1
		y_pred1[i][int(y_pred[i])-1] = 1
	auc = roc_auc_score(y_test1,y_pred1, multi_class='ovr', average='macro')
	f1 = f1_score(y_test1, y_pred1,average = 'macro') * 1.0

	return acc, auc, f1

def select_batch(batch_size, quota, unlabeled_X, unlabeled_y, labeled_X, labeled_y, test_X, test_y, model):
	acc, auc, f1 = test_model2(labeled_X, labeled_y,test_X, test_y)
	print(str(round(acc,4)) +' '+str(round(auc,4)) +' ' + str(round(f1,4)))
	res_acc=[]
	res_auc=[]
	res_f1=[]

	res_acc.append(acc)
	res_auc.append(auc)
	res_f1.append(f1)

	X = np.array(labeled_X+unlabeled_X)
	y = np.array(labeled_y+unlabeled_y)


	Y_idx = []
	
	point_record = []


	already_selected1 = list(range(0,len(labeled_y)))

	while (quota>0):

		if quota < batch_size:  
			batch_size = quota


		quota = quota - batch_size
		
		if model.name == 'graph_density':
			Y_idx = model.select_batch_(N = batch_size, already_selected=already_selected1)
		elif model.name == 'margin' or model.name == 'informative_diverse' or model.name == 'margin_cluster_mean' or model.name == 'kcenter':
			selected_model = utils.get_model(method = 'logistic')
			selected_model.fit(np.array(labeled_X),np.array(labeled_y))
			Y_idx = model.select_batch_(model = selected_model, N = batch_size, already_selected=already_selected1)
		elif model.name == 'hierarchical':
			labeled_hier=dict(zip(already_selected1, labeled_y))
			Y_idx = model.select_batch_(N = batch_size, already_selected = already_selected1,labeled = labeled_hier, y=y)
		else:
			Y_idx = model.select_batch_(already_selected1,batch_size)
		already_selected1.extend(Y_idx)

		for i in Y_idx:
			labeled_X.append(X[i])
			labeled_y.append(y[i])
			tmpx = copy.deepcopy(X[i].tolist())
			tmpx.append(int(y[i]))
			
			point_record.append(tmpx)

		acc_T, auc_T, f1_T = test_model2(labeled_X, labeled_y,test_X, test_y)
		print(str(round(acc_T,4)) +' '+str(round(auc_T,4)) +' ' + str(round(f1_T,4)))

		res_acc.append(acc_T)
		res_auc.append(auc_T)
		res_f1.append(f1_T)
		if quota <= 0:
			break

	return res_acc, res_auc, res_f1

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum


def main():

	ds_name = str(sys.argv[1])

	strategy = str(sys.argv[2])
	
	bs = int(sys.argv[3])
	
	iternum = int(sys.argv[4])
	
	n_labeled = int(sys.argv[5])
	
	sys.stdout = Logger('./logfile/'+ds_name+ '_'  + strategy + '_' + str(bs) + '_' + str(n_labeled) + '_' + str(iternum) +'_log.txt')
	random.seed(4666)

	warnings.filterwarnings('ignore')

	dataset_filepath = os.path.join(os.path.abspath('..') + '/Dataset/dealeddata', '%s.txt' % ds_name)
	print(dataset_filepath)

	test_size = 0.4
	#n_labeled = 80

	T1 = 0
	#quota = 100

	res_acc_total = []
	res_auc_total = []
	res_f1_total = []


	
	while(T1 < iternum): 
		print(str(T1)+'th iteration')
		T1 = T1 + 1
		dict_data=dict()
		labeled_data=[]
		test_data=[]
		unlabeled_data=[]

		dict_data,labeled_data,test_data,unlabeled_data = split_data(dataset_filepath, test_size, n_labeled)
		

		dict_train_idx=labeled_data+unlabeled_data
		quota = len(unlabeled_data)

		unlabeled_X_init, unlabeled_y_init = form_dataset(dict_data,unlabeled_data)
		labeled_X_init, labeled_y_init = form_dataset(dict_data,labeled_data)
		test_X, test_y = form_dataset(dict_data,test_data)


		trainX = labeled_X_init+unlabeled_X_init
		trainy = labeled_y_init+unlabeled_y_init

		#AL strategy

		AL_strategy = AL_MAPPING[strategy](np.array(trainX),np.array(trainy), 0)

		unlabeled_X =  copy.deepcopy(unlabeled_X_init)
		unlabeled_y =  copy.deepcopy(unlabeled_y_init)
		labeled_X =  copy.deepcopy(labeled_X_init)
		labeled_y =  copy.deepcopy(labeled_y_init)
		res_acc,res_auc, res_f1 = select_batch(bs, quota, unlabeled_X, unlabeled_y, labeled_X, labeled_y, test_X, test_y, AL_strategy)
		res_acc_total.append(res_acc) 
		res_auc_total.append(res_auc) 
		res_f1_total.append(res_f1)

	# cal mean & standard deviation
	acc_m = []
	auc_m = []
	f1_m = []
	for i in range(len(res_acc_total)):
		acc_m.append(get_aubc(quota, bs, res_acc_total[i]))
		auc_m.append(get_aubc(quota, bs, res_auc_total[i]))
		f1_m.append(get_aubc(quota, bs, res_f1_total[i]))
		print(str(i)+': '+str(acc_m[i])+' '+str(auc_m[i])+' '+str(f1_m[i]))
	mean_acc,stddev_acc = get_mean_stddev(acc_m)
	mean_auc,stddev_auc = get_mean_stddev(auc_m)
	mean_f1,stddev_f1 = get_mean_stddev(f1_m)
	
	print('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc))
	print('mean auc: '+str(mean_auc)+'. std dev auc: '+str(stddev_auc))
	print('mean f1: '+str(mean_f1)+'. std dev f1: '+str(stddev_f1))
	
	file_name_totres =  ds_name + '_total_result_' +  strategy + '_' + str(bs) + '.txt'
	file_totres =  open(os.path.join(os.path.abspath('..') + '/result', '%s' % file_name_totres),'w')
	
	for i in range(len(acc_m)):
		tmp0 = str(i)+': '+str(acc_m[i])+' '+str(auc_m[i])+' '+str(f1_m[i]) + '\n'
		file_totres.writelines(tmp0)
	file_totres.writelines(str(mean_acc)+' '+str(stddev_acc)+'\n')
	file_totres.writelines(str(mean_auc)+' '+str(stddev_auc)+'\n')
	file_totres.writelines(str(mean_f1)+' '+str(stddev_f1)+'\n')
	
	# cal avg
	res_acc = cal_average(res_acc_total) 
	res_auc = cal_average(res_auc_total) 
	res_f1 = cal_average(res_f1_total)

	file_name_res = ds_name + '_result_' +  strategy + '_' + str(bs) + '.txt'
	file_res =  open(os.path.join(os.path.abspath('..') + '/result', '%s' % file_name_res),'w')

	for i in range(0,len(res_acc)):
		tmp1 = str(round(res_acc[i],4)) + ' ' + str(round(res_auc[i],4)) + ' ' + str(round(res_f1[i],4)) + '\n'
		file_res.writelines(tmp1)

	file_res.close()
	file_totres.close()


if __name__ == '__main__':
    main()
