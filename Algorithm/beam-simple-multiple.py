#--- only for my model---#

#library
import copy
import os
import random
import warnings
import datetime
import time
import math
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier

import heapq
import sys



def split_data(dataset_filepath, test_size, n_labeled):
	#X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
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
	#print(test_data)
	#print(len(test_data))

	temp1=copy.deepcopy(temp)

	Flag = False
	#ount = 0
	#print(Flag)
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
		#print(count)
		#count = count + 1
	#print(Flag)


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

def get_best_point(model, dict_data,labeled_data,unlabeled_data,test_data):

	test_X,test_y = form_dataset(dict_data,test_data)
	scalar=StandardScaler()
	old_X,old_y = form_dataset(dict_data,labeled_data)
	old_X_tf = scalar.fit_transform(old_X)
	test_X_tf=scalar.transform(test_X)
	model.fit(old_X_tf,old_y)
	test_y_pred= model.predict(test_X_tf)
	old_acc = accuracy_score(test_y, test_y_pred) * 1.0
	acc_increase = []
	for i in range(len(unlabeled_data)):
		new_data = copy.deepcopy(labeled_data)
		new_data.append(unlabeled_data[i])
		new_X,new_y = form_dataset(dict_data,new_data)
		new_X_tf = scalar.fit_transform(new_X)
		test_X_tf=scalar.transform(test_X)
		model.fit(new_X_tf,new_y)
		test_y_pred2= model.predict(test_X_tf)
		new_acc = accuracy_score(test_y, test_y_pred2) * 1.0
		acc_increase.append(float(new_acc - old_acc))
	idx = acc_increase.index(max(acc_increase))
	return max(acc_increase),unlabeled_data[idx]

def get_best_point_partial(model, dict_data,labeled_data,unlabeled_data,test_data):
	all_X,all_y = form_dataset(dict_data,labeled_data+unlabeled_data+test_data)
	test_X,test_y = form_dataset(dict_data,test_data)
	scalar=StandardScaler()
	old_X,old_y = form_dataset(dict_data,labeled_data)
	old_X_tf = scalar.fit_transform(old_X)
	test_X_tf=scalar.transform(test_X)
	model.partial_fit(old_X_tf,old_y,classes=np.array(list(set(all_y))))
	test_y_pred= model.predict(test_X_tf)
	old_acc = accuracy_score(test_y, test_y_pred) * 1.0
	acc_increase = []
	for i in range(len(unlabeled_data)):
		new_data = copy.deepcopy(labeled_data)
		new_data.append(unlabeled_data[i])
		new_X,new_y = form_dataset(dict_data,new_data)
		new_X_tf = scalar.fit_transform(new_X)
		test_X_tf=scalar.transform(test_X)
		model.partial_fit(new_X_tf,new_y,classes=np.array(list(set(all_y))))
		test_y_pred2= model.predict(test_X_tf)
		new_acc = accuracy_score(test_y, test_y_pred2) * 1.0
		acc_increase.append(float(new_acc - old_acc))
	idx = acc_increase.index(max(acc_increase))
	return max(acc_increase),unlabeled_data[idx]

def get_best_seq(dict_data,labeled_data,unlabeled_data,test_data):
	num_node = 3
	kernel = 1.0 * RBF(1.0)
	quota = len(unlabeled_data)
	#clf1 = LR(penalty='l2')
	#clf2 = LDA()
	clf3 = SVC(kernel = 'rbf',gamma='auto')
	#clf4 = LinearSVC(random_state=0, tol=1e-5)
	#clf5 = GaussianProcessClassifier(kernel=kernel,random_state=0)
	clf1 = SGDClassifier(loss = 'log',penalty = 'l2')
	clf4 = SGDClassifier(loss = 'hinge',penalty = 'l2')
	init_1 = copy.deepcopy(labeled_data)
	#init_2 = copy.deepcopy(labeled_data)
	init_3 = copy.deepcopy(labeled_data)
	init_4 = copy.deepcopy(labeled_data)
	#init_5 = copy.deepcopy(labeled_data)
	seq = []
	seq.append(init_1)
	max_query1,idx_query1 = get_best_point_partial(clf1, dict_data,init_1,unlabeled_data,test_data)
	seq[0].append(idx_query1)
	#seq.append(init_2)
	#max_query2,idx_query2 = get_best_point(clf2, dict_data,init_2,unlabeled_data,test_data)
	#seq[1].append(idx_query2)
	seq.append(init_3)
	max_query3,idx_query3 = get_best_point(clf3, dict_data,init_3,unlabeled_data,test_data)
	seq[1].append(idx_query3)
	seq.append(init_4)
	max_query4,idx_query4 = get_best_point_partial(clf4, dict_data,init_4,unlabeled_data,test_data)
	seq[2].append(idx_query4)
	#seq.append(init_5)
	#max_query5,idx_query5 = get_best_point(clf5, dict_data,init_5,unlabeled_data,test_data)
	#seq[4].append(idx_query5)
	seq_tmp = []
	res_tmp = []

	seq_un = []
	un_1 = copy.deepcopy(unlabeled_data)
	#un_2 = copy.deepcopy(unlabeled_data)
	un_3 = copy.deepcopy(unlabeled_data)
	un_4 = copy.deepcopy(unlabeled_data)
	#un_5 = copy.deepcopy(unlabeled_data)
	seq_un.append(un_1)
	#seq_un.append(un_2)
	seq_un.append(un_3)
	seq_un.append(un_4)
	#seq_un.append(un_5)
	#print(seq_un)
	#print(idx_query1)
	#print(idx_query2)
	#print(idx_query3)
	#print(idx_query4)
	#print(idx_query5)
	seq_un[0].remove(idx_query1)
	#seq_un[1].remove(idx_query2)
	seq_un[1].remove(idx_query3)
	seq_un[2].remove(idx_query4)
	#seq_un[4].remove(idx_query5)

	quota = quota -1

	while (quota > 0):
		print(quota)
		seq_tmp = []
		res_tmp = []
		sequn_tmp = []
		#clf1
		for i in range(len(seq)):
			res = []
			max_query,idx_query = get_best_point_partial(clf1, dict_data,seq[i],seq_un[i],test_data)
			tmp = copy.deepcopy(seq[i])
			tmp.append(idx_query)
			seq_tmp.append(tmp)
			res_tmp.append(max_query)
			tmp_un = copy.deepcopy(seq_un[i])
			tmp_un.remove(idx_query)
			sequn_tmp.append(tmp_un)
		'''
		#clf2
		for i in range(len(seq)):
			max_query,idx_query = get_best_point(clf2, dict_data,seq[i],seq_un[i],test_data)
			tmp = copy.deepcopy(seq[i])
			tmp.append(idx_query)
			seq_tmp.append(tmp)
			res_tmp.append(max_query)
			tmp_un = copy.deepcopy(seq_un[i])
			tmp_un.remove(idx_query)
			sequn_tmp.append(tmp_un)
		'''
		#clf3
		for i in range(len(seq)):
			max_query,idx_query = get_best_point(clf3, dict_data,seq[i],seq_un[i],test_data)
			tmp = copy.deepcopy(seq[i])
			tmp.append(idx_query)
			seq_tmp.append(tmp)
			res_tmp.append(max_query)
			tmp_un = copy.deepcopy(seq_un[i])
			tmp_un.remove(idx_query)
			sequn_tmp.append(tmp_un)
		#clf4
		for i in range(len(seq)):
			max_query,idx_query = get_best_point_partial(clf4, dict_data,seq[i],seq_un[i],test_data)
			tmp = copy.deepcopy(seq[i])
			tmp.append(idx_query)
			seq_tmp.append(tmp)
			res_tmp.append(max_query)
			tmp_un = copy.deepcopy(seq_un[i])
			tmp_un.remove(idx_query)
			sequn_tmp.append(tmp_un)
		'''
		#clf5
		for i in range(len(seq)):
			max_query,idx_query = get_best_point(clf5, dict_data,seq[i],seq_un[i],test_data)
			tmp = copy.deepcopy(seq[i])
			tmp.append(idx_query)
			seq_tmp.append(tmp)
			res_tmp.append(max_query)
			tmp_un = copy.deepcopy(seq_un[i])
			tmp_un.remove(idx_query)
			sequn_tmp.append(tmp_un)
		'''
		quota = quota - 1




		res2 = list(map(res_tmp.index,heapq.nlargest(num_node,res_tmp)))
		seq[:]=[]
		seq_un[:]=[]
		for i in res2:
			seq.append(seq_tmp[i])
			seq_un.append(sequn_tmp[i])
		#print(seq)
		#print(seq_un)
		if(quota == 0):
			max_query_final = seq_tmp[res_tmp.index(max(res_tmp))]
			#print(max_query_final)

	res_acc = []
	res_auc = []
	res_f1 = []

	for idx in range(20,len(max_query_final)+1):
		#print(res_best[0:idx])
		acc,auc,f1 = test_model2(dict_data,max_query_final[0:idx],test_data)
		print(str(round(acc,4))+', '+str(round(auc,4))+', '+str(round(f1,4)))
		res_acc.append(acc)
		res_auc.append(auc)
		res_f1.append(f1)
	res = []
	res.append(res_acc)
	res.append(res_auc)
	res.append(res_f1)


	return res


def test_model(dict_data,labeled_data,test_data):
	X_train, y_train = form_dataset(dict_data,labeled_data)
	X_test, y_test = form_dataset(dict_data,test_data)
	scalar=StandardScaler()
	X_train_tf = scalar.fit_transform(X_train)
	X_test_tf=scalar.transform(X_test)
	clf = SVC(kernel = 'rbf',gamma='auto')
	clf.fit(X_train_tf,y_train)
	y_pred = clf.predict(X_test_tf)
	#print(y_train)
	#print(y_test)
	#print(y_pred)
	acc = accuracy_score(y_test, y_pred) * 1.0
	auc = roc_auc_score(y_test, y_pred) * 1.0
	f1 = f1_score(y_test, y_pred) * 1.0
	return acc, auc, f1

def test_model2(dict_data,labeled_data,test_data):
	X_train, y_train = form_dataset(dict_data,labeled_data)
	X_test, y_test = form_dataset(dict_data,test_data)
	#X_train = labeled_X
	#X_test = test_X
	#y_train = labeled_y
	#y_test = test_y

	#clf = LR() 
	scalar=StandardScaler() 
	X_train_tf = scalar.fit_transform(X_train)
	X_test_tf=scalar.transform(X_test)
	#grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
	#grid.fit(X, y_train)
	#clf.fit(X_train,y_train) 
	#clf = LR(penalty='l1')
	clf = SVC(kernel = 'rbf',gamma='auto')
	#kernel = 1.0 * RBF(1.0)
	#clf = GaussianProcessClassifier(kernel=kernel,random_state=0)
	#clf.fit(X_train_tf,y_train) 
	#y_pred = clf.predict(X_test_tf)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	acc = accuracy_score(y_test, y_pred) * 1.0

	idx = int(np.max(y_test))
	y_test1=np.zeros((len(y_pred),idx),dtype = 'int')
	y_pred1=np.zeros((len(y_pred),idx),dtype = 'int')

	for i in range(0,len(y_pred)):
		y_test1[i][int(y_test[i])-1] = 1
		y_pred1[i][int(y_pred[i])-1] = 1

	auc = roc_auc_score(y_test1,y_pred1, multi_class='ovr', average='macro')

	f1 = f1_score(y_test1, y_pred1,average = 'macro') * 1.0
	#acc=clf.score(X_test,y_test)
	return acc, auc, f1

def cal_average(res_total):
	res = []
	for i in range(len(res_total[0])):
		sum = 0.0
		for j in range(len(res_total)):
			sum = sum + res_total[j][i]
		sum = float((sum *1.0) / (len(res_total)*1.0))
		res.append(sum)
	return res

class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass


def main():

    warnings.filterwarnings('ignore')
    ds_name = str(sys.argv[1])
    n_labeled = int(sys.argv[2])
    rs = 4666
    random.seed(rs)
    test_size = 0.4
    #n_labeled = 20
    T1 = 0
    res_acc_t = []
    res_auc_t = []
    res_f1_t = []

    startt = datetime.datetime.now()

    res = []
    sys.stdout = Logger(ds_name+ '_' + str(n_labeled) + '_best_log.txt')
    for i in range(1):
      T1 = T1 + 1
      print('This is %d iteration.' % T1)
      starttime = datetime.datetime.now()
      dataset_filepath = os.path.join(os.path.abspath('..') + '/Dataset/dealeddata', '%s.txt' % ds_name)
      test_size = 0.4
      #n_labeled = 20

      dict_data,labeled_data,test_data,unlabeled_data = split_data(dataset_filepath, test_size, n_labeled)
      dict_train_idx=labeled_data+unlabeled_data

      quota = len(unlabeled_data)

      res.append(get_best_seq(dict_data,labeled_data,unlabeled_data,test_data))
      endtime = datetime.datetime.now()
      print('Running time of task %s is %f' % (i, float((endtime-starttime).seconds)))

    for rg in res:
      res_acc_t.append(rg[0])
      res_auc_t.append(rg[1])
      res_f1_t.append(rg[2])

    res_acc = cal_average(res_acc_t)
    res_auc = cal_average(res_auc_t)
    res_f1 = cal_average(res_f1_t)
    file_name_res = ds_name + '_' + str(n_labeled) + '_result_best.txt'
    file_res =  open(os.path.join(os.path.abspath('..') + '/resultbest', '%s' % file_name_res),'w')

    for i in range(0,len(res_acc)):
        tmp1 = str(round(res_acc[i],4)) + ' ' + str(round(res_auc[i],4)) + ' ' + str(round(res_f1[i],4)) + '\n'
        file_res.writelines(tmp1)

    file_res.close()

    endt = datetime.datetime.now()
    print('Total running time is %f' % float((endt-startt).seconds))





if __name__ == '__main__':
    main()
