import copy
import os
import random
import warnings
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
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


# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import SVM, LogisticRegression, SklearnAdapter
from libact.query_strategies import UncertaintySampling, RandomSampling, ActiveLearningByLearning, HintSVM, QueryByCommittee, QUIRE, VarianceReduction, DWUS
from libact.labelers import IdealLabeler

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

		
		acc.append(acc_T)
		auc.append(auc_T)
		f1.append(f1_T)
		print(str(round(acc_T,4)) +' '+str(round(auc_T,4)) +' ' + str(round(f1_T,4)))

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

	model.fit(X_train_tf, y_train_1)
	y_pred = model.predict(X_test_tf)
	acc_T = accuracy_score(y_test, y_pred) * 1.0
	auc_T = roc_auc_score(y_test, y_pred) * 1.0
	f1_T = f1_score(y_test, y_pred) * 1.0


	acc.append(acc_T)
	auc.append(auc_T)
	f1.append(f1_T)
	print(str(round(acc_T,4)) +' '+str(round(auc_T,4)) +' ' + str(round(f1_T,4)))
	return acc, auc, f1

def split_data(dataset_filepath, test_size, n_labeled):
	X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
	#X, y = load_svmlight_file(dataset_filepath)
	#X = X.toarray().tolist()
	#y = y.tolist()
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

	for i in labeled_data_temp:
		labeled_data.append(temp1[i])
		temp.remove(temp1[i])

	unlabeled_data=copy.deepcopy(temp)

	return dict_data, labeled_data, test_data, unlabeled_data

def split_train_test_data(dict_data, labeled_data, test_data, unlabeled_data,n_labeled):
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    x_temp_l,y_temp_l=form_dataset(dict_data,labeled_data)
    x_temp_u,y_temp_u=form_dataset(dict_data,unlabeled_data)
    X_train=x_temp_l+x_temp_u
    y_train=y_temp_l+y_temp_u

    X_test,y_test=form_dataset(dict_data,test_data)

    #while len(np.unique((y_train[:n_labeled]))) != 2:
    #    X_train, X_test, y_train, y_test = \
    #        train_test_split(X, y, test_size=test_size)

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds

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

class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass


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

	model_name = str(sys.argv[2])
	
	bs = int(sys.argv[3])
	
	iternum = int(sys.argv[4])
	
	n_labeled = int(sys.argv[5])
	
	sys.stdout = Logger('./logfile/'+ds_name+ '_'  + model_name + '_' + str(bs) + '_' + str(n_labeled) + '_' + str(iternum)  +'_log.txt')
	random.seed(4666)

	warnings.filterwarnings('ignore')


	res_albl_acc_total=[]
	res_albl_auc_total=[]
	res_albl_f1_total=[]

	res_us_acc_total = []
	res_us_auc_total = []
	res_us_f1_total = []

	res_hint_acc_total = []
	res_hint_auc_total = []
	res_hint_f1_total = []

	res_qbc_acc_total = []
	res_qbc_auc_total = []
	res_qbc_f1_total = []

	res_quire_acc_total = []
	res_quire_auc_total = []
	res_quire_f1_total = []

	res_vr_acc_total = []
	res_vr_auc_total = []
	res_vr_f1_total = []


	res_dwus_acc_total = []
	res_dwus_auc_total = []
	res_dwus_f1_total = []

	res_acc_total = []
	res_auc_total = []
	res_f1_total = []
	
	dataset_filepath = os.path.join(os.path.abspath('..') + '/Dataset/dealeddata', '%s.txt' % ds_name)
	print(dataset_filepath)



	test_size = 0.4
	T1 = 0

	num_of_iter = iternum
	while(T1 < num_of_iter): 
		T1 = T1 + 1
		print('Iteration: %s' % T1)
		
		dict_data=dict()
		labeled_data=[]
		test_data=[]
		unlabeled_data=[]

		dict_data,labeled_data,test_data,unlabeled_data = split_data(dataset_filepath, test_size, n_labeled)


		dict_train_idx=labeled_data+unlabeled_data


		trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
			split_train_test_data(dict_data, labeled_data, test_data, unlabeled_data,n_labeled)
		trn_ds_us = copy.deepcopy(trn_ds)
		trn_ds_albl = copy.deepcopy(trn_ds)
		trn_ds_hint = copy.deepcopy(trn_ds)
		trn_ds_qbc = copy.deepcopy(trn_ds)
		trn_ds_quire = copy.deepcopy(trn_ds)
		trn_ds_vr = copy.deepcopy(trn_ds)
		trn_ds_dwus = copy.deepcopy(trn_ds)
		quota = len(unlabeled_data)

		#us

		if model_name == 'us':
			quota_us = quota
			qs_us = UncertaintySampling(trn_ds_us, model=LogisticRegression(C=0.1),method='entropy')
			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_us, auc_us, f1_us = run(trn_ds_us, tst_ds, lbr, model, qs_us, quota_us,n_labeled)

			res_acc_total.append(acc_us)
			res_auc_total.append(auc_us)
			res_f1_total.append(f1_us)

			print('us!')


		#albl
		if model_name == 'albl':
			quota_albl = quota
			qs_albl = ActiveLearningByLearning(
	            trn_ds_albl, # Dataset object
	            T=quota_albl, # qs.make_query can be called for at most 100 times
	            query_strategies=[
	                UncertaintySampling(trn_ds_albl, model=LogisticRegression(C=1.)),
	                UncertaintySampling(trn_ds_albl, model=LogisticRegression(C=.01)),
	                HintSVM(trn_ds_albl)
	                ],
	            model=LogisticRegression()
	        )
			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_albl, auc_albl, f1_albl = run(trn_ds_albl, tst_ds, lbr, model, qs_albl, quota_albl,n_labeled)

			res_acc_total.append(acc_albl)
			res_auc_total.append(auc_albl)
			res_f1_total.append(f1_albl)

			print('albl!')

		#HintSVM
		if model_name == 'HintSVM':
			quota_hint = quota
			qs_hint = HintSVM(trn_ds_hint, cl=1.0, ch=1.0)
			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_hint, auc_hint, f1_hint  = run(trn_ds_hint, tst_ds, lbr, model, qs_hint, quota_hint,n_labeled)

			res_acc_total.append(acc_hint)
			res_auc_total.append(auc_hint)
			res_f1_total.append(f1_hint)

			print('hint!')

		#QBC
		if model_name == 'qbc':
			quota_qbc = quota

			adapter1 = SklearnAdapter(LDA())

			kernel_adapter2 = 1.0 * RBF(1.0)
			adapter2 = SklearnAdapter(GaussianProcessClassifier(kernel=kernel_adapter2,random_state=0))
			qs_qbc = QueryByCommittee(trn_ds_qbc, models = [LogisticRegression(penalty = 'l2'), 
				SVM(kernel = 'linear'), 
				SVM(kernel = 'rbf'),  adapter1],disagreement = 'vote')
			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_qbc, auc_qbc, f1_qbc  = run(trn_ds_qbc, tst_ds, lbr, model, qs_qbc, quota_qbc,n_labeled)

			res_acc_total.append(acc_qbc)
			res_auc_total.append(auc_qbc)
			res_f1_total.append(f1_qbc)

			print('qbc!')

		#QUIRE
		if model_name == 'QUIRE':
			quota_quire = quota

			qs_quire = QUIRE(trn_ds_quire)

			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_quire, auc_quire, f1_quire  = run(trn_ds_quire, tst_ds, lbr, model, qs_quire, quota_quire,n_labeled)

			res_acc_total.append(acc_quire)
			res_auc_total.append(auc_quire)
			res_f1_total.append(f1_quire)

			print('quire!')

		#vr
		if model_name == 'vr':
			quota_vr = quota
			qs_vr = VarianceReduction(trn_ds_vr,model=LogisticRegression(solver='liblinear', multi_class="ovr"),
	                sigma=0.1)
			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_vr, auc_vr, f1_vr = run(trn_ds_vr, tst_ds, lbr, model, qs_vr, quota_vr,n_labeled)

			res_acc_total.append(acc_vr)
			res_auc_total.append(auc_vr)
			res_f1_total.append(f1_vr)

			print('vr!')


		#dwus
		if model_name == 'dwus':
			quota_dwus = quota

			qs_dwus = DWUS(trn_ds_dwus)

			model = SVC(kernel = 'rbf',gamma='auto')
			#kernel = 1.0 * RBF(1.0)
			#model = GaussianProcessClassifier(kernel=kernel,random_state=0)
			lbr = IdealLabeler(fully_labeled_trn_ds)
			acc_dwus, auc_dwus, f1_dwus  = run(trn_ds_dwus, tst_ds, lbr, model, qs_dwus, quota_dwus,n_labeled)


			res_acc_total.append(acc_dwus)
			res_auc_total.append(auc_dwus)
			res_f1_total.append(f1_dwus)

			print('dwus!')

	res_acc = cal_average(res_acc_total)
	res_auc = cal_average(res_auc_total)
	res_f1 = cal_average(res_f1_total)

	file_name  = ds_name + '_result_' + model_name + '_' + str(bs) + '_' + str(n_labeled) + '_' + str(iternum)  + '.txt'
	file_result = open(os.path.join(os.path.abspath('..') + '/result', '%s.txt' % file_name),'w')
	for i in range(len(res_acc)):
		tmp1 = str(round(res_acc[i],4)) + ' ' + str(round(res_auc[i],4)) + ' ' + str(round(res_f1[i],4)) + '\n'
		file_result.writelines(tmp1)
	file_result.close()

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
	
	file_name_totres =  ds_name + '_total_result_' +  model_name + '_' + str(bs) + '.txt'
	file_totres =  open(os.path.join(os.path.abspath('..') + '/result', '%s' % file_name_totres),'w')
	
	for i in range(len(acc_m)):
		tmp0 = str(i)+': '+str(acc_m[i])+' '+str(auc_m[i])+' '+str(f1_m[i]) + '\n'
		file_totres.writelines(tmp0)
	file_totres.writelines(str(mean_acc)+' '+str(stddev_acc)+'\n')
	file_totres.writelines(str(mean_auc)+' '+str(stddev_auc)+'\n')
	file_totres.writelines(str(mean_f1)+' '+str(stddev_f1)+'\n')
	
if __name__ == '__main__':
    main()
