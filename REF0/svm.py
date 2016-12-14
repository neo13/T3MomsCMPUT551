#!/usr/bin/python2.7
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

#load data
dataset = np.load("../dataset/REF2.npy")
training_set = dataset[:, 0:-1]
lables = dataset[:, -1]

#define params
C = [0.01, 0.1, 0.5, 1, 5]
gamma = [0.001, 0.03, 0.1, 1]
coef_list = [0.1,1,3]
print "SVM learning is running ..."
for c in C:
	clf = svm.SVC(kernel = 'linear', C = c)
	scores = cross_val_score(clf, training_set, lables, cv =3, scoring='roc_auc', n_jobs=-1)
	print "printing scores for SVM (linear) [C=%s]:" %(c)
	print scores

for c in C:
	for g in gamma:
		clf = svm.SVC(kernel = 'rbf', C = c, gamma=g)
		scores = cross_val_score(clf, training_set, lables, cv =3, scoring='f1')
		print "printing scores for SVM (rbf) [C=%s]:" %(c)
		print scores

for c in C:
	for g in gamma:
		for coef in coef_list:
			clf = svm.SVC(kernel = 'sigmoid', C = c, coef0=coef)
			scores = cross_val_score(clf, training_set, lables, cv =3, scoring='f1')
			print "printing scores for SVM (sigmoid) [C=%s, coef=%s]:" %(c, coef)
			print scores
