#!/usr/bin/python2.7
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

#load data
dataset = np.load("../dataset/REF2.npy")
training_set = dataset[:, 0:-1]
lables = dataset[:, -1]

#Linear classifier
for loss in ['perceptron','hinge','modified_huber']:
	for eta0 in [0.001, 0.01, 0.1, 1]:
		clf = SGDClassifier(loss=loss, penalty='l2', learning_rate='constant', eta0=eta0)
		scores = cross_val_score(clf, training_set, lables, cv =5, scoring='f1')
		print "printing scores for SGDClassifier [loss = (%s) learing_rate=(%s)]:" %(loss, eta0)
		print scores