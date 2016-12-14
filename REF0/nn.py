#!/usr/bin/python2.7
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

#load data
dataset = np.load("../dataset/REF2.npy")
training_set = dataset[:, 0:-1]
lables = dataset[:, -1]

print "NN learning is running ..."
for alpha in [0.005, 0.01, 0.03, 0.05]:
	size = int(5.377131332/(alpha*16))
	clf = MLPClassifier(solver='adam', hidden_layer_sizes=(8, size), random_state=1)
	scores = cross_val_score(clf, training_set, lables, cv =3, scoring='f1')
	print "printing scores for NN (linear) [size=(8, %s)]:" %(size)
	print scores
