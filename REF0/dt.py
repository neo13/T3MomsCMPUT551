#!/usr/bin/python2.7
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#load data
dataset = np.load("../dataset/REF2.npy")
training_set = dataset[:, 0:-1]
lables = dataset[:, -1]

#DecisionTreeClassifier
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, training_set, lables, cv =5, scoring='recall')
print "printing scores for Decision Tree Classifier:"
print scores

#RandomForestClassifier
for estimators in [5, 10, 20, 50, 100]:
	clf = RandomForestClassifier(n_estimators=estimators)
	scores = cross_val_score(clf, training_set, lables, cv =3, scoring='recall')
	print "printing scores for Random Forest Classifier [#estimators = (%s)]:" %(estimators)
	print scores

#ExtraTreesClassifier
for estimators in [5, 10, 20, 50, 100]:
	clf = ExtraTreesClassifier(n_estimators=estimators)
	scores = cross_val_score(clf, training_set, lables, cv =3, scoring='recall')
	print "printing scores for Extra Trees Classifier [#estimators = (%s)]:" %(estimators)
	print scores

#AdaBoostClassifier
for estimators in [5, 10, 20, 50, 100]:
	clf = AdaBoostClassifier(n_estimators=estimators)
	scores = cross_val_score(clf, training_set, lables, cv =3, scoring='recall')
	print "printing scores for AdaBoost Classifier [#estimators = (%s)]:" %(estimators)
	print scores