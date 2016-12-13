#!/usr/bin/python2.7
import numpy as np
import re, csv
from sklearn import preprocessing

# Loads different column from our base dataset into different numpy array, we will use scikit preprocessing tools
# encode text fields into usable features
dataset_num = np.genfromtxt("base_2010.csv", delimiter=',', dtype=None, skip_header=1, usecols=(1,2,5,6,7,8,10,14,15,19))
dataset_str = np.genfromtxt("base_2010.csv", delimiter=',', dtype=None, skip_header=1, usecols=(17,20,21))

label_index = 0
num_row = dataset_num.shape[0]

# we will turn the labels into binary form which means y==1 iff premature, y==0 otherwise
y = np.array([1 if int(row[label_index])==1 else 0 for row in dataset_num]).reshape((num_row,1))
dataset_num = np.delete(dataset_num, label_index, axis = 1)

# we will encode the text fields into usable features and later add them to our dataset
le_postal_code = preprocessing.LabelEncoder()
coded_postal_code = le_postal_code.fit_transform(dataset_str[:,1])
coded_socio_encon = np.array(coded_socio_encon).reshape((num_row,1))

le_country = preprocessing.LabelEncoder()
coded_country = le_country.fit_transform(dataset_str[:,2])
coded_country = np.array(coded_country).reshape((num_row,1))

le_socio_econ = preprocessing.LabelEncoder()
coded_socio_encon = le_socio_econ.fit_transform(dataset_str[:,0])
coded_postal_code = np.array(coded_postal_code).reshape((num_row,1))

# add encoded text fields and binarized labels to the dataset
dataset = np.c_[dataset_num, coded_socio_encon,coded_postal_code,coded_country,y]

np.save("dataset", dataset)