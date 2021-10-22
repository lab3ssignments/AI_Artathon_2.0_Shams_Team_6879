# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:44:34 2021

@author: MSI-PC
"""




from utils import readExcel, saveExcel
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.cluster as c
import matplotlib.pyplot as plt
from PIL import Image
import time
# read the Data
basePath = r'D:\Projects\shareif/'

savePath1=r'D:\Projects\shareif/art.xlsx'


testName='art111'


test = readExcel(savePath1, sheet_name = 'art')

test.head()




#test1 = t[test['Mud Type']=='KCL/Polymer'].drop(0,axis=0) 
test1 = test.iloc[range(1,361),range(2,22)].values
x = []
r = np.zeros((12,30))
for k in range(20):
	r= test1[:,k].reshape((12,30))
	
	x.append(r)



for i in range(14):
	plt.imshow(x[i])
	plt.show()





plt.imshow(x[0])
plt.imshow(x[1])
plt.imshow(x[2])
plt.imshow(x[3])
plt.imshow(x[4])
plt.imshow(x[5])
plt.imshow(x[6])














