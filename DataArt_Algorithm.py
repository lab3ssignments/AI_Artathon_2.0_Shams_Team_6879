"""
This file is cereated to clean the data:
	take out reapeated rows 
	normalize and scale data.
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

# read the Data
basePath = r'D:\Projects\shareif/'

savePath1=r'D:\Projects\shareif/art.xlsx'


testName='art111'


test = readExcel(savePath1, sheet_name = 'art')

test.head()




#test1 = t[test['Mud Type']=='KCL/Polymer'].drop(0,axis=0) 
test1 = test.iloc[range(1,364),range(2,14)]
# test1 = test1[test1['MBT'] <35]

# inputTest = test.columns[0:11]
# outputTest = test.columns[5:7]

X = test1.iloc[:,0:11].values.astype(float)
Y = test1.iloc[:,11].values.astype(float).reshape(-1,1)

# =============================================================================
# # DATA ENCODING #
# =============================================================================

# DATA SPLIT #

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

# Features Scale 
#sc = preprocessing.StandardScaler()
sc = preprocessing.MinMaxScaler()
 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



sc1 = preprocessing.StandardScaler()
sc1 = preprocessing.MinMaxScaler()

Y_train = sc1.fit_transform(Y_train)
Y_test = sc1.transform(Y_test)
# =============================================================================
# # Part 2 - Building the ANN
# =============================================================================

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

# # Adding the input layer and the first hidden layer
# ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# # # Adding the second hidden layer
# ann.add(tf.keras.layers.Dense(units=128, activation='relu'))

# # # Adding the second hidden layer
# ann.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Adding the second hidden layer
# ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the third  hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

# Adding the second hidden layer
# ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding the fourth hidden layer
ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

# Adding the fourth hidden layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1)) 


# Part 3 - Training the ANN

# =============================================================================
# # Compiling the ANN
# =============================================================================

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
 	initial_learning_rate=1e-3,
 	decay_steps=1000,
 	decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

ann.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])
#ann.compile(optimizer = 'Adam', loss = 'mean_absolute_error', metrics=['mean_absolute_error'])

# Training the ANN on the Training set
checkpoint_path=basePath+testName+'.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
												save_weights_only=True,
												save_freq = 5000,
												verbose=1)
# model.load_weights(checkpoint_path)




# =============================================================================
# # Training the ANN on the Training set
# =============================================================================
ann.fit(X_train, Y_train, batch_size = 32, epochs = 150, callbacks=[cp_callback])



#ann.fit(X_train, Y_train, batch_size = 32, epochs = 600, callbacks=[cp_callback])

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation



# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Part 5
# Predicting the Training set results
y_predT = ann.predict(X_train)
#print(np.concatenate((y_predT.reshape(len(y_predT),1), Y_train.reshape(len(Y_train),1)),1))
TotalT = np.concatenate((y_predT.reshape(len(y_predT),1), Y_train.reshape(len(Y_train),1)),1)
TotalpdT = pd.DataFrame(TotalT, columns=(['Y_pred','Y_train']))
TotalpdT.index = map(int, TotalpdT.index)
TotalpdT.reset_index(inplace=True)
TotalpdT['difference'] = (TotalpdT['Y_pred']-TotalpdT['Y_train']).abs()*100/TotalpdT['Y_train']
TotalpdT.loc['mean'] = TotalpdT.mean()

TotalpdT.to_excel(basePath+testName+'Train'+'.xlsx')

# Predicting the Testing set results
y_pred = ann.predict(X_test)
Total = np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1)
#Total = np.concatenate((y_predT.reshape(len(y_predT),1), Y_train.reshape(len(Y_train),1)),1)
#plot the output
Totalpd = pd.DataFrame(Total, columns=(['Y_pred','Y_test']))
Totalpd.index = map(int, Totalpd.index)
Totalpd.reset_index(inplace=True)
Totalpd['difference'] = (Totalpd['Y_pred']-Totalpd['Y_test']).abs()*100/Totalpd['Y_test']
Totalpd.loc['mean'] = Totalpd.mean()
Totalpd.to_excel(basePath+testName+'.xlsx')


dataMin = int(Totalpd.iloc[:,[1,2]].min().min())
dataMax = int(Totalpd.iloc[:,[1,2]].max().max())
x = np.array(range(dataMin, dataMax))
y = pd.DataFrame(x)
y.index = map(int, x)


# =============================================================================
# 
# =============================================================================
ax = Totalpd['Y_test'].plot(kind='line',
				  color='red',
				  figsize=(14,8)
	)
ax1 = Totalpd['Y_pred'].plot(kind='line',
					color='green',
					figsize=(14,8),
					ax = ax
	)
ax.legend(['Actual Output','Predicted Output' ], loc = 'upper left', fontsize='x-large')
plt.show()
#ax.figure.savefig(basePath+testName+'line.jpg')
# =============================================================================
# 
# =============================================================================
ax = Totalpd.plot(kind='scatter',
				  x = 'Y_test',
				  y='Y_pred',
				  color='red',
				  figsize=(14,8)
	)
# ax0 = Totalpd.plot(kind='scatter',
# 				  color='blue',
# 				  x = 'index',
# 				  y='Y_test',
# 				   ax=ax
# 	)
ax1 = y.plot(kind='line',
					color='green',
					figsize=(14,8),
					ax = ax
	)
ax.legend(['45 degree line','Data predected vs ground truth' ], loc = 'upper left', fontsize='x-large')
plt.show()
#ax.figure.savefig(basePath+testName+'scatter.jpg')
# =============================================================================
# 
# =============================================================================
ax = Totalpd['difference'].plot(kind='bar',
				  color='red',
				  figsize=(14,8)
	)
plt.axhline(y = 7, color = 'blue', linestyle = '-')
ax.legend(['Allowed Range' ,'Erorr Percentage'], loc = 'upper left', fontsize='x-large')
ax.set_ylim(ymin=0, ymax=100)
plt.show()
# ax.figure.savefig(basePath+testName+'error.jpg')
#=============================================================================
# 
# =============================================================================
ax = Totalpd['Y_test'].plot(kind='hist',
				  color='blue',
				  figsize=(14,8)
	)
ax.legend(['Histogram of testing data'], loc = 'upper left', fontsize='x-large')
plt.show()
# ax.figure.savefig(basePath+testName+'histTestData.jpg')
# =============================================================================
# 
# =============================================================================
ax = Totalpd['Y_pred'].plot(kind='hist',
				  color='blue',
				  figsize=(14,8)
	)
ax.legend(['Histogram of predicted data'], loc = 'upper left', fontsize='x-large')
plt.show()
# ax.figure.savefig(basePath+testName+'histPre.jpg')

# =============================================================================
# 
# =============================================================================
plt.hist(Totalpd['difference'])
plt.ylabel('Frequency')
plt.title('Histogram of Error')
plt.show()
# plt.savefig(basePath+testName+'histError.jpg')
# =============================================================================
# TRAINING VIS AND VERIFACATION
# =============================================================================
ax = TotalpdT['Y_train'].plot(kind='hist',
				  color='blue',
				  figsize=(14,8)
	)
ax.legend(['Histogram of training data'], loc = 'upper left', fontsize='x-large')
plt.show()
# ax.figure.savefig(basePath+testName+'histTrainData.jpg')
# =============================================================================
# 
# =============================================================================
ax = TotalpdT['Y_pred'].plot(kind='hist',
				  color='blue',
				  figsize=(14,8)
	)
ax.legend(['Histogram of predicted data'], loc = 'upper left', fontsize='x-large')
plt.show()
# ax.figure.savefig(r'D:\Projects\Proxec\\Python\DataProject\Training/'+testName+'histPre.jpg')











er = Totalpd[Totalpd['difference']>7]
