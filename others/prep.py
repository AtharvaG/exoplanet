# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:03:52 2018

@author: Atharv
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 

df = pd.read_csv('planets.csv', skiprows = 358) 

dt_y = dataTransit.iloc[:, cols2exclude]  #when we have column values 
dt_X = dataTransit.iloc[:, ~dataTransit.columns.isin(dataTransit.columns[cols2exclude])] 
dt_X.reset_index(drop = True, inplace = True)
dt_X.dropna(axis =1 , how = "all", inplace = True)
dt_X = dt_X.drop(['pl_pnum'], axis =1 )

pca_df = dt_std_df.loc[:, column]   #when we have column names 

from sklearn.preprocessing import Imputer 
values = dt_X.values  
imputer = Imputer()
transformed_values = imputer.fit_transform(values)    
from sklearn.preprocessing import StandardScaler
dt_std = StandardScaler().fit_transform(transformed_values)   #fills nan values with mean of the column values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
accuracy = regressor.score(X_test,y_test)
from sklearn.metric import accuracy_score
y_pred = regressor.predict(X_test)
accuracy_score(y_test,y_pred)

from sklean.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, p =2)
knn.fit(X_train,y_train)
knn.score(x_test,y_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.score(X_test, Y_test)

