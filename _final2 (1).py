#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pickle


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


# In[2]:


# Read dataset to pandas dataframe
dataset = pd.read_csv('Employee.csv') 

# Create an instance of LabelEncoder
le = LabelEncoder()

# Fit and transform the 'Education' column
dataset['Education'] = le.fit_transform(dataset['Education'])
dataset['City'] = le.fit_transform(dataset['City'])
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['EverBenched'] = le.fit_transform(dataset['EverBenched'])





X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train,x_test,y_train,y_test=train_test_split(X,y)

param_grid = {
    'C': [1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf', 'linear'],
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_

model_svm_tuned = SVC(
    C=best_params['C'],
    gamma=best_params['gamma'],
    kernel=best_params['kernel']
)

model_svm_tuned.fit(x_train, y_train)
predictions_tuned_svm = model_svm_tuned.predict(x_test)

training_score = model_svm_tuned.score(x_train, y_train)
test_score = model_svm_tuned.score(x_test, y_test)




pickle.dump(model_svm_tuned,open('svm_model.pkl','wb'))




