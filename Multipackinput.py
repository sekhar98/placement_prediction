# -*- coding: utf-8 -*-
"""Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset=pd.read_csv("FData2.csv")
x=dataset.iloc[:, [21,23,32]].values 
print("Enter 0,3 or 5")
p=int(input())
if(p==0) :
    y=dataset.iloc[:, 30].values
elif(p==3) :
    y=dataset.iloc[:, 34].values
elif(p==5) :
    y=dataset.iloc[:, 35].values
else :
    print("Please enter an integer from:0,3,5") 
    

#Spiltting between training and testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#x_train=x_train.reshape(1,-1)

x_t=x_test
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
#x=sc_x.fit_transform(x)

#x_test=pd.read_csv("test.csv")
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
#classifier.fit(x_train,y_train)
#logisticregression 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#for testing with input
#classifier.fit(x,y)


#prediction 
y_pred=classifier.predict(x_test)

#confusion matrix 
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,y_pred)

acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100

for i in range(len(y_pred)):
    print(x_t[i],y_pred[i]) 
print("Accuracy is:",acc)

