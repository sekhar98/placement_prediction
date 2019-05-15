# -*- coding: utf-8 -*-
"""Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset=pd.read_csv("File.csv")
#replace the file in "" of read_csv with your own file about plcements data in your college.

#x includes all the variables that you are considering for building the model. Here 21,23,32 are the columns of dependant variables.
x=dataset.iloc[:, [21,23,32]].values 
print("Enter 0,3 or 5")
p=int(input())

#y is the output for the selected variables. These can be either '0' or '1'. If it is '0', then outcome is negative and '1' means a positive outcome.
#If your dataset doesn't have these values of '0' or '1' columns, use if statements along with df.iat[i,j] to fill the column with '0' or '1'. 
#Here, p=0 indicates whether a candidate has secured a placement worth of 0LPA or not. p=3 indicates whether a candidate has secured a placement of worth 3LPA or more than that.
#Build these columns in dataset for various values of "p". 
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

#Below Naive Bayes code has also been included as comments. Uncomment the code of Naive Bayes and comment the logisticRegression code.
#Check out which classification algorithm would give more accuracy for your dataset.

#naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
#classifier.fit(x_train,y_train)


#logisticregression 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#This below code is used when you dont want to split the dataset into training and test datasets but you would rather want to use an input dataset.
#The outcome values of this input dataset has to be predicted.

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

