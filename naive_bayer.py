#Naive Bayers in Fraud Detection
#Naive Bayer's Formula P(A/B) = P(BIA)*P(A)/P(B) 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime 
#IMPORT DATA 
TRANSACTION_DATASET = pd.read_csv("C:\\Users\\e5603947\\OneDrive FIS\\Desktop\\MACHINE LEARNING Logistic Regression
X = TRANSACTION_DATASET 
X=X.drop("FRAUD", axis-1) 
X=X.drop("TIMESTAMP", axis=1) 
Y= TRANSACTION_DATASET.iloc[:, 14].values #OUTPUT 
#SPLIT 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.25, random_state=1) 
#FEATURE SCALING 
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
x_test=sc.transform(X_test) #Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confus7ion_matrix(Y_test,Y_pred)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)*100