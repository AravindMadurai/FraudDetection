import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#IMPORT DATA
TRANSACTION_DATASET = pd.read_csv("") 
#ANALYZE DATA
sns.countplot(x="FRAUD", data=TRANSACTION_DATASET) # aggregation report on fraud data
TRANSACTION_DATASET.isnull().sum() # Null Check in dataset.

#TRAIN DATA
X-TRANSACTION_DATASET.drop("FRAUD", axis-1) #dropping the o/p column from the i/p dataset for prediction
X-X.drop("TIMESTAMP", axis=1)
X-X.drop("MAOIST_CITY",axis=1)
X-X.drop("WHITELISTED IP", axis=1)
X=X.drop("CARD_NUMBER",axis=1)
X=X.drop("HOTLISTED_CARD",axis=1)
X=X.drop("HOUR_H",axis=1) 
X=X.drop("MINUTES M",axis-1)
X=X.drop("SECONDS_S",axis=1)
X=X.drop("MCC",axis-1) #ACCURACY IS REDUCING IF YOU REMOVE 
X=X.drop("TRAN_CODE",axis=1)
Y=TRANSACTION_DATASET.FRAUD

#Split Dataset => train and test data 
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test size=0.3, random_state=1)

#LOGISTIC REGRESSION
from sklearn.linear model 
import LogisticRegression 
model=LogisticRegression()

#X_train=X_train.drop("TIMESTAMP", axis=1)
#element = datetime.datetime.strptime (X_train. TIMESTAMP, "%d/%m/%Y") #X_train.TIMESTAMP= datetime.datetime.timestamp (element)
model.fit(X_train,Y_train)

#X_test=X_test.drop("TIMESTAMP", axis-1)
#PREDICTION
predictions = model.predict(X_test)

#CONFUSTION MATRIX ON THE PREDICTION
from sklearn.metrics import confusion_matrix 
confusion_matrix(Y_test, predictions)

#ACCURACY
from sklearn.metrics import accuracy_score 
accuracy_score(Y_test, predictions)*100