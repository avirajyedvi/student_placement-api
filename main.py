import numpy as np
import pandas as pd
import os

cwd=os.getcwd()

print("Current WD", cwd)

df=pd.read_csv('students_placement.csv')

df.head

df.shape

df.columns

df.sample(10)

X = df.iloc[:,:3]
y = df.iloc[:,-1]

X
y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf  = scaler.transform(X_test)
accuracy_score(Y_test,
               LogisticRegression()
               .fit(X_train_trf,Y_train)
                .predict(X_test_trf))

print("ACCURACY SCORE of LogisticRegression Model ==" , accuracy_score)

from sklearn.ensemble import RandomForestClassifier
accuracy_score(Y_test,RandomForestClassifier().fit(X_train,Y_train).predict(X_test))

print("ACCURACY SCORE of RandomForestClassifier Model ==" , accuracy_score)

from sklearn.svm import SVC
accuracy_score(Y_test,SVC(kernel='rbf').fit(X_train,Y_train).predict(X_test))

print("ACCURACY SCORE of SVM Model ==" , accuracy_score)

svc = SVC(kernel='rbf')
svc.fit(X_train,Y_train)

Y_predict = svc.predict(X_test)

accuracy_score_svc= accuracy_score(Y_test,Y_predict)
print("Accuracy Score for SVM model is" , round(accuracy_score_svc*100,2),'%')

##Preparing Pickle File for creating  model.pkl file
import pickle
pickle.dump(svc,open('model.pkl','wb'))




