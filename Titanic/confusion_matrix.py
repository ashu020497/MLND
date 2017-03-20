import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
X = pd.read_csv('titanic_data.csv')
X=X._get_numeric_data()
y = X['Survived']
del X['Age'],X['Survived']
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
print "The confusion Matrix is ",confusion_matrix(y_test,clf1.predict(X_test))
print "The confusion Matrix is ",confusion_matrix(y_test,clf2.predict(X_test))
