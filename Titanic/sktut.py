import pandas as pd
import numpy as np
X = pd.read_csv('titanic_data.csv')
X = X.select_dtypes(include=[object])
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
ohe = OneHotEncoder()
for feature in X:
	X[feature] = le.fit_transform(X[feature])
	print X[feature]
xt = ohe.fit_transform(X)
print xt
