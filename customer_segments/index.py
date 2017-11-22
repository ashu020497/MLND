import numpy as np
import pandas as pd
from IPython.display import display 

import visuals as vs


#%matplotlib inline

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
indices = [120,38,360]

samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)
new_data = data.drop('Grocery',axis=1)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_data, data['Grocery'], test_size=.25, random_state=42)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print score
log_data = np.log(data)

log_samples = np.log(samples)

pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
outliers  = []
for feature in log_data.keys():
    
  
    Q1 = np.percentile(log_data[feature], 25)
    

    Q3 = np.percentile(log_data[feature], 75)
    

    step = 1.5 * (Q3 - Q1)

    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])


    outliers += log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist()
print outliers

good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)

pca_samples = pca.transform(log_samples)

pca_results = vs.pca_results(good_data, pca)
pca = PCA(n_components=2)
pca.fit(good_data)


reduced_data = pca.transform(good_data)

pca_samples = pca.transform(log_samples)

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
vs.biplot(good_data, reduced_data, pca)
for n in range(2, 5):
    from sklearn.mixture import GMM
    
    clusterer = GMM(n_components=n, random_state=11)

    clusterer.fit(reduced_data)

    preds = clusterer.predict(reduced_data)

    centers = clusterer.means_

    sample_preds = clusterer.predict(pca_samples)

    from sklearn.metrics import silhouette_score
    score = silhouette_score(reduced_data, preds, random_state=42)
    print('silhouette score for {} clusters is {}'.format(n,score)
    
clusterer = GMM(n_components=n, random_state=11)

clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.means_
sample_preds = clusterer.predict(pca_samples)
vs.cluster_results(reduced_data, preds, centers, pca_samples)
log_centers = pca.inverse_transform(centers)

true_centers = np.exp(log_centers)

segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
display(np.exp(good_data).describe())
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred