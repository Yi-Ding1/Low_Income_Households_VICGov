"""
This program computes the optimal k value using the elbow method
and performs k-means clustering along with features generated
using PCA. Return the plot of the K-means clustering.
Author: Qinglin Huang
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

'''remove the columns that can not be used for the model'''
data = pd.read_csv('communities_modified.csv')
data = data.drop(columns=['Community Name','ABS remoteness category'])


''' defines the target feature and the features we want to compare it with'''
X_COLS = ['Requires assistance with core activities, %', 
          'Did not complete year 12, %', 
          'Holds degree or higher, %','ARIA+ (avg)',
          '2012 ERP age 70+, %']
y_COL = ['Equivalent household income <$600/week, %']


'''apply PCA to the comparing features to reduce demonsion of the data'''
sklearn_pca = PCA(n_components=1)
X_pca = sklearn_pca.fit_transform(data.drop(y_COL, axis=1))


'''prints the variance explained the PCA'''
explained_var = sklearn_pca.explained_variance_ratio_
print(f"PCA variance explained: {explained_var.sum()}")



'''normalise the data'''
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_pca)
Y_scaled = scaler.fit_transform(data[y_COL])
X_scaled_df = pd.DataFrame(X_scaled, columns=['PCA'])
Y_scaled_df = pd.DataFrame(Y_scaled, columns=['Equivalent household income <$600/week, %'])
data_s_pca = pd.concat([X_scaled_df, Y_scaled_df], axis=1)


'''calculate the sse for every value of k, and draw the elbow method graph based on it'''
distortions = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_) 
    
plt.plot(k_range, distortions, 'bx-')
plt.title('The Elbow Method showing the optimal k (PCA)')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.show()


'''the k was chosen to be 3 based on the graph'''
'''apply clustering to the normalized data'''
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit(data_s_pca)


'''plot the graph of kmean cluster'''
colormap = {0: 'red', 1: 'green', 2: 'blue'}
    
fig = plt.figure(figsize=(7, 10))
plt.scatter(data_s_pca['PCA'], 
            data_s_pca['Equivalent household income <$600/week, %'],
            c=[colormap.get(x) for x in clusters.labels_],
            s=10)
    

plt.title("principal component versus household income <$600/week when "f"k = {len(set(clusters.labels_))}")
plt.xlabel("Principal Component")
plt.ylabel("Equivalent household income <$600/week")
plt.show()






