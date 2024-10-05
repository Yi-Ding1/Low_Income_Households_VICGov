import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


'''remove the columns that can not be used for the model'''
data = pd.read_csv('communities_modified.csv')
data = data.drop(columns=['Community Name','ABS remoteness category'])


''' defines the target feature and the features we want to compare it with'''
X_COLS = ['Requires assistance with core activities, %', 
          'Did not complete year 12, %', 
          'Holds degree or higher, %','ARIA+ (avg)',
          '2012 ERP age 70+, %']
y_COL = ['Equivalent household income <$600/week, %']


'''normalise the data'''
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data.drop(y_COL, axis=1))
Y_scaled = scaler.fit_transform(data[y_COL])


'''calculate the sse for every value of k, and draw the elbow method graph based on it'''
distortions = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_) 
    
plt.plot(k_range, distortions, 'bx-')
plt.title('The Elbow Method showing the optimal k')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.show()


'''the k was chosen to be 3 based on the graph'''
'''apply clustering based on all the features'''
'''assign every data point to the corresponding cluster'''
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters


'''plot a graph for every feature with the target feature with colours representing the clusters'''
for i in X_COLS:
    d1 = data[data['Cluster']==0]
    d2 = data[data['Cluster']==1]
    d3 = data[data['Cluster']==2]
    plt.scatter(d1[i], d1['Equivalent household income <$600/week, %'], color = 'blue', s=10)
    plt.scatter(d2[i], d2['Equivalent household income <$600/week, %'], color = 'green', s=10)
    plt.scatter(d3[i], d3['Equivalent household income <$600/week, %'], color = 'red', s=10)
    plt.xlabel(i)
    plt.ylabel('Equivalent household income <$600/week, %')
    i_title = i.replace(', %', '')
    plt.title(i_title + " versus income <$600/week")
    plt.show()
