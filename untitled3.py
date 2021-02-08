# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 05:16:36 2021

@author: FnH
"""


# Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('order_preparation.csv')
#analysis is done on revenue
good_data=[]
x = dataset.iloc[:, -1:].values.flatten()
for i in range(17963):
    if x[i] < 1180 and x[i]>50:
        good_data.append(x[i])

from abc_analysis import abc_analysis, abc_plot

# Perform an ABC analysis with plotting
dctAnalysis = abc_analysis(good_data, True)
A_i=dctAnalysis['Aind']
B_i=dctAnalysis['Bind']
C_i=dctAnalysis['Cind']
#A =[a for a in good_data[A_i] ]
A=[]
B=[]
C=[]
for i in range(len(A_i)):
    A.append(good_data[A_i[i]])
    
for i in range(len(B_i)):
    B.append(good_data[B_i[i]]) 
    
for i in range(len(C_i)):
    C.append(good_data[C_i[i]])
# Plot saved results of an ABC analysis
abc_plot(dctAnalysis)





import seaborn as sns
sns.distplot(good_data)

#Create bins function
def bins(a):
    for bar in range(20, 1300, 100):
        if a <= bar:
            return bar
# Create new column to apply the bin function
            
#dataset["rev_dist"] = pd.DataFrame(good_data, columns=['good_data']).apply(lambda a: bins(a))

# Create a support column of 1’s to facilitate the pivot table
#dataset["count"] = 1
# Create a pivot table of the revenue distributions
#pivot_table
#X= pd.pivot_table(dataset, index = ["rev_dist"], values = ["count"], aggfunc = np.sum)

#from sklearn.cluster import KMeans'''

#''''kmeans = KMeans(n_clusters=3)'''
#''''kmeans.fit(pivot_table)'''
'''
# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
'''