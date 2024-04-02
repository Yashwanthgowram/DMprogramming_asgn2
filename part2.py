from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering,KMeans
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

model_sse_inertial={}
model_sse_manual={}

def fit_kmeans(data, max_clusters):
    """
    Fit KMeans and calculate SSE for each k, keeping the structure and functionality.
    
    Parameters:
    - data: Dataset.
    - max_clusters: Maximum number of clusters to try.
    
    Returns:
    - Tuple of two lists containing SSE values calculated using the inertia attribute and manually.
    """
    inertia_sse_list = [0] * max_clusters  # Initialize with appropriate length
    manual_sse_list = [0] * max_clusters   # Initialize with appropriate length
    for cluster_count in range(1, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=cluster_count, n_init=10)
        cluster_labels = kmeans_model.fit_predict(data)
        
        cluster_centers = kmeans_model.cluster_centers_
        total_manual_sse = 0
        
        # Improved manual SSE calculation
        for center_idx, center in enumerate(cluster_centers):
            points_in_cluster = data[cluster_labels == center_idx]
            total_manual_sse += np.sum((points_in_cluster - center) ** 2)
        
        inertia_sse_list[cluster_count - 1] = kmeans_model.inertia_
        manual_sse_list[cluster_count - 1] = total_manual_sse
    
    return inertia_sse_list, manual_sse_list




def compute():
    answers = {}

    # Parameters for make_blobs
    n_samples=20
    center_box=(-20,20)
    centers=5
    random_state=12
    x,label = datasets.make_blobs(n_samples=n_samples, centers=centers, center_box=center_box, random_state=random_state)

    list_1=x[0:,0:1]
    list_2=x[0:,1:] 

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [list_1, list_2, label]

    # dct value: the `fit_kmeans` function
    X_data = np.concatenate([answers["2A: blob"][0], answers['2A: blob'][1]], axis=1)
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_values = fit_kmeans(X_data, 8)[1]
    k_values = range(1, 9)

    plt.plot(k_values, sse_values, marker='o')
    plt.title('SSE vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.savefig("part2C.png")

    result_dict = answers["2C: SSE plot"] = sse_vs_k_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    sse_values_d = fit_kmeans(X_data, 8)[0]  # Pass 8 as the maximum number of clusters
    sse_vs_k_values_d = [[x, y] for x, y in zip(range(1, 9), sse_values_d)]

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = sse_vs_k_values_d
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "no"

    return answers



# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
