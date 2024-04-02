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
import myplots as myplt
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, labels, num_clusters, linkage_type='ward'):
    scaler = StandardScaler()
    cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_type)
    scaled_data = scaler.fit_transform(data)
    cluster_model.fit(scaled_data, labels)
    predictions = cluster_model.labels_
    return predictions

def fit_modified(data, labels, linkage_type):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    Z = linkage(scaled_data, linkage_type)
    distances = [next_dist - curr_dist for curr_dist, next_dist in zip(Z[:-1, 2], Z[1:, 2])]
    distance_threshold = Z[np.argmax(distances), 2]
    cluster_model = AgglomerativeClustering(n_clusters=None, linkage=linkage_type, distance_threshold=distance_threshold)
    cluster_model.fit(scaled_data, labels)
    updated_labels = cluster_model.labels_
    return updated_labels
   
    


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    total_samples = 100
    chosen_state = 42
    dct = answers["4A: datasets"] = {}
    noisy_circles = datasets.make_circles(n_samples=total_samples, factor=0.5, noise=0.05, random_state=chosen_state)
    noisy_moons = datasets.make_moons(n_samples=total_samples, noise=0.05, random_state=chosen_state)
    blobs= datasets.make_blobs(n_samples=total_samples, random_state=chosen_state)
    Blobs_with_varied_variances = datasets.make_blobs(n_samples=total_samples, cluster_std=[1.0, 2.5, 0.5], random_state=chosen_state)

    data_blob, labels_blob = datasets.make_blobs(n_samples=total_samples, random_state=chosen_state)
    transformation_matrix = [[0.6, -0.6], [-0.4, 0.8]]
    data_aniso_transformed = np.dot(data_blob, transformation_matrix)
    aniso_dataset = (data_aniso_transformed, labels_blob)
    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    dct["nc"] = [noisy_circles[0],noisy_circles[1]]
    dct["nm"] = [noisy_moons[0],noisy_moons[1]]
    dct["bvv"] = [Blobs_with_varied_variances[0],Blobs_with_varied_variances[1]]
    dct["add"] = [aniso_dataset[0],aniso_dataset[1]]
    dct["b"] = [blobs[0],blobs[1]]

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    def fit_hierarchical_cluster_B(data, labels, linkage_type, num_clusters):
        scaler = StandardScaler()
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_type)
        scaled_data = scaler.fit_transform(data)
        cluster_model.fit(scaled_data, labels)
        predictions = cluster_model.labels_
        return predictions
    cluster_results = {}
    for dataset_name, dataset_info in answers["4A: datasets"].items():
        dataset_clusters = {}
        dataset_data = dataset_info[0]
        dataset_labels = dataset_info[1]
        results_list = []

        for linkage_type in ['single', 'complete', 'ward', 'average']:
            predictions = fit_hierarchical_cluster_B(dataset_data, dataset_labels, linkage_type, 2)
            dataset_clusters[linkage_type] = predictions

        results_list.append((dataset_data, dataset_labels))
        results_list.append(dataset_clusters)
        cluster_results[dataset_name] = results_list

    myplt.plot_part1C(cluster_results, 'Part4B.jpg')

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified
    cluster_results = {}
    for dataset_name, dataset_info in answers["4A: datasets"].items():
        dataset_clusters = {}
        dataset_data = dataset_info[0]
        dataset_labels = dataset_info[1]
        results_list = []

        for linkage_type in ['single', 'complete', 'ward', 'average']:
            predictions = fit_modified(dataset_data, dataset_labels, linkage_type)
            dataset_clusters[linkage_type] = predictions

        results_list.append((dataset_data, dataset_labels))
        results_list.append(dataset_clusters)
        cluster_results[dataset_name] = results_list

    myplt.plot_part1C(cluster_results, 'Part4C.jpg')

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
