# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#%%
def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df=df.drop(columns=['Channel', 'Region'])
    return df
#%%
# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    statistics = df.agg(['mean', 'std', 'min', 'max']).round().astype(int)
    statistics = statistics.transpose()
    statistics.index.name = 'Attribute'
    statistics_df = pd.DataFrame(statistics)
    return statistics_df
#%%
# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    means = df.mean()
    stds = df.std()
    standardize_df = (df - means) / stds
    standardize_df = pd.DataFrame(standardize_df)
    return standardize_df
#%%
# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    kmeans_model = KMeans(n_clusters=k, init='random', n_init=1, random_state=42)
    kmeans_model.fit(df)
    y = kmeans_model.labels_
    return pd.Series(y)

#%%
# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    kmeans_plus_model = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42)
    kmeans_plus_model.fit(df)
    y = kmeans_plus_model.labels_
    return pd.Series(y)
#%%
# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    agg_model = AgglomerativeClustering(n_clusters=k)
    y = agg_model.fit_predict(df)
    return pd.Series(y)
#%%
# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
    score = silhouette_score(X, y)
    return score
#%%

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    results_df = pd.DataFrame(columns=['Algorithm', 'data type', 'k', 'Silhouette Score'])
    algorithms = ['Kmeans', 'Agglomerative']
    data_types = ['Original', 'Standardized']
    k_values = [3, 5, 10]  
    for algorithm in algorithms:
        for k in k_values:
            if algorithm == 'Kmeans':
                y = kmeans(df, k)
            elif algorithm == 'Agglomerative':
                y = agglomerative(df, k)
            score = clustering_score(df, y)
            results_df = pd.concat([results_df, pd.DataFrame({'Algorithm': algorithm, 'data type': 'Original', 'k': k, 'Silhouette Score': score}, index=[0])], ignore_index=True)

    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    for algorithm in algorithms:
        for k in k_values:
            if algorithm == 'Kmeans':
                y = kmeans(df_standardized, k)
            elif algorithm == 'Agglomerative':
                y = agglomerative(df_standardized, k)
            score = clustering_score(df_standardized, y)
            results_df = pd.concat([results_df, pd.DataFrame({'Algorithm': algorithm, 'data type': 'Standardized', 'k': k, 'Silhouette Score': score}, index=[0])], ignore_index=True)
    return results_df
#%%
# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    best_score = rdf['Silhouette Score'].max()
    return best_score
#%%
# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(df)
    labels = kmeans_model.labels_
    n_attributes = df.shape[1]
    attribute_names = df.columns
    fig, axes = plt.subplots(n_attributes, n_attributes, figsize=(15, 15))
    for i in range(n_attributes):
        for j in range(n_attributes):
            if i != j:
                for label in range(3): 
                    axes[i, j].scatter(df[labels == label][attribute_names[i]], df[labels == label][attribute_names[j]], label=f'Cluster {label}')
                axes[i, j].set_xlabel(attribute_names[i])
                axes[i, j].set_ylabel(attribute_names[j])
                axes[i, j].legend()
            else:
                axes[i, j].set_axis_off()  
    plt.tight_layout()
    plt.show()