import numpy as np
import pandas as pd

import hdbscan
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering, MeanShift, AffinityPropagation, OPTICS, Birch

from .tf_idf_pipeline import calculate_clustering_metrics

def perform_all_clustering(X, n_clusters, algorithm):
    clustering_algorithms = {
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=0.3),
        'Spectral': SpectralClustering(n_clusters=n_clusters),
        'MeanShift': MeanShift(bandwidth=2),
        'AffinityPropagation': AffinityPropagation(),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=10),
        'OPTICS': OPTICS(min_samples=10),
        'BIRCH': Birch(n_clusters=n_clusters),
        'KMeans': KMeans(n_clusters=n_clusters, n_init=10)
    }

    model = clustering_algorithms[algorithm]
    try:
        # labels = model.fit_predict(X)
        labels = model.fit_predict(X.toarray())
    except KeyError:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    # except:
        
    return labels

def calculate_all_metrics(X, n_clusters, ds, all_labels):
    df = ds.copy()
    all_metrics = []
    for algorithm, labels in all_labels.items():
        try:
            metrics, df = calculate_clustering_metrics(X, n_clusters, df, labels)
            metrics['algorithm'] = algorithm
        except Exception as e:
            print(f"An error occurred with {algorithm}: {str(e)}")
            metrics = pd.DataFrame({
                'algorithm': [algorithm],
                'N clusters': [0],
                'SS': [0],
                'ARI': [0],
                'AMI': [0],
                'Equal': [False],
                'Acc': [0],
                'LA': [0],
            })
        all_metrics.append(metrics)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True).round(2)
    return all_metrics_df, df
