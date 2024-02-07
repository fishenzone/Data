import numpy as np
from tqdm import tqdm
from .tf_idf_pipeline import calculate_clustering_metrics

def get_acc(df, dist_matrix, clusters, col, mode=None):

  df['cluster'] = clusters

  if mode == 'adj':
    df = df[df['cluster']!=-1].copy().reset_index(drop=True)
    clusters = np.array([i for i in clusters if i != -1])
    dist_matrix = None

  n_clusters = len(set(clusters))
  metrics, df = calculate_clustering_metrics(dist_matrix, n_clusters, df, clusters, col)

  return metrics.round(3), df

def get_average_dist(df, dist_matrix):
  n = len(df)
  total_sum_of_distances = np.sum(dist_matrix)
  total_number_of_distances = n * n - n
  average_distance = total_sum_of_distances / total_number_of_distances
  start = max(np.floor(average_distance-1), 0.1)
  end = max(np.ceil(average_distance+1), 1)
  return start, end
