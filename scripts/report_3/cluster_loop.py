import numpy as np
from tqdm import tqdm

def custom_distance(u, v, all_fields, mode='mean'):
    dists = []
    mode_to_func = {'mean': np.mean, 'median': np.median, 'sum': np.sum}
    func = mode_to_func[mode]

    for field_id in all_fields:
        points_u = np.array(u.get(field_id, []))
        points_v = np.array(v.get(field_id, []))
        if len(points_u) > 0 and len(points_v) > 0:
            stat_u = func([list(point) for point in points_u], axis=0)
            stat_v = func([list(point) for point in points_v], axis=0)
            dists.append(np.sqrt(np.sum((stat_u - stat_v)**2)))
        else:
            len_fields = max(len(points_u), len(points_v), 1)
            dists.append(0.6 * len_fields)
    return np.sum(dists)

def get_dist_matrix(df, dt):
  all_fields = set()
  for fe in df.field_embeddings:
    all_fields.update(fe.keys())

  n = len(df)
  dist_matrix = np.zeros((n, n))
  desc = f'Calculating dist_matrix for {dt}"'
  for i in tqdm(range(n), desc=desc):
      for j in range(i+1, n):
          dist_matrix[i, j] = custom_distance(df['field_embeddings'].iloc[i], df['field_embeddings'].iloc[j], all_fields, mode='sum')
          dist_matrix[j, i] = dist_matrix[i, j]

  return dist_matrix

def get_acc(df, dist_matrix, clusters, mode=None):

  df['cluster'] = clusters

  if mode == 'adj':
    df = df[df['cluster']!=-1].copy().reset_index(drop=True)
    clusters = np.array([i for i in clusters if i != -1])
    dist_matrix = None

  n_clusters = len(set(clusters))
  metrics, df = calculate_clustering_metrics(dist_matrix, n_clusters, df, clusters)

  return metrics.round(3), df

def get_average_dist(df, dist_matrix):
  n = len(df)
  total_sum_of_distances = np.sum(dist_matrix)
  total_number_of_distances = n * n - n
  average_distance = total_sum_of_distances / total_number_of_distances
  start = max(np.floor(average_distance-1), 0.1)
  end = max(np.ceil(average_distance+1), 1)
  return start, end