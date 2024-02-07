import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.cluster import HDBSCAN, AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering, MeanShift, AffinityPropagation, OPTICS, Birch
from umap import UMAP

from .tf_idf_pipeline import calculate_clustering_metrics

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

def get_df_from_path(path):
  df = pd.read_feather(path)
  df['field_embeddings'] = df['field_embeddings'].apply(lambda d: {k: v for k, v in d.items() if v is not None})
  df = df[df['field_embeddings'].apply(lambda x: bool(x))]
  df = df.groupby('doctype').filter(lambda x: len(x) > 20).reset_index(drop=True)
  df.reset_index(drop=True, inplace=True)

  print(f"df.shape: {df.shape}\ndoctypes: {df.doctype.unique().tolist()}\nproject: {path.split('/')[-1]}")
  return df

def kmeans_loop(reduced_data, field_cluster, df_dt, col):
    kmeans = KMeans(n_clusters=field_cluster)
    clusters = kmeans.fit_predict(reduced_data)
    metrics, ds = calculate_clustering_metrics(reduced_data, field_cluster, df_dt, clusters, col)
    return metrics, ds

def get_nclusters_from_df_dt(df, dt):
    df_dt = df[df.doctype==dt].copy().reset_index(drop=True)
    dist_matrix = get_dist_matrix(df_dt, dt)
    all_metrics = pd.DataFrame()

    n_components = 2

    umap = UMAP(metric="precomputed", n_components=n_components)
    reduced_data = umap.fit_transform(dist_matrix)

    for field_cluster in range(2, 9):

      metrics, ds = kmeans_loop(reduced_data, field_cluster, df_dt, 'doctype')
      all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

    # all_metrics = all_metrics.sort_values(by="SS", ascending=False)
    # return all_metrics.iloc[0]['N clusters']

    idx = all_metrics['SS'].idxmax()
    n_clusters = all_metrics.loc[idx, 'N clusters']
    metrics, ds = kmeans_loop(reduced_data, n_clusters, df_dt, 'doctype')

    return metrics, ds, n_clusters
