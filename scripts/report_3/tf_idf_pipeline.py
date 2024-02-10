import os
import re
import glob
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, rand_score

# cluster_to_doctype = df.groupby('cluster')['doctype'].agg(lambda x: x.value_counts().index[0]).to_dict()

def calculate_accuracy(df: pd.DataFrame, labels: np.array, col: str) -> Tuple[float, float, float, pd.DataFrame]:
    df['cluster'] = labels
    cluster_to_col = df.groupby('cluster')[col].agg(lambda x: x.value_counts().idxmax()).to_dict()
    
    pred_col = f'predicted_{col}'
    df[pred_col] = df['cluster'].map(cluster_to_col)

    accuracy = accuracy_score(df[col], df[pred_col])

    # Mapping each unique predicted_doctype to a unique cluster
    doctype_to_adjusted_cluster = {doctype: cluster for cluster, doctype in enumerate(df[pred_col].unique())}
    df['adjusted_cluster'] = df[pred_col].map(doctype_to_adjusted_cluster)

    ari = adjusted_rand_score(df[col], df['adjusted_cluster'])
    ami = adjusted_mutual_info_score(df[col], df['adjusted_cluster'])
    rs = rand_score(df[col], df['adjusted_cluster'])
    
    return accuracy, ari, ami, rs, df

def load_dataframe_from_feather(file_path):
    df = pd.read_feather(file_path)
    return df

def remove_stopwords_and_punctuations(df):
    stop_words = set(stopwords.words('russian')).union(stopwords.words('english'))
    df['text_processed'] = df['text'].apply(lambda x: re.sub('[^\w\s]', '', x.lower()))
    return df, list(stop_words)

def apply_tfidf_vectorization(df, stop_words, min_df=5, max_df=0.5, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df['text_processed'])
    return X

def perform_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels

def calculate_clustering_metrics(X, n_clusters, df, labels, col):

    sil_score = silhouette_score(X, labels)
    rs = rand_score(df[col], labels)
    ami_score = adjusted_mutual_info_score(df[col], labels)

    is_equal = n_clusters == df[col].nunique()
    accuracy, ari, ami_score, rs, df = calculate_accuracy(df, labels, col)

    results = pd.DataFrame({
        'N clusters': [n_clusters],
        'SS': [sil_score],
        'ARI': [ari],
        'RS': [rs],
        'AMI': [ami_score],
        'Equal': [is_equal],
        'Acc': [accuracy],
    })
    return results, df

def iterate_over_cluster_sizes(df, X):
    all_results = pd.DataFrame()

    for n_clusters in range(2, df['doctype'].nunique() + 4):
        labels = perform_kmeans_clustering(X, n_clusters)
        results, df = calculate_clustering_metrics(X, n_clusters, df, labels)
        all_results = pd.concat([all_results, results], ignore_index=True)

    all_results[['SS', 'ARI', 'AMI', 'Acc', 'LA']] = all_results[['SS', 'ARI', 'AMI', 'Acc', 'LA']].round(3)
    return all_results, df

def visualize_clusters_with_tsne(df, X, labels=None, calculate=False):
    if calculate:
        n_clusters = int(df.doctype.nunique())
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(X)

    df['cluster'] = labels
    clusters = df['cluster'].value_counts().index.tolist()
    doctypes = df['doctype'].value_counts().index.tolist()

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X.toarray())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    colors = 'cmykrgb'
    for i, cluster in enumerate(clusters):
        # Plot predicted clusters
        axs[0].scatter(X_2d[df['cluster'] == cluster, 0], X_2d[df['cluster'] == cluster, 1], c=colors[i % len(colors)],
                        label=f'Cluster {cluster} (n={np.sum(df["cluster"] == cluster)})')
        # Plot actual doctypes
        doctype = doctypes[i]
        axs[1].scatter(X_2d[df['doctype'] == doctype, 0], X_2d[df['doctype'] == doctype, 1],
                        c=colors[i % len(colors)], label=
                        f'{doctype[:min(len(doctype), 20)]} (n={df[df["doctype"] == doctype].shape[0]})')

    axs[0].legend(bbox_to_anchor=(0, -0.05), loc='upper left')
    axs[0].set_title(f'Predicted Clusters (T-SNE plot)')
    axs[1].legend(bbox_to_anchor=(0, -0.05), loc='upper left')
    axs[1].set_title('Actual Doctypes')

    plt.subplots_adjust(right=0.85)
    plt.show()

def perform_text_processing_and_clustering(df, method):
    df, stop_words = remove_stopwords_and_punctuations(df)

    if method == 'tfidf':
        X = apply_tfidf_vectorization(df, stop_words)
        all_results, df = iterate_over_cluster_sizes(df, X)
        # equal_results = all_results[all_results['Equal'] == True]
        visualize_clusters_with_tsne(df, X, calculate=True)
        return all_results, df
    elif method == 'word2vec':
        df = split_text_into_tokens(df)
        word_vectors = generate_word2vec_model(df)
        X = generate_document_vectors_for_dataframe(df, word_vectors)
        all_results, df = iterate_over_cluster_sizes(df, X)
        return all_results, df
    else:
        print("Not yet.")