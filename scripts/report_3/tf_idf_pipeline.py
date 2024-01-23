import os
import re
import glob
from Typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

# from umap import UMAP

# cluster_to_doctype = df.groupby('cluster')['doctype'].agg(lambda x: x.value_counts().index[0]).to_dict()

def calculate_accuracy(df: pd.DataFrame, labels: np.array) -> Tuple[float, pd.DataFrame]:   
    df['cluster'] = labels
    clusters = df['cluster'].value_counts().index.tolist()

    used_doctypes = set()
    cluster_to_doctype = {}


    for cluster in clusters:
        doctypes = df[df['cluster'] == cluster]['doctype'].value_counts().index.tolist()
        for doctype in doctypes:
            if doctype not in used_doctypes:
                cluster_to_doctype[cluster] = doctype
                used_doctypes.add(doctype)
                break

    remaining_clusters = [cluster for cluster in clusters if cluster not in cluster_to_doctype]
    remaining_doctypes = df[~df['doctype'].isin(used_doctypes)]['doctype'].value_counts().index.tolist()

    for cluster, doctype in zip(remaining_clusters, remaining_doctypes):
        cluster_to_doctype[cluster] = doctype

    df['predicted_doctype'] = df['cluster'].map(cluster_to_doctype)
    df.loc[df.predicted_doctype.isna(), 'predicted_doctype'] = doctype
    accuracy = accuracy_score(df['doctype'], df['predicted_doctype'])

    return accuracy, df

def load_dataframe_from_feather(file_path):
    df = pd.read_feather(file_path)
    return df

def remove_stopwords_and_punctuations(df):
    stop_words = set(stopwords.words('russian')).union(stopwords.words('english'))
    df['text_processed'] = df['text'].apply(lambda x: re.sub('[^\w\s]', '', x.lower()))
    return df, list(stop_words)

def apply_tfidf_vectorization(df, stop_words):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(df['text_processed'])
    return X

def perform_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels

def calculate_clustering_metrics(X, n_clusters, df, labels):

    sil_score = silhouette_score(X, labels)
    ari = adjusted_rand_score(df['doctype'], labels)
    ami_score = adjusted_mutual_info_score(df['doctype'], labels)

    is_equal = n_clusters == df['doctype'].nunique()
    accuracy, df = calculate_accuracy(df, labels)

    results = pd.DataFrame({
        'N clusters': [n_clusters],
        'SS': [sil_score],
        'ARI': [ari],
        'AMI': [ami_score],
        'Equal': [is_equal],
        'Acc': [accuracy]
    })
    return results, df

def iterate_over_cluster_sizes(df, X):
    all_results = pd.DataFrame()

    for n_clusters in range(2, df['doctype'].nunique() + 4):
        labels = perform_kmeans_clustering(X, n_clusters)
        results, df = calculate_clustering_metrics(X, n_clusters, df, labels)
        all_results = pd.concat([all_results, results], ignore_index=True)

    all_results[['SS', 'ARI', 'AMI', 'Acc']] = all_results[['SS', 'ARI', 'AMI', 'Acc']].round(3)
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
    # elif method == 'word2vec':
    #     df = split_text_into_tokens(df)
    #     word_vectors = generate_word2vec_model(df)
    #     X = generate_document_vectors_for_dataframe(df, word_vectors)
    #     all_results, df = iterate_over_cluster_sizes(df, X)
    #     return all_results, df
    else:
        print("Not yet.")