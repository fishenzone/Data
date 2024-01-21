import os
import re
import glob

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

from umap import UMAP

# cluster_to_doctype = df.groupby('cluster')['doctype'].agg(lambda x: x.value_counts().index[0]).to_dict()

def split_text_into_tokens(df):
    df['text_tokenized'] = df['text_processed'].apply(lambda x: x.split())
    return df

def generate_word2vec_model(df):
    model = Word2Vec(sentences=df['text_tokenized'], vector_size=250, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    return word_vectors

def calculate_document_vector(word2vec_model, doc):
    doc = [word for word in doc if word in word2vec_model.key_to_index]
    return np.mean(word2vec_model[doc], axis=0)

def generate_document_vectors_for_dataframe(df, word_vectors):
    df['doc_vector'] = df['text_tokenized'].apply(lambda x: calculate_document_vector(word_vectors, x))
    X = np.vstack(df['doc_vector'].values)
    return X