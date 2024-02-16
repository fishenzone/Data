def evaluate_search(query_embedding, true_index, index, top_k=5):
    """Search for the query embedding and evaluate if the true index is in top results.
    
    Args:
        query_embedding (np.array): The embedding of the query paragraph.
        true_index (int): The true index of the paragraph in the all_paragraphs list.
        index (faiss.Index): The FAISS index containing all paragraph embeddings.
        top_k (int): Number of top results to consider for evaluation.
    
    Returns:
        top_1_correct (bool): True if the true index is the top 1 result.
        top_5_correct (bool): True if the true index is within the top 5 results.
    """
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    top_1_correct = true_index == I[0][0]
    top_5_correct = true_index in I[0][:5]
    return top_1_correct, top_5_correct
