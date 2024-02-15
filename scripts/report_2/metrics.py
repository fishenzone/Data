from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import editdistance

def calculate_metrics(pred_labels, true_labels):
    # Ensure the lists are of the same length
    assert len(pred_labels) == len(true_labels), "The length of predictions and true labels must match."
    
    # Flatten the lists if they contain sequences (e.g., lists of characters)
    flattened_pred = [char for seq in pred_labels for char in seq]
    flattened_true = [char for seq in true_labels for char in seq]
    
    # Calculate character-level accuracy
    char_accuracy = np.mean([p == t for p, t in zip(flattened_pred, flattened_true)])
    
    # Calculate average Levenshtein distance (edit distance)
    avg_edit_distance = np.mean([editdistance.eval(p, t) for p, t in zip(pred_labels, true_labels)])
    
    # Assuming binary or multiclass classification, convert sequences to string labels if necessary
    # Note: You may need to adjust this depending on how your labels are structured
    pred_labels_str = [''.join(seq) for seq in pred_labels]
    true_labels_str = [''.join(seq) for seq in true_labels]
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels_str, pred_labels_str, average='macro')
    recall = recall_score(true_labels_str, pred_labels_str, average='macro')
    f1 = f1_score(true_labels_str, pred_labels_str, average='macro')
    
    return {
        'Character-Level Accuracy': char_accuracy,
        'Average Edit Distance': avg_edit_distance,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Example usage:
pred_labels = [...]  # Your model predictions
true_labels = [...]  # Ground truth labels
metrics = calculate_metrics(pred_labels, true_labels)
print(metrics)


import pandas as pd


df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                   'B': [6, 7, 8, 9, 10],
                   'C': [11, 12, 13, 14, 15]})

max_row = df.max()
median_row = df.median()
mean_row = df.mean()

df.loc['min'] = df.min(axis=0)
df.loc['max'] = df.max(axis=0)
df.loc['median'] = df.median(axis=0)
df.loc['mean'] = df.mean(axis=0)

# Filter columns where max value is 0
filtered_df = df.loc[:, df.max() == 0]

# Print the filtered DataFrame
print(filtered_df)

