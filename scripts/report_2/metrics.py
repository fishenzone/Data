from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import editdistance

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
