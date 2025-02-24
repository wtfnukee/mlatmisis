import numpy as np
import pandas as pd


data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(data, index=labels)
# # print(df) 


# def mean_table(df):
#     ndf=df.groupby(['animal', 'visits'])['age'].mean().unstack()
#     return ndf

# print(mean_table(df))
# data = {
#         "grps": ["a", "a", "a", "a", "b", "b", "b", "b"],
#         "vals": [100, 200, 109, 50, 23, 100, 33, 67],
#     }
# df = pd.DataFrame(data)
# print(df)
# print()
# ndf = df.groupby('grps')['vals'].apply(lambda x: x.nlargest(3).sum())
# print(ndf)

import matplotlib.pyplot as plt

df_clean = df.dropna(subset=['age'])

plt.figure(figsize=(10, 6))
plt.plot(df_clean.index, df_clean['age'], color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Возраст животных')
plt.xlabel('Метка животного')
plt.ylabel('Возраст (лет)')
plt.grid(True)
plt.show()