import pandas as pd
import numpy as np
df=pd.read_csv('RED Note Comments Data.csv')
df['Content'] = df['Content'].str.replace(r'\[.*?\]', '')
df['Content'].str.replace(r'@', '')
df.replace('', np.nan, regex=True, inplace=True)
df = df.dropna()
df.to_csv('Filtered RED Note Comments Data.csv', index=False)