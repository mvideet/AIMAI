import pandas as pd
import numpy as np
df = pd.read_csv("combo11.csv",nrows = 29)
df2 = pd.read_excel("/Users/mvideet/Desktop/radio.xlsx", nrows=29)
b= df.iloc[15]
print(b)
a = str(df2[['Clinical signs/ symptoms']].iloc[27])
print("heloo")
print(a)
value = df2.at[27, 'Clinical signs/ symptoms']
value2= df.at[15, 'embedded_symptoms']
print(value2)