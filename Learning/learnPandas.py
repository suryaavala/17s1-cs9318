import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

#to read a csv
df = pd.read_csv('./asset/lecture_data.txt', sep='\t')

print(df.head())
