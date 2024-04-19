import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

import sklearn 
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df=pd.DataFrame(np.c_[housing.data, housing.target],columns=housing.feature_names + ['target'])
print(df.head())
print(df.describe())

# Check for missing values
print(df.isnull().sum())
# Handle missing values if needed
# df = df.dropna()


sns.histplot(df['target'], bins=30, kde=True)
plt.show()
