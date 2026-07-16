import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('Bestsellers with categories.csv')

# 1-2. Null check & handle
print("Nulls:\n", df.isnull().sum())
if df.isnull().sum().any():
    for c in df:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode()[0] if df[c].dtype=='object' else df[c].median(), inplace=True)

# 3-4. Genre stats
freq = df['Genre'].value_counts()
print("\nFrequency:\n", freq)
print("\nMode:", freq.idxmax())

# 5. Charts
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
freq.plot.bar()
plt.subplot(1,2,2)
freq.plot.pie(autopct='%1.1f%%')
plt.show()

# 6. Numerical boxplot
num = df.select_dtypes(include=['int64','float64'])
plt.figure()
sns.boxplot(data=num)
plt.show()

# 7. Transform
print("\nNormalized:\n", pd.DataFrame(MinMaxScaler().fit_transform(num), columns=num.columns).head())
print("\nStandardized:\n", pd.DataFrame(StandardScaler().fit_transform(num), columns=num.columns).head())