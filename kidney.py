import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper


matplotlib.style.use('ggplot')


scaleFeatures = False


df = pd.read_csv('F:\\CSV files\\kidney_disease.csv')
df.dropna(axis = 0, how = 'any', inplace = True)
print(df.head())


labels = ['red' if i=='ckd' else 'green' for i in df.classification]


df = df[['bgr', 'wc', 'rc']]


print(df)
print(df.dtypes) 	
df.bgr = pd.to_numeric(df.bgr)
df.wc = pd.to_numeric(df.wc)
df.rc = pd.to_numeric(df.rc)
print(df)
print(df.dtypes) 	


print(df.var())
print("This is the describe output: ", df.describe())




if scaleFeatures: df = helper.scaleFeatures(df)


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(df)
T = pca.transform(df)


ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


