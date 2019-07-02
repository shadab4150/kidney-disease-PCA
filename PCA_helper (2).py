import math
import pandas as pd
from sklearn import preprocessing



def scaleFeatures(df):
  
  
  # http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
  
  # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
  


  
  scaled = preprocessing.StandardScaler().fit_transform(df)
  scaled = pd.DataFrame(scaled, columns=df.columns)
  print("New Variances:\n", scaled.var())
  print("New Describe:\n", scaled.describe())
  return scaled


def drawVectors(transformed_features, components_, columns, plt, scaled):
  if not scaled:
    return plt.axes()

  num_columns = len(columns)

  # This funtion will project your *original* feature (columns) onto your principal component feature-space, so that you can
 
  
  # Scale the principal components by the max value in the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])

  ## visualize projections

 
  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print("Features by importance:\n", important_features)

  ax = plt.axes()

  for i in range(num_columns):
    
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

  return ax
