import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn import decomposition

from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
np.set_printoptions(suppress=True, precision=8)

wine = load_wine()
df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
y = wine['target']

df['target']=y

df.head(5)

X=df.drop('target',axis=1)
y=df['target']

X_mean = X - np.mean(X , axis = 0)
covariance_matrix = np.cov(X_mean , rowvar = False)
eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
print('The eigen values are',sorted_eigenvalue)
print('\n')
sorted_eigenvectors = eigen_vectors[:,sorted_index]
print('The eigen vectors are',sorted_eigenvectors)
print('\n')
eigenvector_subset = sorted_eigenvectors[:,0:2]
X_reduced = np.dot(eigenvector_subset.transpose() , X_mean.transpose() ).transpose()
df1= pd.DataFrame(X_reduced, columns = ['PC1','PC2'])
df_X_reduced=pd.concat([df1 , pd.DataFrame(y)] , axis = 1)
print('The first two principle component scores are:')
df_X_reduced


df_X_reduced

import numpy as np
from sklearn.decomposition import PCA

## TODO 2
pca = decomposition.PCA()
pca.fit(X)
X_transformed = pca.transform(X)

explained_variance=pca.explained_variance_
proportion_variance=pca.explained_variance_ratio_
cummulative_proportion_variance=np.cumsum(pca.explained_variance_ratio_)


print('The first 2 principle components are',X_transformed)
print('\n')
print('The value of explained variance is',explained_variance)
print('\n')
print('The value of cummulative proportion of variance is',cummulative_proportion_variance)


from sklearn.preprocessing import StandardScaler
### TODO 3
pca=PCA(n_components=2)
pca_scaled=StandardScaler().fit_transform(X)
df1_PCA=pca.fit_transform(pca_scaled)
df_PCA=pd.DataFrame(df1_PCA, columns=['Principle Component 1','Principle Component 2'])

a=pca.explained_variance_
print('The value of explained variance is',a)
print('\n')
b=pca.explained_variance_ratio_
print('The value of proportion variance is',b)
print('\n')
c=np.sum(pca.explained_variance_ratio_)
print('The value of proportion variance is',c)
print('\n')
print('The two principle component scores after scaling X is:')
df_PCA



import plotly.express as px
import seaborn as sns
#sns.scatterplot(ql'pc1'l, [0] * len(res), hue=y, s=50)
fig=px.scatter (pca_scaled, x=0, y=1, color=y, title='2D plane defined by the first two PCs calculated with standardized data', 
                color_discrete_map='RdBu')
fig.show()

x = StandardScaler().fit_transform(X)
y = pd.DataFrame(y, columns=['target'])
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
pc_data = pd.DataFrame(data = principal_components
, columns = ['principal component 1', 'principal component 2'])
y[['target']].head()
finalDf = pd.concat([pc_data, y], axis = 1)
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 12)
ax.set_ylabel('Principal Component 2', fontsize = 12
)
ax.set_title('2 Component PCA', fontsize = 20)
targets = [0,1,2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    , finalDf.loc[indicesToKeep, 'principal component 2']
    , c = color
    , s = 50)
    ax.legend(targets)
