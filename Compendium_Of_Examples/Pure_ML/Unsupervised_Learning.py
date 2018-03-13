from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GMM
from sklearn.decomposition import PCA,FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from operator import itemgetter
from scipy.stats import kurtosis
from sklearn import metrics
from scipy.stats.mstats import normaltest





bmi = pd.ExcelFile("L:\GA_ML\HW_3\s_aba_normal_total.xlsx")
data = bmi.parse("s_aba_train_normal")
#--------------------------------------------------------------------------------------------------------

targetName = 'rings'

Y = data[targetName]
X = data.drop(targetName,axis=1)


def reconError(old, new):
    errs = old.values - new.values
    errs  = errs **2
    sumerrs = sum(errs)
    sumerrs = sum(sumerrs)
    return sumerrs


'''
#Kmeans
scs = []
for i in range(2,20):
    knn = KMeans(n_clusters=i).fit(X)
    labels = knn.labels_
    sc = silhouette_score(X, labels)
    scs.append([i,sc])
    #print(sc)

scs = sorted(scs,key=itemgetter(1),reverse=True)
scs = pd.DataFrame(scs,columns=['K','Score'])

sns.lmplot(x='K',y='Score',data=scs, fit_reg=False)
plt.show()
'''

'''
#Finds the K that maximizes AR score
goods  = []
for i in range(2,20):
    labels = KMeans(n_clusters=i).fit(X).labels_
    labels_true = Y.tolist()
    #labels = [3]*100+[4]
    #labels_true = [1]*99+[0]*2
    goodness = metrics.adjusted_rand_score(labels_true,labels)
    goods.append([i,goodness])
goods = pd.DataFrame(goods)
print(goods)
'''

'''
#EM
scs = []
for i in range(2,20):
    gmm = GMM(n_components=i).fit(X)
    labels = gmm.predict(X)
    sc = silhouette_score(X, labels)
    scs.append([i,sc])
    #print(sc)

scs = sorted(scs,key=itemgetter(1),reverse=True)
scs = pd.DataFrame(scs,columns=['K','Score'])

sns.lmplot(x='K',y='Score',data=scs, fit_reg=False)
plt.show()

gmm = GMM(n_components=2).fit(X)
labels = gmm.predict(X)
'''

'''
#Finds the K that maximizes AR score
goods  = []
for i in range(2,20):
    gmm = GMM(n_components=i).fit(X)
    labels = gmm.predict(X)
    labels_true = Y.tolist()
    #labels = [3]*100+[4]
    #labels_true = [1]*99+[0]*2
    goodness = metrics.adjusted_rand_score(labels_true,labels)
    print(goodness)
    goods.append([i,goodness])
goods = pd.DataFrame(goods)
print(goods)
'''







#Dimension Reduction Algorithms---------------------------------------------------------------------------------------------

'''
k, p = normaltest(X,axis=0)
p = pd.DataFrame(p)
print(p)
'''

'''
pca = PCA(n_components=10)
pca.fit(X)
newdata = pca.fit_transform(X)
newdata = pd.DataFrame(newdata)
print(pd.DataFrame(pca.components_))
print(pd.DataFrame(pca.explained_variance_ratio_))
covar_o = np.cov(np.transpose(X.values))
eigval_o, eigvec_o = np.linalg.eig(covar_o)
#print(eigvec_o)
#print(eigval_o)
#covar = np.cov(np.transpose(newdata.values))
#eigval, eigvec = np.linalg.eig(covar)
#print(eigvec)
#print(eigval)
'''

'''
pca = PCA(n_components=1).fit(X)
newdata = pca.transform(X)
newdata = pd.DataFrame(newdata)
recon = pca.inverse_transform(newdata)
recon = pd.DataFrame(recon)
print(reconError(X, recon))
#print(ica.components_)
'''

'''
stats = []
k = kurtosis(X,axis = 0, fisher = True)
k_mean = np.mean(kurtosis(X,axis = 0, fisher = True))
k_std = np.std(kurtosis(X,axis = 0, fisher = True))
stats.append([0,k_mean,k_std])
for i in range(1,11):
    ica = FastICA(n_components=i).fit(X)
    newdata = ica.transform(X)
    newdata = pd.DataFrame(newdata)
    #print(ica.components_)
    k = kurtosis(newdata,axis = 0, fisher = True)
    k_mean = np.mean(kurtosis(newdata,axis = 0, fisher = True))
    k_std = np.std(kurtosis(newdata,axis = 0, fisher = True))
    stats.append([i,k_mean,k_std])
stats = pd.DataFrame(stats)
print(stats)
'''


'''
ica = FastICA(n_components=1).fit(X)
newdata = ica.transform(X)
newdata = pd.DataFrame(newdata)
recon = ica.inverse_transform(newdata)
recon = pd.DataFrame(recon)
print(reconError(X, recon))
#print(ica.components_)
'''

'''
rp = GaussianRandomProjection(n_components=7)
newdata = rp.fit_transform(X)
newdata = pd.DataFrame(newdata)
mat = np.matrix(rp.components_)
recon = np.matrix(newdata.values) * mat
recon = pd.DataFrame(recon)
print(reconError(X, recon))
'''



'''
#Finds the K that maximizes AR score
avg = pd.DataFrame()
for j in range(0,10):
    goods  = []
    for i in range(2,20):
        labels = KMeans(n_clusters=i).fit(newdata).labels_
        labels_true = Y.tolist()
        goodness = metrics.adjusted_rand_score(labels_true,labels)
        #print(goodness)
        goods = goods + [goodness]
    goods = pd.DataFrame(goods)
    avg = pd.concat([avg,goods],axis=1)
print(avg)



avg = pd.DataFrame()
for j in range(0,10):
    goods  = []
    for i in range(2,20):
        gmm = GMM(n_components=i).fit(newdata)
        labels = gmm.predict(newdata)
        labels_true = Y.tolist()
        goodness = metrics.adjusted_rand_score(labels_true,labels)
        #print(goodness)
        goods = goods + [goodness]
    goods = pd.DataFrame(goods)
    avg = pd.concat([avg,goods],axis=1)
print(avg)
'''

'''
fa = FeatureAgglomeration(n_clusters=7).fit(X)
newdata = fa.fit_transform(X)
newdata = pd.DataFrame(newdata)
print(X.head(10))
print(newdata.head(10))
'''


fa = FeatureAgglomeration(n_clusters=5).fit(X)
newdata = fa.transform(X)
recon = fa.inverse_transform(newdata)
recon = pd.DataFrame(recon)
print(reconError(X, recon))
print(pd.DataFrame(fa.labels_))
print(fa.n_leaves_)
print(fa.n_components)
print(pd.DataFrame(fa.children_))

'''
#Finds the K that maximizes AR score
goods  = []
for i in range(2,20):
    labels = KMeans(n_clusters=i).fit(newdata).labels_
    labels_true = Y.tolist()
    goodness = metrics.adjusted_rand_score(labels_true,labels)
    goods.append([i,goodness])
print(pd.DataFrame(goods))



#Finds the K that maximizes AR score
goods  = []
for i in range(2,20):
    gmm = GMM(n_components=i).fit(newdata)
    labels = gmm.predict(newdata)
    labels_true = Y.tolist()
    goodness = metrics.adjusted_rand_score(labels_true,labels)
    goods.append([i,goodness])
print(pd.DataFrame(goods))
'''










'''
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = np.array(X.iloc[:,[6,7]])
y = Y.values.tolist()

n_clusters = 2

# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X)

silhouette_avg = silhouette_score(X, cluster_labels)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhoutte score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors)

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1],
            marker='o', c="white", alpha=1, s=200)

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

plt.show()

'''