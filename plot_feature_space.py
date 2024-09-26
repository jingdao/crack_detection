import numpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from util import loadPLY, savePLY

obj_id = 3
column = loadPLY('data/%s%d.ply' % ('column', obj_id))
features = numpy.load('tmp/%s%d_tle.npy'%('column', obj_id))
#equalize resolution
resolution = 0.001
unequalized_points = column
equalized_idx = []
equalized_map = {}
unequalized_idx = []
for i in range(len(unequalized_points)):
    k = tuple(numpy.round(unequalized_points[i,:3] / resolution).astype(int))
    if not k in equalized_map:
        equalized_map[k] = len(equalized_idx)
        equalized_idx.append(i)
    unequalized_idx.append(equalized_map[k])
column = unequalized_points[equalized_idx]
print('column', column.shape, 'features', features.shape)

K = 5
cluster_algorithm = KMeans(n_clusters=K,init='k-means++',random_state=0)
cluster_algorithm.fit(features)
cluster_labels = cluster_algorithm.labels_.astype(numpy.int)
numpy.random.seed(2)
cluster_color = numpy.random.randint(0, 255, (K+1, 3))
cluster_color[-1] = [100, 100, 100]
obj_color = cluster_color[cluster_labels, :]
column[:, 3:6] = obj_color
savePLY('viz/%s%d_%s_%s.ply' % ('column', obj_id, 'tle', 'kmeans'), column)

X_embedded = PCA(n_components=2).fit_transform(features)
subset = numpy.random.choice(len(features), 1000, replace=False)
plt.figure()
plt.scatter(X_embedded[subset,0], X_embedded[subset,1], color=obj_color[subset,:]/255.0)

forest = IsolationForest(random_state=0, behaviour='new', contamination='auto').fit(features)
score = forest.decision_function(features)
score = (score - score.min()) / (score.max() - score.min())
obj_color = plt.get_cmap('jet')(score)[:, :3] * 255
plt.figure()
plt.scatter(X_embedded[subset,0], X_embedded[subset,1], color=obj_color[subset,:]/255.0)
plt.show()
