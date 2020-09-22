import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import itertools
import math
from sklearn.decomposition import PCA
from pyquaternion import Quaternion
import os
from scipy.spatial import ConvexHull
import networkx as nx
import argparse

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def savePLY(filename, points):
	f = open(filename, 'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

def loadFPFH(filename):
	pcd = open(filename,'r')
	for l in pcd:
		if l.startswith('DATA'):
			break
	features = []
	for l in pcd:
		features.append([float(t) for t in l.split()[:33]])
	features = numpy.array(features)
	return features

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print('Saved %d points to %s' % (l,filename))

def get_crack_dimensions(crack_points):
    # project to plane
    crack_points_centered = crack_points - numpy.mean(crack_points, axis=0)
    cov = crack_points.T.dot(crack_points_centered) / len(crack_points)
    U, S, V = numpy.linalg.svd(cov)
    normal = V[2]
    offset = -numpy.mean(crack_points.dot(normal))
    # rotate to XY plane
    from_vec = normal
    to_vec = numpy.array([0,0,1])
    q = Quaternion(scalar=from_vec.dot(to_vec)+1, vector=numpy.cross(from_vec, to_vec)).normalised
    R = q.rotation_matrix
    crack_points_XY = crack_points.dot(R.T)[:, :2]
    cloud = crack_points_XY[ConvexHull(crack_points_XY).vertices]
    edgeAngles = []
    for i in range(len(cloud)-1):
        theta = numpy.arctan2(cloud[i+1,1]-cloud[i,1], cloud[i+1,0]-cloud[i,0])
        edgeAngles.append(theta)
    minArea = float('inf');
    crack_width = 0
    crack_length = 0
    for theta in edgeAngles:
        R = numpy.array([
            [numpy.cos(theta), numpy.sin(theta)],
            [-numpy.sin(theta), numpy.cos(theta)]
        ])
        rotated = R.dot(cloud.T).T
        xmin, ymin = rotated.min(axis=0)
        xmax, ymax = rotated.max(axis=0)
        area = (xmax-xmin) * (ymax-ymin)
        if xmax-xmin > ymax-ymin and area < minArea:
            minArea = area
            crack_width = xmax-xmin
            crack_length = ymax-ymin
            box = numpy.array([
                [xmin,ymin],
                [xmax,ymin],
                [xmin,ymax],
                [xmax,ymax],
            ])
            box = R.T.dot(box.T).T
            loadX = R[0,0] * (xmin+xmax)/2 + R[1,0] * (ymin+ymax)/2
            loadY = R[0,1] * (xmin + xmax)/2 + R[1,1] * (ymin + ymax)/2
    return crack_width, crack_length

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='feat')
parser.add_argument('--clustering', type=str, default='kmeans')
parser.add_argument('--param', type=int, default=0)
parser.add_argument('--viz', action='store_true')
args = parser.parse_args()

if args.mode=='pointnet2':
    from pointnet_features import get_pointnet_features

K_param = {
    'rgb':3,
    'int':3,
    'norm':3,
    'curv':2,
    'fpfh':2,
    'pointnet2':5,
    'feat':5,
}
agg_precision = []
agg_recall = []
agg_F1 = []
agg_length_err = []
agg_width_err = []
for column_id in range(1, 8):
#for column_id in [1]:
    column = numpy.loadtxt('data/column%d.ply' % column_id, skiprows=13)
    column_gray = numpy.mean(column[:,3:6], axis=1).reshape(-1,1)

    try:
        crack_mask = numpy.load('data/column%d_mask.npy'%column_id)
    except FileNotFoundError:
        crack = numpy.loadtxt('tmp/column%d_crack.ply' % (column_id), skiprows=14)
        crack_set = set([tuple(p) for p in crack[:,:3]])
        crack_mask = numpy.zeros(len(column), dtype=bool)
        for i in range(len(column)):
            p = tuple(column[i,:3])
            if p in crack_set:
                crack_mask[i] = True
        numpy.save('data/column%d_mask.npy'%column_id, crack_mask)
        column[:,3:6] = column_gray
        column[crack_mask, 3:6] = [255,255,0]
        savePLY('tmp/column%d_gt.ply'%column_id, column)

    try:
        crack_main_mask = numpy.load('tmp/column%d_main_mask.npy'%column_id)
    except FileNotFoundError:
        crack = column[crack_mask]
        resolution = 0.008
        voxel_map = {}
        edges = []
        for i in range(len(crack)):
            k = tuple(numpy.round(crack[i, :3] / resolution).astype(int))
            if not k in voxel_map:
                voxel_map[k] = []
            voxel_map[k].append(i)
        for k in voxel_map:
            for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                if kk in voxel_map:
                    for i in voxel_map[k]:
                        for j in voxel_map[kk]:
                            edges.append([i, j])
        G = nx.Graph(edges)
        clusters = nx.connected_components(G)
        clusters = [list(c) for c in clusters]
        max_cluster_size = max([len(c) for c in clusters])
        crack_main_mask = numpy.zeros(len(crack_mask), dtype=bool)
        for c in clusters:
            if len(c)==max_cluster_size:
                crack_main_mask[numpy.nonzero(crack_mask)[0][c]] = True
                break
        numpy.save('tmp/column%d_main_mask.npy'%column_id, crack_main_mask)
        column[:,3:6] = column_gray
        column[crack_main_mask, 3:6] = [255,255,0]
        savePLY('tmp/column%d_main_gt.ply'%column_id, column)

    if args.mode=='feat':
        features = numpy.load('tmp/column%d_feat.npy'%column_id)
    elif args.mode=='fpfh':
        try:
            fpfh = numpy.load('tmp/column%d_fpfh.npy'%column_id)
        except FileNotFoundError:
            savePCD('tmp/tmp.pcd', column)
            R1 = 0.015
            R2 = 0.01
            os.system('pcl_normal_estimation tmp/tmp.pcd tmp/normal.pcd -radius %f' % R1)
            os.system('pcl_fpfh_estimation tmp/normal.pcd tmp/fpfh.pcd -radius %f' % R2)
            os.system('pcl_convert_pcd_ascii_binary tmp/fpfh.pcd tmp/fpfh_ascii.pcd 0')
            fpfh = loadFPFH('tmp/fpfh_ascii.pcd')
            numpy.save('data/column%d_fpfh.npy'%column_id, fpfh)
        features = fpfh
    elif args.mode=='pointnet2':
        try:
            features = numpy.load('tmp/column%d_pointnet2.npy'%column_id)
        except FileNotFoundError:
            features = get_pointnet_features(column[:, :6])
            numpy.save('tmp/column%d_pointnet2.npy'%column_id, features)
    elif args.mode=='norm' or args.mode=='curv':
        try:
            normals = numpy.load('tmp/column%d_norm.npy'%column_id)
            curvatures = numpy.load('tmp/column%d_curv.npy'%column_id)
        except FileNotFoundError:
            normal_grid = {}
            resolution = 0.015
            for i in range(len(column)):
                k = tuple(numpy.round(column[i,:3]/resolution).astype(int))
                if not k in normal_grid:
                    normal_grid[k] = []
                normal_grid[k].append(i)
            normals = []
            curvatures = []
            for i in range(len(column)):
                k = tuple(numpy.round(column[i,:3]/resolution).astype(int))
                neighbors = []
                for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                    kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                    if kk in normal_grid:
                        neighbors.extend(normal_grid[kk])
                accA = numpy.zeros((3,3))
                accB = numpy.zeros(3)
                for n in neighbors:
                    p = column[n,:3]
                    accA += numpy.outer(p,p)
                    accB += p
                cov = accA / len(neighbors) - numpy.outer(accB, accB) / len(neighbors)**2
                U,S,V = numpy.linalg.svd(cov)
                curvature = S[2] / (S[0] + S[1] + S[2])
                normals.append(numpy.fabs(V[2]))
                curvatures.append(0 if math.isnan(curvature) else numpy.fabs(curvature)) # change to absolute values?
            normals = numpy.array(normals) #(N,3)
            curvatures = numpy.array(curvatures) #(N,)
            numpy.save('tmp/column%d_norm.npy'%column_id, normals)
            numpy.save('tmp/column%d_curv.npy'%column_id, curvatures)
        features = curvatures.reshape(-1, 1) if args.mode=='curv' else normals
    else:
        rgb = column[:,3:6]
        intensity = column[:,6]
        features = intensity.reshape(-1, 1) if args.mode=='int' else rgb

    if args.clustering in ['kmeans', 'gmm', 'dbscan', 'meanshift', 'spectral']:
        K = K_param[args.mode]
        if args.clustering=='kmeans':
            cluster_algorithm = KMeans(n_clusters=K,init='k-means++',random_state=0)
        elif args.clustering=='gmm':
            cluster_algorithm = GaussianMixture(n_components=K, covariance_type='full')
        elif args.clustering=='dbscan':
            cluster_algorithm = DBSCAN(eps=0.5, algorithm='brute')
        elif args.clustering=='meanshift':
            cluster_algorithm = MeanShift(min_bin_freq=1000, cluster_all=False, max_iter=10)
        elif args.clustering=='spectral':
            cluster_algorithm = SpectralClustering(n_clusters=K)
        cluster_algorithm.fit(features)
        if hasattr(cluster_algorithm, 'labels_'):
            print('labels')
            cluster_labels = cluster_algorithm.labels_.astype(numpy.int)
        else:
            cluster_labels = cluster_algorithm.predict(features)
        if args.clustering in ['dbscan', 'meanshift']:
            K = cluster_labels.max() + 1
        counts = [numpy.sum(cluster_labels==k) for k in range(K)]	
        if args.clustering=='meanshift':
            predict_mask = cluster_labels==-1
        else:
            predict_mask = cluster_labels==numpy.argmin(counts)
    elif args.clustering=='isolation':
        forest = IsolationForest(random_state=0, behaviour='new', contamination='auto').fit(features)
        score = forest.decision_function(features)
        predict_mask = score<0
    elif args.clustering=='svm':
        svm = OneClassSVM(gamma='auto', max_iter=1).fit(features)
        score = svm.score_samples(features)
        predict_mask = svm.predict(features)<0
    elif args.clustering=='cov':
        cov = EllipticEnvelope(random_state=0).fit(features)
        predict_mask = cov.predict(features)<0
    elif args.clustering=='lof':
        clf = LocalOutlierFactor(n_neighbors=5)
        predict_mask = clf.fit_predict(features)<0
        score = clf.negative_outlier_factor_

    if args.viz:
        # save visualization of features
        if args.mode in ['norm', 'curv']:
            column[:,3:6] = normals * 255
            savePLY('viz/column%d_%s.ply' % (column_id, 'norm'), column)
            curvatures /= curvatures.max()
            column[:, 3:6] = plt.get_cmap('jet')(curvatures)[:, :3] * 255
            savePLY('viz/column%d_%s.ply' % (column_id, 'curv'), column)
        else:
            X_embedded = PCA(n_components=3).fit_transform(features)
            embedded_color = (X_embedded - X_embedded.min(axis=0)) / (X_embedded.max(axis=0) - X_embedded.min(axis=0))
            column[:,3:6] = embedded_color * 255
            savePLY('viz/column%d_%s.ply' % (column_id, args.mode), column)
        # save visualization of clustering results
        if args.clustering=='meanshift':
            cluster_color = numpy.random.randint(0, 255, (K+1, 3))
            cluster_color[-1] = [100, 100, 100]
            column[:, 3:6] = cluster_color[cluster_labels, :]
            savePLY('viz/column%d_%s_%s.ply' % (column_id, args.mode, args.clustering), column)
        elif args.clustering in ['kmeans', 'gmm', 'dbscan', 'spectral']:
            cluster_color = numpy.random.randint(0, 255, (K, 3))
            column[:, 3:6] = cluster_color[cluster_labels, :]
            savePLY('viz/column%d_%s_%s.ply' % (column_id, args.mode, args.clustering), column)
        elif args.clustering in ['isolation', 'svm', 'lof']:
            score = (score - score.min()) / (score.max() - score.min())
            column[:, 3:6] = plt.get_cmap('jet')(score)[:, :3] * 255
            savePLY('viz/column%d_%s_%s.ply' % (column_id, args.mode, args.clustering), column)
        # save visualization of predict_mask
        column[:,3:6] = column_gray
        column[predict_mask,3:6] = [255,255,0]
        print('Found %d/%d/%d cluster'%(numpy.sum(predict_mask), numpy.sum(crack_mask), len(column)))
        savePLY('tmp/column%d_%s.ply' % (column_id, args.mode), column)

    tp = numpy.sum(numpy.logical_and(predict_mask, crack_mask))
    precision = 1.0 * tp / numpy.sum(predict_mask)
    recall = 1.0 * tp / numpy.sum(crack_mask)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    agg_precision.append(precision)
    agg_recall.append(recall)
    agg_F1.append(F1)

    gt_length, gt_width = get_crack_dimensions(column[crack_main_mask, :3])
    intersect_mask = crack_main_mask & predict_mask
    if numpy.sum(intersect_mask) > 3:
        predict_length, predict_width = get_crack_dimensions(column[intersect_mask, :3])
        length_err = numpy.abs(predict_length - gt_length)
        width_err = numpy.abs(predict_width - gt_width)
        agg_length_err.append(length_err)
        agg_width_err.append(width_err)
    else:
        predict_length = predict_width = length_err = width_err = numpy.nan

    print('Column %d precision %.2f recall %.2f F1 %.2f length %.3f/%.3f width %.3f/%.3f'%(column_id, precision, recall, F1, predict_length, gt_length, predict_width, gt_width))

print('Overall %s precision %.2f recall %.2f F1 %.2f length %.3f width %.3f'%(args.mode, numpy.mean(agg_precision), numpy.mean(agg_recall), numpy.mean(agg_F1), numpy.mean(agg_length_err), numpy.mean(agg_width_err)))
