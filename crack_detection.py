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
import os
import networkx as nx
import argparse
from util import loadPLY, savePLY, loadFPFH, savePCD, get_crack_dimensions

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type=float, default=0.001)
parser.add_argument('--mode', type=str, default='tle')
parser.add_argument('--clustering', type=str, default='kmeans')
parser.add_argument('--param', type=int, default=0)
parser.add_argument('--viz', action='store_true')
args = parser.parse_args()

if args.mode=='pointnet2':
    from pointnet_features import get_pointnet_features
if args.mode=='tle':
    from triplet_loss_embedding import get_triplet_loss_embedding

K_param = {
    'rgb':3,
    'int':3,
    'norm':3,
    'curv':2,
    'fpfh':2,
    'pointnet2':5,
    'tle':5,
}
agg_precision = []
agg_recall = []
agg_F1 = []
agg_length_err = []
agg_width_err = []
for column_id in range(1, 8):
#for column_id in [99]:
    column = loadPLY('data/column%d.ply' % column_id)
    column_gray = numpy.mean(column[:,3:6], axis=1).reshape(-1,1)

    try:
        crack_mask = numpy.load('data/column%d_mask.npy'%column_id)
    except FileNotFoundError:
        crack = loadPLY('tmp/column%d_crack.ply' % column_id)
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

    #equalize resolution
    unequalized_points = column
    equalized_idx = []
    equalized_map = {}
    unequalized_idx = []
    for i in range(len(unequalized_points)):
        k = tuple(numpy.round(unequalized_points[i,:3] / args.resolution).astype(int))
        if not k in equalized_map:
            equalized_map[k] = len(equalized_idx)
            equalized_idx.append(i)
        unequalized_idx.append(equalized_map[k])

    column = unequalized_points[equalized_idx]
    column_gray = column_gray[equalized_idx]
    crack_mask = crack_mask[equalized_idx]
    crack_main_mask = crack_main_mask[equalized_idx]
    print('Downsample @ %.3f %d -> %d' % (args.resolution, len(unequalized_points), len(column)))

    if args.mode=='tle':
        try:
            features = numpy.load('tmp/column%d_tle.npy'%column_id)
            assert len(features) == len(column)
        except (FileNotFoundError, AssertionError):
            features = get_triplet_loss_embedding(column[:, :6])
            numpy.save('tmp/column%d_tle.npy'%column_id, features)
    elif args.mode=='fpfh':
        try:
            fpfh = numpy.load('tmp/column%d_fpfh.npy'%column_id)
            assert len(fpfh) == len(column)
        except (FileNotFoundError, AssertionError):
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
            assert len(features) == len(column)
        except (FileNotFoundError, AssertionError):
            features = get_pointnet_features(column[:, :6])
            numpy.save('tmp/column%d_pointnet2.npy'%column_id, features)
    elif args.mode=='norm' or args.mode=='curv':
        try:
            normals = numpy.load('tmp/column%d_norm.npy'%column_id)
            curvatures = numpy.load('tmp/column%d_curv.npy'%column_id)
            assert len(normals) == len(column)
        except (FileNotFoundError, AssertionError):
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
                curvatures.append(0 if math.isnan(curvature) else numpy.fabs(curvature)) 
            normals = numpy.array(normals)
            curvatures = numpy.array(curvatures)
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
        elif args.mode in ['fpfh', 'pointnet2', 'tle']:
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
    precision = 1.0 * tp / (numpy.sum(predict_mask) + 1e-6)
    recall = 1.0 * tp / (numpy.sum(crack_mask) + 1e-6)
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
