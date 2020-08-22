import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
import itertools
import math
from sklearn.decomposition import PCA
from pyquaternion import Quaternion
import os
from scipy.spatial import ConvexHull
import networkx as nx

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
#    print(len(crack_points), crack_points.min(axis=0), crack_points.max(axis=0))
    # project to plane
    crack_points_centered = crack_points - numpy.mean(crack_points, axis=0)
    cov = crack_points.T.dot(crack_points_centered) / len(crack_points)
#    print('cov', cov)
    U, S, V = numpy.linalg.svd(cov)
    normal = V[2]
    offset = -numpy.mean(crack_points.dot(normal))
#    print('normal',normal, offset)
    # rotate to XY plane
    from_vec = normal
    to_vec = numpy.array([0,0,1])
    q = Quaternion(scalar=from_vec.dot(to_vec)+1, vector=numpy.cross(from_vec, to_vec)).normalised
    R = q.rotation_matrix
#    print(R)
#    print(q.axis, q.angle/numpy.pi*180)
#    print(normal.dot(R.T))
    crack_points_XY = crack_points.dot(R.T)[:, :2]
#    print(crack_points_XY.min(axis=0), crack_points_XY.max(axis=0))
    cloud = crack_points_XY[ConvexHull(crack_points_XY).vertices]
#    numpy.savetxt('crack_points.txt', crack_points)
#    numpy.savetxt('crack_points_XY.txt', crack_points_XY)
#    numpy.savetxt('hull.txt', hull)
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
#    print('crack', crack_width, crack_length)
    return crack_width, crack_length

mode = sys.argv[1]
#mode = 'rgb'
#mode = 'int'
#mode = 'norm'
#mode = 'curv'
#mode = 'fpfh'
#mode = 'feat'
save_viz = False
agg_precision = []
agg_recall = []
agg_F1 = []
agg_length_err = []
agg_width_err = []
for column_id in range(1, 8):
#for column_id in [7]:
    column = numpy.loadtxt('data/column%d.ply' % column_id, skiprows=13)
#    print('Column',column.shape)

    try:
        crack_mask = numpy.load('data/column%d_mask.npy'%column_id)
    except FileNotFoundError:
        crack = numpy.loadtxt('data/column%d_crack.ply' % (column_id), skiprows=14)
        crack_set = set([tuple(p) for p in crack[:,:3]])
        crack_mask = numpy.zeros(len(column), dtype=bool)
        for i in range(len(column)):
            p = tuple(column[i,:3])
            if p in crack_set:
                crack_mask[i] = True
        numpy.save('data/column%d_mask.npy'%column_id, crack_mask)
        column[:,3:6] = numpy.mean(column[:,3:6], axis=1).reshape(-1,1)
        column[crack_mask, 3:6] = [255,255,0]
        savePLY('data/column%d_gt.ply'%column_id, column)

    try:
        crack_main_mask = numpy.load('data/column%d_main_mask.npy'%column_id)
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
        numpy.save('data/column%d_main_mask.npy'%column_id, crack_main_mask)
        column[:,3:6] = numpy.mean(column[:,3:6], axis=1).reshape(-1,1)
        column[crack_main_mask, 3:6] = [255,255,0]
        savePLY('data/column%d_main_gt.ply'%column_id, column)
#    print('Crack',numpy.sum(crack_mask),numpy.sum(crack_main_mask))

    if mode=='feat':
        features = numpy.load('data/column%d_feat.npy'%column_id)
        if save_viz:
            X_embedded = PCA(n_components=3).fit_transform(features)
            embedded_color = (X_embedded - X_embedded.min(axis=0)) / (X_embedded.max(axis=0) - X_embedded.min(axis=0))
            column[:,3:6] = embedded_color * 255
            savePLY('viz/column%d_%s.ply' % (column_id, mode), column)
        K = 5
        kmeans = KMeans(n_clusters=K,init='k-means++',random_state=0)
        kmeans.fit(features)
        counts = [numpy.sum(kmeans.labels_==k) for k in range(K)]	
        predict_mask = kmeans.labels_==numpy.argmin(counts)
    elif mode=='fpfh':
        try:
            fpfh = numpy.load('data/column%d_fpfh.npy'%column_id)
        except FileNotFoundError:
            savePCD('tmp/tmp.pcd', column)
            R1 = 0.015
            R2 = 0.01
            os.system('pcl_normal_estimation tmp/tmp.pcd tmp/normal.pcd -radius %f' % R1)
            os.system('pcl_fpfh_estimation tmp/normal.pcd tmp/fpfh.pcd -radius %f' % R2)
            os.system('pcl_convert_pcd_ascii_binary tmp/fpfh.pcd tmp/fpfh_ascii.pcd 0')
            fpfh = loadFPFH('tmp/fpfh_ascii.pcd')
            numpy.save('data/column%d_fpfh.npy'%column_id, fpfh)
        if save_viz:
            X_embedded = PCA(n_components=3).fit_transform(fpfh)
            embedded_color = (X_embedded - X_embedded.min(axis=0)) / (X_embedded.max(axis=0) - X_embedded.min(axis=0))
            column[:,3:6] = embedded_color * 255
            savePLY('viz/column%d_%s.ply' % (column_id, mode), column)
        K = 2
        kmeans = KMeans(n_clusters=K,init='k-means++',random_state=0)
        kmeans.fit(fpfh)
        counts = [numpy.sum(kmeans.labels_==k) for k in range(K)]	
        predict_mask = kmeans.labels_==numpy.argmin(counts)
    elif mode=='norm' or mode=='curv':
        try:
            normals = numpy.load('data/column%d_norm.npy'%column_id)
            curvatures = numpy.load('data/column%d_curv.npy'%column_id)
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
            numpy.save('data/column%d_norm.npy'%column_id, normals)
            numpy.save('data/column%d_curv.npy'%column_id, curvatures)
        if save_viz:
            column[:,3:6] = normals * 255
            savePLY('viz/column%d_%s.ply' % (column_id, 'norm'), column)
            curvatures /= curvatures.max()
#            column[:,3:6] = curvatures.reshape(-1,1)*255
            column[:, 3:6] = plt.get_cmap('jet')(curvatures)[:, :3] * 255
            savePLY('viz/column%d_%s.ply' % (column_id, 'curv'), column)
        K = 2 if mode=='curv' else 3
        kmeans = KMeans(n_clusters=K,init='k-means++',random_state=0)
        kmeans.fit(curvatures.reshape(-1,1) if mode=='curv' else normals)
        counts = [numpy.sum(kmeans.labels_==k) for k in range(K)]
        predict_mask = kmeans.labels_==numpy.argmin(counts)
    else:
        rgb = column[:,3:6]
        intensity = column[:,6]
        K = 3
        kmeans = KMeans(n_clusters=K,init='k-means++',random_state=0)
        if mode=='int':
            kmeans.fit(intensity.reshape(-1,1))
        else:
            kmeans.fit(rgb)
        counts = [numpy.sum(kmeans.labels_==k) for k in range(K)]
        predict_mask = kmeans.labels_==numpy.argmin(counts)

    if save_viz:
        column[:,3:6] = numpy.mean(column[:,3:6], axis=1).reshape(-1,1)
        column[predict_mask,3:6] = [255,255,0]
        print('Found %d/%d/%d cluster'%(numpy.sum(predict_mask), numpy.sum(crack_mask), len(column)))
        savePLY('data/column%d_%s.ply' % (column_id, mode), column)

    tp = numpy.sum(numpy.logical_and(predict_mask, crack_mask))
    precision = tp / numpy.sum(predict_mask)
    recall = tp / numpy.sum(crack_mask)
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
        length_err = numpy.nan
        width_err = numpy.nan

    print('Column %d precision %.2f recall %.2f F1 %.2f length %.3f width %.3f'%(column_id, precision, recall, F1, length_err, width_err))

print('Overall %s precision %.2f recall %.2f F1 %.2f length %.3f width %.3f'%(mode, numpy.mean(agg_precision), numpy.mean(agg_recall), numpy.mean(agg_F1), numpy.mean(agg_length_err), numpy.mean(agg_width_err)))
