import numpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
import itertools
import math
from sklearn.decomposition import PCA
import os

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

#mode = sys.argv[1]
#mode = 'norm'
#mode = 'curv'
#mode = 'int'
#mode = 'rgb'
#mode = 'feat'
mode = 'fpfh'
save_viz = False
print('params',sys.argv)
agg_precision = []
agg_recall = []
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
#    print('Crack',numpy.sum(crack_mask))

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
    agg_precision.append(precision)
    agg_recall.append(recall)
    print('Column %d precision %.2f recall %.2f'%(column_id, precision, recall))

print('Overall %s precision %.2f recall %.2f'%(mode, numpy.mean(agg_precision), numpy.mean(agg_recall)))
