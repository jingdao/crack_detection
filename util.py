import numpy
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull

def loadPLY(filename):
	vertices = []
	faces = []
	numV = 0
	numF = 0
	f = open(filename,'r')
	while True:
		l = f.readline()
		if l.startswith('element vertex'):
			numV = int(l.split()[2])
		elif l.startswith('element face'):
			numF = int(l.split()[2])
		elif l.startswith('end_header'):
			break
	for i in range(numV):
		l = f.readline()
		vertices.append([float(j) for j in l.split()])
	return numpy.array(vertices)

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


