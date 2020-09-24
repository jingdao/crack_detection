import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import sys
import itertools

class MCPNet:
	def __init__(self,batch_size, neighbor_size, feature_size, hidden_size, embedding_size):
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size-2))
		self.label_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.neighbor_pl = tf.placeholder(tf.float32, shape=(batch_size, neighbor_size, feature_size))

		#NETWORK_WEIGHTS
		kernel1 = tf.get_variable('mcp_kernel1', [1,feature_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('mcp_bias1', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel2 = tf.get_variable('mcp_kernel2', [1,hidden_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('mcp_bias2', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel3 = tf.get_variable('mcp_kernel3', [feature_size-2+hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias3 = tf.get_variable('mcp_bias3', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel4 = tf.get_variable('mcp_kernel4', [hidden_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias4 = tf.get_variable('mcp_bias4', [embedding_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.kernels = [kernel1, kernel2, kernel3, kernel4]
		self.biases = [bias1, bias2, bias3, bias4]

		#MULTI-VIEW CONTEXT POOLING
		neighbor_fc = tf.nn.conv1d(self.neighbor_pl, kernel1, 1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias1)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.nn.conv1d(neighbor_fc, kernel2, 1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias2)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.reduce_max(neighbor_fc, axis=1)
		concat = tf.concat(axis=1, values=[self.input_pl, neighbor_fc])

		#FEATURE EMBEDDING BRANCH (for instance label prediction)
		fc3 = tf.matmul(concat, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		self.fc4 = tf.matmul(fc3, kernel4)
		self.fc4 = tf.nn.bias_add(self.fc4, bias4)
		self.embeddings = tf.nn.l2_normalize(self.fc4, dim=1)

resolution = 0.1
batch_size = 1
num_neighbors = 50
neighbor_radii = 0.3
hidden_size = 200
embedding_size = 10
feature_size = 6

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size)
saver = tf.train.Saver()
MODEL_PATH = 'models/mcpnet.ckpt'
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored from %s'%MODEL_PATH)

def get_triplet_loss_embedding(pcd):

    unequalized_points = pcd.copy()
    unequalized_points[:,:3] *= 10
#    print('unequalized', unequalized_points.shape)

    #equalize resolution
    equalized_idx = []
    equalized_map = {}
    coarse_map = {}
    unequalized_idx = []
    for i in range(len(unequalized_points)):
        k = tuple(numpy.round(unequalized_points[i,:3]/resolution).astype(int))
        if not k in equalized_map:
            equalized_map[k] = len(equalized_idx)
            equalized_idx.append(i)
            kk = tuple(numpy.round(unequalized_points[i,:3]/neighbor_radii).astype(int))
            if not kk in coarse_map:
                coarse_map[kk] = []
            coarse_map[kk].append(equalized_map[k])
        unequalized_idx.append(equalized_map[k])
    points = unequalized_points[equalized_idx]
#    print('equalized',points.shape)

    #compute neighbors for each point
    neighbor_array = numpy.zeros((len(points), num_neighbors, 6), dtype=float)
    for i in range(len(points)):
        p = points[i,:6]
        k = tuple(numpy.round(points[i,:3]/neighbor_radii).astype(int))
        neighbors = []
        for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
            kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
            if kk in coarse_map:
                neighbors.extend(coarse_map[kk])
        neighbors = numpy.random.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
        neighbors = points[neighbors, :6].copy()
        neighbors -= p
        neighbor_array[i,:,:] = neighbors
#    print('neighbors',points.shape)
        
    #compute embedding for each point
    embeddings = numpy.zeros((len(points), embedding_size), dtype=float)
    input_points = numpy.zeros((batch_size, feature_size-2), dtype=float)
    input_neighbors = numpy.zeros((batch_size, num_neighbors, feature_size), dtype=float)
    num_batches = 0
    for i in range(len(points)):
        input_points[0,:] = points[i, 2:6]
        input_points[0,0] = 0.5
        input_neighbors[0,:,:] = neighbor_array[i, :, :feature_size]
        emb_val = sess.run(net.embeddings, {net.input_pl:input_points, net.neighbor_pl:input_neighbors})
        embeddings[i] = emb_val
        num_batches += 1

    return embeddings[unequalized_idx]
