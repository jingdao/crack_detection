import numpy
import matplotlib.pyplot as plt
from util import get_crack_dimensions
import sys
import os
plt.rcParams.update({'font.size': 15})

mode_to_string = {
    'rgb': "RGB color",
    'int': "Intensity",
    'norm': "Normal",
    'curv': "Curvature",
    'fpfh': "FPFH",
    'pointnet2': "PointNet++",
    'feat': "CrackEmbed",
    'tle': "CrackEmbed",
}

labels = []
scores = []
f = open('results/column_feature_analysis.txt', 'r')
for l in f:
    if l.startswith('Overall'):
        labels.append(mode_to_string[l.split()[1]])
        scores.append([float(l.split()[7])*100, float(l.split()[9])*1000+1, float(l.split()[11])*1000])
f.close()
metrics = ['F1-score (%)', 'Length error (mm)', 'Width error (mm)']
width = 1.0 / (len(labels) + 1)
ind = numpy.arange(len(metrics))
plt.style.use('dark_background')
plt.figure()
for i in range(len(labels)):
    plt.bar(ind + width*i, scores[i], width, label=labels[i])
plt.xticks(ind + 0.5, metrics)
plt.legend()

labels = []
scores = []
f = open('results/slab_feature_analysis.txt', 'r')
for l in f:
    if l.startswith('Overall'):
        labels.append(mode_to_string[l.split()[1]])
        scores.append([float(l.split()[7])*100, float(l.split()[9])*1000+1, float(l.split()[11])*1000])
f.close()
width = 1.0 / (len(labels) + 1)
plt.style.use('dark_background')
plt.figure()
for i in range(len(labels)):
    plt.bar(ind + width*i, scores[i], width, label=labels[i])
plt.xticks(ind + 0.5, metrics)
plt.legend()

labels = ['Isolation Forest', 'One-class SVM', 'Robust Covariance', 'Local Outlier Factor', 'K-means', 'Gaussian Mixture Model', 'Mean-shift']
f = open('results/clustering_analysis.txt', 'r')
scores = []
for l in f:
    if l.startswith('Overall'):
        scores.append([float(l.split()[7])*100, float(l.split()[9])*1000+1, float(l.split()[11])*1000])
f.close()
scores = numpy.array(scores)
scores = numpy.minimum(scores, 100)
width = 1.0 / (len(labels) + 1)
plt.style.use('dark_background')
plt.figure()
for i in range(len(labels)):
    plt.bar(ind + width*i, scores[i], width, label=labels[i])
plt.xticks(ind + 0.5, metrics)
plt.legend()
plt.show()

F1 = {}
width = {}
resolutions = []
f = open('results/results_resolution_analysis.txt', 'r')
for l in f:
    if l.startswith('Downsample'):
        resolution = float(l.split()[2])
        if not resolution in resolutions:
            resolutions.append(resolution)
    elif l.startswith('Overall'):
        mode = l.split()[1]
        Fscore = float(l.split()[7])
        Wscore = float(l.split()[11])
        if not mode in F1:
            F1[mode] = []
            width[mode] = []
        Fscore = max(Fscore, 0.1)
        if Wscore==0:
            Wscore = numpy.nan
        F1[mode].append(Fscore)
        width[mode].append(Wscore)
f.close()

plt.style.use('dark_background')
plt.figure()
for m in F1:
    plt.loglog(resolutions, F1[m], '-x', label=mode_to_string[m])
plt.xlabel('Resolution (m)')
plt.ylabel('F1 score')
plt.legend()
plt.figure()
for m in F1:
    plt.loglog(resolutions, width[m], '-x', label=mode_to_string[m])
plt.xlabel('Resolution (m)')
plt.ylabel('Width Error (m)')
plt.legend()
plt.show()

F1 = {}
f1_array = []
f = open('results/column_feature_analysis.txt', 'r')
for l in f:
    if l.startswith('Column'):
        f1_array.append(float(l.split()[7]))
    if l.startswith('Overall'):
        mode = l.split()[1]
        F1[mode] = f1_array
        f1_array = []
f.close()

if os.path.exists('tmp/dimensions.npy'):
    dimensions = numpy.load('tmp/dimensions.npy')
else:
    dimensions = []
    for column_id in range(1, 8):
        column = numpy.loadtxt('data/column%d.ply' % column_id, skiprows=13)
        print('column', column_id, column.shape)

        crack_mask = numpy.load('data/column%d_mask.npy'%column_id)
        crack_main_mask = numpy.load('tmp/column%d_main_mask.npy'%column_id)
        gt_length, gt_width = get_crack_dimensions(column[crack_main_mask, :3])
        print(numpy.sum(crack_mask), numpy.sum(crack_main_mask), gt_length, gt_width)
        outlier_ratio = 1.0 * numpy.sum(crack_main_mask) / len(column)
        dimensions.append([gt_width, outlier_ratio])
    numpy.save('tmp/dimensions.npy', numpy.array(dimensions))

plt.style.use('dark_background')
plt.figure()
for m in F1:
    x = dimensions[:, 0]
    y = F1[m]
    x, y = zip(*sorted(zip(x, y)))
    plt.plot(x, y, '-x', label=mode_to_string[m])
plt.xlabel('Crack width (m)')
plt.ylabel('F1 score')
plt.legend()
plt.figure()
for m in F1:
    x = dimensions[:, 1]
    y = F1[m]
    x, y = zip(*sorted(zip(x, y)))
    plt.plot(x, y, '-x', label=mode_to_string[m])
plt.xlabel('Outlier ratio')
plt.ylabel('F1 score')
plt.legend()
plt.show()
