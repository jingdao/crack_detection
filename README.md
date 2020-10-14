# Crack Segmentation from Disaster Site Point Clouds using Anomaly Detection


## Crack segmentation

```
python crack_detection.py --obj column --mode tle --clustering isolation --resolution 0.001 --viz
```

```
for i in rgb int norm curv fpfh pointnet2 tle
do
    python -u crack_detection.py --obj slab --mode $i --clustering isolation --resolution 0.01 --viz >> results/slab_feature_analysis.txt
done
```

## Dependencies

1. numpy
2. scipy
3. matplotlib
4. networkx
5. pyquaternion
6. scikit-learn
7. tensorflow
