from openni import openni2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cloudToMesh as ctm
import meshManager as HaF
import time

# #Take images from camera then convert it to mesh
# ctm.convertMesh()

#Load the 3D mesh
i=1
for i in range(1,12):
    dataset_name = f"dataset{i}.ply"

    mesh = o3d.io.read_triangle_mesh(dataset_name)
    obj=HaF.MeshManager()

    #Calculating triangles
    start_time = time.time()
    obj.calculate_triangle_areas(mesh)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for calculating: {elapsed_time} seconds")

    # #Showing Histogram
    # obj.show_histogram()

    #Filtering
    start_time = time.time()
    obj.apply_filters(mesh)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for filtering: {elapsed_time} seconds")

    #Clustering
    start_time = time.time()
    obj.apply_DBSCAN(mesh,dataset_name)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for clustering: {elapsed_time} seconds")
