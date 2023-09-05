import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch

class MeshManager:
    def __init__(self):
        self.triangle_areas = []

    def calculate_triangle_areas(self, mesh):
        for triangle in mesh.triangles:
            vertices = np.asarray(mesh.vertices)[triangle]
            a, b, c = vertices[0], vertices[1], vertices[2]

            edge1 = b - a
            edge2 = c - a

            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            self.triangle_areas.append(area)

    def show_histogram(self):
        plt.hist(self.triangle_areas, bins=1000, edgecolor='black')
        plt.title("Triangle Area Histogram")
        plt.xlabel("Area")
        plt.ylabel("Frequency")
        plt.show()

    def apply_filters(self, mesh, threshold_area=0.00005):
        triangles_to_keep = np.array(self.triangle_areas) <= threshold_area

        filtered_triangles = np.array(mesh.triangles)[triangles_to_keep]
        filtered_vertices = np.array(mesh.vertices)

        return filtered_triangles, filtered_vertices

    def apply_DBSCAN(self, mesh, dataset_name, eps=0.02, min_samples=20):

        filtered_triangles, filtered_vertices = self.apply_filters(mesh)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_vertices)
        labels = clustering.labels_

        filtered_colors = labels / (np.max(labels) + 1.0)
        filtered_colors = plt.cm.jet(filtered_colors)[:, :3]

        filtered_mesh = o3d.geometry.TriangleMesh()
        filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
        filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)

        filtered_mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh(f"filtered_clustered_{dataset_name}", filtered_mesh)

        total_triangles = len(mesh.triangles)
        remaining_triangles = len(filtered_mesh.triangles)
        num_clusters = len(np.unique(labels))
        print(f"Total number of triangles before filtering: {total_triangles}")
        print(f"Number of triangles after filtering and clustering: {remaining_triangles}")
        print(f"Number of clusters: {num_clusters}")

    def apply_clustering_without_filter(self, mesh, dataset_name, eps=0.02, min_samples=20):


        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mesh.vertices)
        labels = clustering.labels_

        filtered_colors = labels / (np.max(labels) + 1.0)
        filtered_colors = plt.cm.jet(filtered_colors)[:, :3]

        filtered_mesh = o3d.geometry.TriangleMesh()
        filtered_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        filtered_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)
        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)

        filtered_mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh(f"filtered_clustered_{dataset_name}", filtered_mesh)

        total_triangles = len(mesh.triangles)
        remaining_triangles = len(filtered_mesh.triangles)
        num_clusters = len(np.unique(labels))
        print(f"Total number of triangles before filtering: {total_triangles}")
        print(f"Number of triangles after filtering and clustering: {remaining_triangles}")
        print(f"Number of clusters: {num_clusters}")
    
    def apply_birch(self, mesh, dataset_name, threshold=0.01, branching_factor=50):
        filtered_triangles, filtered_vertices = self.apply_filters(mesh)

        clustering = Birch(threshold=threshold, branching_factor=branching_factor).fit(filtered_vertices)
        labels = clustering.labels_

        filtered_colors = labels / (np.max(labels) + 1.0)
        filtered_colors = plt.cm.jet(filtered_colors)[:, :3]

        filtered_mesh = o3d.geometry.TriangleMesh()
        filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
        filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)

        filtered_mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh(f"filtered_birch_{dataset_name}", filtered_mesh)

        total_triangles = len(mesh.triangles)
        remaining_triangles = len(filtered_mesh.triangles)
        num_clusters = len(np.unique(labels))
        print(f"Total number of triangles before filtering: {total_triangles}")
        print(f"Number of triangles after filtering and clustering: {remaining_triangles}")
        print(f"Number of clusters: {num_clusters}")
    
