import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import random

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

    def apply_filters(self, mesh,threshold_area=0.00005 ): #threshold_area=0.00005
        triangles_to_keep = np.array(self.triangle_areas) <= threshold_area

        filtered_triangles = np.array(mesh.triangles)[triangles_to_keep]
        filtered_vertices = np.array(mesh.vertices)

        return filtered_triangles, filtered_vertices

    def apply_DBSCAN(self, mesh, dataset_name, eps=0.1, min_samples=20):
        filtered_triangles, filtered_vertices = self.apply_filters(mesh)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_vertices)
        labels = clustering.labels_

        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)

        clusters = []  # List to store clusters

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            cluster_points = filtered_vertices[labels == label]
            clusters.append(cluster_points)

        self.labels = labels  # Store the cluster labels

        # Generate a list of distinct colors for each cluster
        distinct_colors = [tuple(random.random() for _ in range(3)) for _ in range(num_clusters)]

        # Create a dictionary to map labels to colors
        label_color_mapping = {label: distinct_colors[i] for i, label in enumerate(unique_labels)}

        # Assign colors to labels based on the label_color_mapping
        filtered_colors = [label_color_mapping[label] for label in labels]

        filtered_mesh = o3d.geometry.TriangleMesh()
        filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
        filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

        # Convert filtered_colors to a valid format for vertex colors
        filtered_colors = np.asarray(filtered_colors, dtype=np.float64)
        filtered_colors = np.clip(filtered_colors, 0.0, 1.0)  # Clip colors to [0, 1]

        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)

        filtered_mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh(f"filtered_clustered_{dataset_name}", filtered_mesh)

        total_triangles = len(mesh.triangles)
        remaining_triangles = len(filtered_mesh.triangles)
        num_clusters = len(unique_labels)
        print(f"Total number of triangles before filtering: {total_triangles}")
        print(f"Number of triangles after filtering and clustering: {remaining_triangles}")
        print(f"Number of clusters: {num_clusters}")

        # Print the label-color mapping
        print("Label-Color Mapping:")
        for label, color in label_color_mapping.items():
            print(f"Label {label}: Color {color}")

        return label_color_mapping , clusters

    def apply_clustering_without_filter(self, mesh, dataset_name, eps=0.1, min_samples=20):

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

    def apply_birch(self, mesh, dataset_name, threshold=0.1, branching_factor=50):
        
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

    def find_object(self, point_cloud1, point_cloud2,label):
        length1= len(point_cloud1)
        length2= len(point_cloud2)

        treshhold=(length1/100)*10

        if (length1 == length2) or (length1+treshhold == length2) or (length2+treshhold == length1):
            print(f"The object you are looking for matches the label cluster {label}")
            self.visualize_point_cloud(point_cloud2)
        else:
            print(f"Object doesn't matches with label cluster {label}")

    def visualize_point_cloud(self,point_cloud):
        # Extract x, y, and z coordinates from the point cloud
        x = [point[0] for point in point_cloud]
        y = [point[1] for point in point_cloud]
        z = [point[2] for point in point_cloud]

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='.')

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()
