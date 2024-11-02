import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import networkx as nx

class SLAM:
    def __init__(self):
        # Initialize key variables for storing keypoints and LiDAR scans
        self.keypoints = []
        self.lidar_scans = []
        self.pose_graph = nx.DiGraph()  
        self.current_pose = np.eye(4) 

    def process_frame(self, points_frame):
        """
        Processes a frame of point cloud data from CARLA.
        Args:
            points_frame (o3d.geometry.PointCloud): The point cloud frame from CARLA.
        """
        points_np = np.asarray(points_frame.points)

        # Extract keypoints using feature extraction methods
        keypoints = self.extract_keypoints(points_np, method='corner_and_plane')

        self.lidar_scans.append(points_np)
        self.keypoints.append(keypoints)

        self.data_association(points_np, keypoints)

        # todo implementation, you would calculate this based on scan matching

        relative_transformation = np.eye(4)  
        self.add_pose_to_graph(relative_transformation)

    def extract_keypoints(self, scan, method='corner_and_plane'):
        """
        Extracts keypoints from a point cloud scan.
        Args:
            scan (np.ndarray): A numpy array of shape (N, 3) representing the point cloud scan.
            method (str): The method used for keypoint extraction. Options: 'corner_and_plane', 'uniform'.
        Returns:
            keypoints (np.ndarray): Extracted keypoints from the scan.
        """
        if method == 'corner_and_plane':
            # Extract corner and plane features using PCA
            keypoints = self._extract_corner_and_plane_features(scan)
        elif method == 'uniform':
            # Uniform sampling of keypoints
            keypoints = self._uniform_sampling(scan)
        else:
            raise ValueError("Unknown method for keypoint extraction")

        return keypoints

    def _extract_corner_and_plane_features(self, scan):
        """
        Extracts corner and plane features using PCA.
        Args:
            scan (np.ndarray): A numpy array of shape (N, 3) representing the point cloud scan.
        Returns:
            keypoints (np.ndarray): Keypoints representing corners and planes.
        """
        keypoints = []
        neighborhood_size = 20  #

        kdtree = KDTree(scan)
        for point in scan:
            # Find neighbors using KD-Tree
            indices = kdtree.query([point], k=neighborhood_size, return_distance=False)[0]
            neighbors = scan[indices]

            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)

            # Check eigenvalue distribution to classify as corner or plane
            if np.min(eigenvalues) / np.max(eigenvalues) < 0.1: 
                keypoints.append(point)
            elif np.max(eigenvalues) / np.sum(eigenvalues) < 0.9:  
                keypoints.append(point)

        return np.array(keypoints)

    def _uniform_sampling(self, scan, sample_rate=0.1):
        """
        Performs uniform sampling to select keypoints from the scan.
        Args:
            scan (np.ndarray): A numpy array of shape (N, 3) representing the point cloud scan.
            sample_rate (float): The fraction of points to sample.
        Returns:
            keypoints (np.ndarray): Uniformly sampled keypoints.
        """
        num_samples = int(sample_rate * len(scan))
        indices = np.random.choice(len(scan), num_samples, replace=False)
        return scan[indices]

    def data_association(self, current_scan, current_keypoints):
        """
        Performs data association between successive frames.
        Args:
            current_scan (np.ndarray): The current point cloud scan.
            current_keypoints (np.ndarray): The extracted keypoints from the current scan.
        """
        if len(self.keypoints) < 2:
            return  
        previous_keypoints = self.keypoints[-2]
        kdtree = KDTree(previous_keypoints)
        _, indices = kdtree.query(current_keypoints, k=1)
        
        print(f"Number of associated keypoints: {len(indices)}")

    def initialize_pose_graph(self):
        """
        Initializes the pose graph with the first node.
        """
        self.pose_graph.add_node(0, pose=self.current_pose)
        print("Pose graph initialized with the first node (pose at origin).")

    def add_pose_to_graph(self, relative_transformation):
        """
        Adds a new pose to the graph using the relative transformation from the previous pose.
        Args:
            relative_transformation (np.ndarray): A 4x4 transformation matrix representing the
                                                  relative pose change from the previous node.
        """
        # Compute the new pose by multiplying the current pose with the relative transformation
        new_pose = np.dot(self.current_pose, relative_transformation)

        self.current_pose = new_pose

        node_index = len(self.pose_graph.nodes)
        self.pose_graph.add_node(node_index, pose=new_pose)

        # Add an edge between the previous pose and the new pose
        self.pose_graph.add_edge(node_index - 1, node_index, transformation=relative_transformation)
        print(f"Added node {node_index} to the pose graph with a new edge.")