import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import networkx as nx
from utility import *

class SLAM:
    def __init__(self):
        # Initialize key variables for storing keypoints and LiDAR scans
        self.keypoints = []
        self.lidar_scans = []
        self.estimated_poses = [] 
        self.cumulative_transformation = np.eye(4)
        self.pose_graph = nx.DiGraph()  
        self.current_pose = np.eye(4) 
        self.node_index = 0

        self.cumulative_x_translation = 0 
        self.cumulative_y_translation = 0 

    
    def set_initial_point(self, x, y):
        """
        Sets the initial point and initializes the cumulative x and y translations.
        """
        self.cumulative_x_translation = x
        self.cumulative_y_translation = y

    # def process_frame(self, points_frame):
    #     """
    #     Processes a frame of point cloud data from CARLA.
    #     Args:
    #         points_frame (o3d.geometry.PointCloud): The point cloud frame from CARLA.
    #     """
    #     points_np = np.asarray(points_frame.points)

    #     # Extract keypoints using feature extraction methods
    #     keypoints = self.extract_keypoints(points_np, method='corner_and_plane')

    #     self.lidar_scans.append(points_np)
    #     self.keypoints.append(keypoints)

    #     self.data_association(points_np, keypoints)

    #     relative_transformation = np.eye(4)  
    #     self.add_pose_to_graph(relative_transformation)



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
            # keypoints = self._extract_corner_and_plane_features(scan, 1e-15, 0.25) # deprecated function
            keypoints = self._extract_features(scan)
        elif method == 'uniform':
            # Uniform sampling of keypoints
            keypoints = self._uniform_sampling(scan)
        else:
            raise ValueError("Unknown method for keypoint extraction")

        return keypoints
    
    def _extract_features(self, scan):
        """
        Extract features using point cloud scan.
        Args:
            scan (np.ndarray): A numpy array of shape (N, 3) representing the point cloud scan.
        Returns:
            keypoints (np.ndarray): Keypoints representing corners and planes.
        """
        # Convert the numpy array to an Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(scan)

        # Estimate normals for the point cloud
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Perform plane segmentation using RANSAC
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        
        # Extract inlier points representing the plane
        plane_cloud = point_cloud.select_by_index(inliers)
        non_plane_cloud = point_cloud.select_by_index(inliers, invert=True)

        # Analyze non-plane points for corner features
        # Using the concept of curvature to detect keypoints
        non_plane_points = np.asarray(non_plane_cloud.points)
        keypoints = []

        # Loop through each point in the non-plane cloud to calculate curvature
        kdtree = o3d.geometry.KDTreeFlann(non_plane_cloud)
        for i in range(len(non_plane_points)):
            # Find the neighbors within a radius
            [_, idx, _] = kdtree.search_radius_vector_3d(non_plane_points[i], 0.1)
            if len(idx) < 5:
                continue
            
            # Compute the covariance matrix of the neighborhood
            neighbors = np.asarray(non_plane_cloud.points)[idx, :]
            covariance_matrix = np.cov(neighbors, rowvar=False)
            
            # Compute eigenvalues to analyze curvature
            eigenvalues = np.linalg.eigvalsh(covariance_matrix)
            curvature = eigenvalues[0] / np.sum(eigenvalues)
            
            # Use a threshold to detect keypoints based on curvature
            if curvature > 0.1:
                keypoints.append(non_plane_points[i])

        # Convert keypoints list to a numpy array
        keypoints = np.array(keypoints)

        # Handle the case where keypoints might be empty
        if keypoints.size == 0:
            keypoints = np.asarray(plane_cloud.points)
        else:
            keypoints = np.vstack((np.asarray(plane_cloud.points), keypoints))

        return keypoints
    

    def _extract_corner_and_plane_features(self, scan, corner_threshold=0.05, plane_threshold=0.001):
        """
        Extracts corner and plane features using PCA.
        Args:
            scan (np.ndarray): A numpy array of shape (N, 3) representing the point cloud scan.
            corner_threshold (float): Threshold to classify a corner based on eigenvalue ratio.
            plane_threshold (float): Threshold to classify a plane based on eigenvalue ratio.
        Returns:
            keypoints (np.ndarray): Keypoints representing corners and planes.
        """
        keypoints = []
        neighborhood_size = 4

        kdtree = KDTree(scan)
        for point in scan:
            # Find neighbors using KD-Tree
            neighborhood_size = min(neighborhood_size, len(scan))
            indices = kdtree.query([point], k=neighborhood_size, return_distance=False)[0]
            neighbors = scan[indices]

            cov_matrix = np.cov(neighbors, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)

            # Check eigenvalue distribution to classify as corner or plane
            if np.min(eigenvalues) / np.max(eigenvalues) < corner_threshold: 
                keypoints.append(point)
            elif np.max(eigenvalues) / np.sum(eigenvalues) < plane_threshold:  
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
        
        # print(f"Number of associated keypoints: {len(indices)}")


    def initialize_pose_graph(self):
        """
        Initializes the pose graph with the first node.
        """
        self.pose_graph.add_node(0, pose=self.current_pose)
        # print("Pose graph initialized with the first node (pose at origin).")



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
        # print(f"Added node {node_index} to the pose graph with a new edge.")


    def build_graph(self, points_frame, heading):
        """
        Builds the pose graph by adding nodes and edges based on relative transformations.
        Args:
            points_frame (o3d.geometry.PointCloud): The current point cloud frame from CARLA.
        """
        # Process the frame to extract keypoints
        points_np = np.asarray(points_frame.points)
        keypoints = self.extract_keypoints(points_np, method='corner_and_plane')

        if(keypoints is not None):
            visualize_keypoints(points_np, keypoints) # to visualize scans and keypoints

        # Store the current scan and keypoints
        self.lidar_scans.append(points_np)
        self.keypoints.append(keypoints)

        if self.node_index == 0:
            # Initialize the graph if this is the first frame
            self.initialize_pose_graph()

            # self.estimated_poses.append(np.eye(4))
            self.estimated_poses.append(self.cumulative_transformation)
        else:
            # Estimate the relative transformation between the current and previous scan
            relative_transformation = self.estimate_relative_transformation(points_frame)

            self.cumulative_transformation = self.update_cumulative_transformation(relative_transformation[0, 3], relative_transformation[1, 3], heading)

            # print("Cumulative Transformation Matrix:")
            # print(self.cumulative_transformation)

            self.estimated_poses.append(self.cumulative_transformation)

            self.add_pose_to_graph(relative_transformation)

        # Increment the node index
        self.node_index += 1



    # def update_cumulative_transformation(self, relative_transformation):
    #     """
    #     Updates the cumulative transformation matrix using the given relative transformation.
    #     Args:
    #         relative_transformation (np.ndarray): A 4x4 transformation matrix.
    #     Returns:
    #         cumulative_transformation (np.ndarray): The updated cumulative transformation matrix.
    #     """
    #     # self.cumulative_transformation = np.dot(self.cumulative_transformation, relative_transformation)
    #     return self.cumulative_transformation
        
    
    def update_cumulative_transformation(self, x_translation, y_translation, heading):
        """
        Updates the cumulative transformation matrix using the given relative transformation and heading.
        Args:
            x_translation (float): The x component of the relative translation.
            y_translation (float): The y component of the relative translation.
            heading (float): The vehicle's heading angle in radians.
        Returns:
            cumulative_transformation (np.ndarray): The updated cumulative transformation matrix.
        """
        # Calculate the projected x and y translations based on the heading angle
        projected_x = x_translation * np.cos(heading) - y_translation * np.sin(heading)
        projected_y = x_translation * np.sin(heading) + y_translation * np.cos(heading)

        # Update the cumulative x and y translations
        self.cumulative_x_translation += projected_x
        self.cumulative_y_translation += projected_y

        # Construct the cumulative transformation matrix
        cumulative_transformation = np.eye(4)
        cumulative_transformation[0, 3] = self.cumulative_x_translation
        cumulative_transformation[1, 3] = self.cumulative_y_translation

        # print("Cumulative Transformation Matrix:")
        # print(cumulative_transformation)

        return cumulative_transformation
    

    
    def estimate_relative_transformation(self, current_frame):
        """
        Estimates the relative transformation between the current and previous LiDAR scans using ICP.
        Args:
            current_frame (o3d.geometry.PointCloud): The current point cloud frame.
        Returns:
            relative_transformation (np.ndarray): A 4x4 transformation matrix.
        """
        if len(self.lidar_scans) < 2:
            return np.eye(4)  # If not enough frames, return identity transformation

        previous_frame_np = self.lidar_scans[-2]
        previous_frame = o3d.geometry.PointCloud()
        previous_frame.points = o3d.utility.Vector3dVector(previous_frame_np)

        # Apply ICP to find the transformation between the previous and current frames

        # point2point
        icp_result = o3d.pipelines.registration.registration_icp(
            source=current_frame,
            target=previous_frame,
            max_correspondence_distance=5.0, 
            init=np.eye(4),  
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # point2plane
        # current_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # previous_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # icp_result = o3d.pipelines.registration.registration_icp(
        #     source=current_frame,
        #     target=previous_frame,
        #     max_correspondence_distance=5.0, 
        #     init=np.eye(4),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        # )

        print("Fitness score:", icp_result.fitness)
        print("Inlier RMSE:", icp_result.inlier_rmse)

        # Extract the transformation matrix from the ICP result
        relative_transformation = icp_result.transformation
        # print("Estimated relative transformation using ICP.")

        # x_translation =+ relative_transformation[0, 3]  
        # y_translation =+ relative_transformation[1, 3]  

        # print(f"x translation: {x_translation}")
        # print(f"y translation: {y_translation}")

        return relative_transformation
    
    def get_estimated_poses(self):
        """
        Returns the list of estimated poses.
        Returns:
            List[numpy.ndarray]: A list of 4x4 transformation matrices representing the poses.
        """
        return self.estimated_poses
    
