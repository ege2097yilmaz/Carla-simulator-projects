import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import networkx as nx
from utility import *
import gtsam

class SLAM:
    def __init__(self, optimization_interval=10,  pcd_filename="real_time_map.pcd"):
        # Initialize key variables for storing keypoints and LiDAR scans
        self.keypoints = []
        self.lidar_scans = []
        self.estimated_poses = [] 
        self.cumulative_transformation = np.eye(4)
        self.pose_graph = nx.DiGraph()  
        self.current_pose = np.eye(4) 
        self.node_index = 0
        self.vis_keypoints = []

        self.cumulative_x_translation = 0 
        self.cumulative_y_translation = 0

        self.loop_closure_distance_threshold = 5.0  # Threshold in meters for loop closure

        self.optimization_interval = optimization_interval  # Frames between optimizations
        self.map_point_cloud = o3d.geometry.PointCloud()
        self.pcd_filename = pcd_filename

    
    def get_keypoints(self, vehicle_x, vehicle_y, vehicle_yaw):
        yaw_rad = np.radians(vehicle_yaw + 360)
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])

        for point in self.vis_keypoints:
            # Apply the rotation matrix to the (x, y) coordinates
            rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1]]))
            
            # Update the point with the rotated and translated coordinates
            point[0] = rotated_point[0] + vehicle_x
            point[1] = rotated_point[1] + vehicle_y

        return self.vis_keypoints
    
    
    def get_estimated_poses(self):
        """
        Returns the list of estimated poses.
        Returns:
            List[numpy.ndarray]: A list of 4x4 transformation matrices representing the poses.
        """
        return self.estimated_poses
    
    
    def get_pose_graph(self):
        return self.pose_graph
    
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
            keypoints = self._extract_corner_and_plane_features(scan, 1e-15, 0.2) # deprecated function 1e-15, 0.9
            # keypoints = self._extract_features(scan)
            self.vis_keypoints = keypoints
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

        new_pose[2, 3] = self.current_pose[2, 3]  # Keep the z position unchanged

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

        # if(keypoints is not None):
        #     visualize_keypoints(points_np, keypoints) # to visualize scans and keypoints

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

            # Perform loop closure detection
            self.detect_and_add_loop_closure(points_frame)

            self.incremental_map_update(points_np, relative_transformation)

            if self.node_index % self.optimization_interval == 0:
                self.optimize_graph()
                self.update_map_with_optimized_poses()
                self.save_map_to_pcd()

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

        #! point2point
        icp_result = o3d.pipelines.registration.registration_icp(
            source=current_frame,
            target=previous_frame,
            max_correspondence_distance=5.0, 
            init=np.eye(4),  
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        #! point2plane
        # current_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # previous_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # icp_result = o3d.pipelines.registration.registration_icp(
        #     source=current_frame,
        #     target=previous_frame,
        #     max_correspondence_distance=5.0, 
        #     init=np.eye(4),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        # )

        # print("Fitness score:", icp_result.fitness)
        # print("Inlier RMSE:", icp_result.inlier_rmse)

        # Extract the transformation matrix from the ICP result
        relative_transformation = icp_result.transformation
        # print("Estimated relative transformation using ICP.")

        # x_translation =+ relative_transformation[0, 3]  
        # y_translation =+ relative_transformation[1, 3]  

        # print(f"x translation: {x_translation}")
        # print(f"y translation: {y_translation}")

        return relative_transformation
    
    """ TODO
    include place recognition algorithms like Scan Context 
    or using geometric consistency checks with previously visited areas
    """
    def detect_and_add_loop_closure(self, current_frame):
        """
        Detects loop closure by comparing the current scan to earlier scans and adds an edge if a match is found.
        Args:
            current_frame (o3d.geometry.PointCloud): The current point cloud frame.
        """
        current_position = self.current_pose[:3, 3]  # Extract the translation component of the current pose

        for i in range(self.node_index - 10):  # Skip the most recent scans to avoid redundant matching
            previous_pose = self.pose_graph.nodes[i]['pose']
            previous_position = previous_pose[:3, 3]  

            # Compute the Euclidean distance between the current and previous positions
            distance = np.linalg.norm(current_position - previous_position)
            if distance > self.loop_closure_distance_threshold:
                continue  # Skip if the distance is greater than the threshold

            previous_frame_np = self.lidar_scans[i]
            previous_frame = o3d.geometry.PointCloud()
            previous_frame.points = o3d.utility.Vector3dVector(previous_frame_np)

            # Use ICP to check for a match
            icp_result = o3d.pipelines.registration.registration_icp(
                source=current_frame,
                target=previous_frame,
                max_correspondence_distance=10.0,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # If ICP fitness is high enough, we have a loop closure
            if icp_result.fitness > 0.9 and self._validate_loop_closure(icp_result.transformation):  
                loop_closure_transformation = icp_result.transformation
                self.pose_graph.add_edge(self.node_index, i, transformation=loop_closure_transformation)
                print(f"Loop closure detected and edge added between node {self.node_index} and node {i}.")
    
    def optimize_graph(self):
        """
        Optimizes the pose graph to minimize the overall error from all constraints (edges) using GTSAM,
        without considering the z-direction.
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # Add all nodes (poses) to the graph
        for node_index in self.pose_graph.nodes:
            pose = self.pose_graph.nodes[node_index]['pose']
            translation = np.copy(pose[:3, 3])  
            translation[2] = 0 
            rotation = pose[:3, :3]
            gtsam_pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation[0], translation[1], 0))
            initial_estimate.insert(node_index, gtsam_pose)

            # Add a prior factor to the first pose to anchor the graph
            if node_index == 0:
                prior_model = gtsam.noiseModel.Diagonal.Variances(np.ones(6) * 0.01)
                graph.add(gtsam.PriorFactorPose3(node_index, gtsam_pose, prior_model))

        # Add all edges (relative transformations) to the graph
        noise_model = gtsam.noiseModel.Diagonal.Variances(np.ones(6))  # Adjust noise model as needed
        for edge in self.pose_graph.edges:
            source, target = edge
            relative_transformation = self.pose_graph.edges[edge]['transformation']
            translation = np.copy(relative_transformation[:3, 3])  
            translation[2] = 0
            rotation = relative_transformation[:3, :3]
            gtsam_relative_pose = gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(translation[0], translation[1], 0))
            graph.add(gtsam.BetweenFactorPose3(source, target, gtsam_relative_pose, noise_model))

        # Optimize the graph
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        result = optimizer.optimize()

        # Update the poses in the graph with the optimized values
        for node_index in self.pose_graph.nodes:
            optimized_pose = result.atPose3(node_index)
            self.pose_graph.nodes[node_index]['pose'] = optimized_pose.matrix()

        initial_error = graph.error(initial_estimate)
        final_error = graph.error(result)

        print("Initial Error:", initial_error)
        print("Final Error:", final_error)

        print("Graph optimization complete.")

    def construct_map(self):
        """
        Constructs a map by transforming and merging all LiDAR scans using the optimized poses.
        Returns:
            o3d.geometry.PointCloud: The constructed 3D map as an Open3D point cloud.
        """
        map_point_cloud = o3d.geometry.PointCloud()

        for node_index in self.pose_graph.nodes:
            # Get the optimized pose for this node
            pose_matrix = self.pose_graph.nodes[node_index]['pose']

            # Transform the corresponding LiDAR scan to the global coordinate frame
            scan_points = self.lidar_scans[node_index]
            scan_point_cloud = o3d.geometry.PointCloud()
            scan_point_cloud.points = o3d.utility.Vector3dVector(scan_points)

            # Apply the transformation to the scan points
            scan_point_cloud.transform(pose_matrix)

            # Merge the transformed scan into the map
            map_point_cloud += scan_point_cloud

        # Optionally, you can down-sample the map to reduce the number of points
        map_point_cloud = map_point_cloud.voxel_down_sample(voxel_size=0.1)

        print("Map construction complete.")
        return map_point_cloud

    def incremental_map_update(self, points_np, transformation):
        current_scan = o3d.geometry.PointCloud()
        current_scan.points = o3d.utility.Vector3dVector(points_np)
        current_scan.transform(np.dot(self.current_pose, transformation))  # Apply global transformation
        self.map_point_cloud += current_scan  # Add to the incremental map

    def update_map_with_optimized_poses(self):
        self.map_point_cloud.clear()
        for node_index in self.pose_graph.nodes:
            pose_matrix = self.pose_graph.nodes[node_index]['pose']
            scan_points = self.lidar_scans[node_index]
            scan_point_cloud = o3d.geometry.PointCloud()
            scan_point_cloud.points = o3d.utility.Vector3dVector(scan_points)
            scan_point_cloud.transform(pose_matrix)
            self.map_point_cloud += scan_point_cloud
        self.map_point_cloud = self.map_point_cloud.voxel_down_sample(voxel_size=0.1)
        print("Map updated with optimized poses.")

    def save_map_to_pcd(self):
        """
        Saves the current map to a PCD file, overwriting it each time.
        """
        o3d.io.write_point_cloud(self.pcd_filename, self.map_point_cloud)
        print(f"Map saved to {self.pcd_filename}.")

    def _validate_loop_closure(self, transformation):
        """
        Validates the loop closure by checking the geometric consistency of the transformation.
        Args:
            transformation (np.ndarray): The transformation matrix for the loop closure.
        Returns:
            bool: True if the loop closure is valid, False otherwise.
        """
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]

        # Check rotation consistency
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            print("Invalid loop closure: Rotation matrix is not valid.")
            return False

        # Check translation magnitude
        if np.linalg.norm(translation) > self.loop_closure_distance_threshold:
            print("Invalid loop closure: Translation magnitude is too large.")
            return False

        return True