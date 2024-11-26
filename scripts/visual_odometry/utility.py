import numpy as np
import carla, math, os
import matplotlib.pyplot as plt
import pandas as pd


def calculate_distance(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

class gnsProcess:
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Compute the distance between two GPS coordinates using the Haversine formula.

        Args:
            lat1, lon1: Latitude and Longitude of the first point (in degrees).
            lat2, lon2: Latitude and Longitude of the second point (in degrees).

        Returns:
            float: Distance in meters.
        """
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def estimate_scale(self, t, gps_prev, gps_curr):
        """
        Estimate the scale factor for monocular visual odometry using GPS data.

        Args:
            t (np.ndarray): Relative translation vector from visual odometry (3x1).
            gps_prev (tuple): Previous GPS reading (latitude, longitude).
            gps_curr (tuple): Current GPS reading (latitude, longitude).

        Returns:
            float: Scale factor.
            np.ndarray: Scaled translation vector (3x1).
        """
        # Compute GPS distance
        gps_distance = self.haversine_distance(gps_prev[0], gps_prev[1], gps_curr[0], gps_curr[1])

        # Compute relative translation magnitude
        t_magnitude = np.linalg.norm(t)

        # Avoid division by zero
        if t_magnitude == 0:
            raise ValueError("Translation magnitude is zero. Cannot compute scale.")

        # Calculate scale factor
        scale = gps_distance / t_magnitude

        # Scale the translation vector
        t_scaled = scale * t
        return scale, t_scaled
    
class Camera:
    def __init__(self):
        self.fx = 640 / (2 * np.tan(np.deg2rad(90) / 2))
        self.cx = 640 / 2
        self.cy = 480 / 2
        self.width = 640
        self.height = 480
        
def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return carla.Quaternion(qx, qy, qz, qw)

def visualize_trajectory(world, trajectory):
    """
    Visualize the trajectory in CARLA by drawing lines between consecutive positions.
    
    Args:
        world (carla.World): The CARLA world instance.
        trajectory (list): List of carla.Location objects representing the trajectory.
    """
    for i in range(len(trajectory) - 1):
        world.debug.draw_line(
            trajectory[i],
            trajectory[i + 1],
            thickness=0.01,
            color=carla.Color(255, 255, 255),
            life_time=0.0 
        )

def track_trajectory(R, t, K, current_position):
    """
    Update the vehicle's trajectory based on motion estimation.
    
    Args:
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).
        K (np.ndarray): Camera intrinsic matrix.
        current_position (np.ndarray): Current position in world coordinates.
    
    Returns:
        updated_position (carla.Location): Updated position in CARLA world.
    """
    # Update the current position using the translation vector
    new_position = current_position + t
    return carla.Location(
        x=new_position[0, 0],
        y=new_position[1, 0],
        z=new_position[2, 0]
    )


def calculate_repeatability(img1, img2, H, detector, threshold=3):
    """
    Calculate the repeatability of keypoint detection between two images.

    Args:
        img1 (np.ndarray): Reference image (grayscale).
        img2 (np.ndarray): Transformed image (grayscale).
        H (np.ndarray): Homography matrix from img1 to img2.
        detector (cv2.Feature2D): Keypoint detector (e.g., ORB, SIFT).
        threshold (float): Distance threshold for keypoint overlap (in pixels).

    Returns:
        float: Repeatability score.
    """
    # Detect keypoints in both images
    keypoints1 = detector.detect(img1, None)
    keypoints2 = detector.detect(img2, None)

    # Convert keypoints to numpy arrays for processing
    pts1 = np.array([kp.pt for kp in keypoints1])
    pts2 = np.array([kp.pt for kp in keypoints2])

    if len(pts1) == 0 or len(pts2) == 0:
        raise ValueError("No keypoints detected in one or both images.")

    # Transform keypoints from image1 to image2 using the homography
    ones = np.ones((pts1.shape[0], 1))
    pts1_homogeneous = np.hstack([pts1, ones])  # Convert to homogeneous coordinates
    pts1_transformed = (H @ pts1_homogeneous.T).T  # Apply homography
    pts1_transformed /= pts1_transformed[:, 2][:, np.newaxis]  # Normalize

    # Extract x, y coordinates of transformed keypoints
    pts1_transformed = pts1_transformed[:, :2]

    # Find overlapping keypoints
    num_overlapping = 0
    for pt in pts1_transformed:
        distances = np.linalg.norm(pts2 - pt, axis=1)  # Compute distances to all keypoints in img2
        if np.min(distances) < threshold:
            num_overlapping += 1

    # Calculate repeatability
    repeatability = num_overlapping / len(pts1)
    return repeatability

def calculate_velocity(R, t, delta_time, R_wc=None):
    """
    Calculate vehicle velocity from motion estimation.

    Args:
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).
        delta_time (float): Time between frames (seconds).
        R_wc (np.ndarray, optional): Rotation matrix from camera to world frame. Defaults to None.

    Returns:
        np.ndarray: Velocity vector (3x1) in the world frame.
        float: Velocity magnitude (scalar).
    """
    if delta_time <= 0:
        raise ValueError("delta_time must be greater than 0.")

    # If R_wc is provided, transform t to the world frame
    if R_wc is not None:
        t_world = np.dot(R_wc, t)
    else:
        t_world = t  

    # Compute velocity
    velocity = t_world / delta_time
    velocity_magnitude = np.linalg.norm(velocity)

    return velocity, velocity_magnitude

def visualize_velocity_data(csv_file):
    """
    Visualize velocity comparison data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing velocity data.
                        The file should have three columns:
                        [Ground Truth Velocity, Visual Odometry Velocity, Velocity Error].
    """
    # Load data from the CSV file
    data = pd.read_csv(csv_file, header=None, names=['Ground Truth Velocity', 'Visual Odometry Velocity', 'Velocity Error'])

    # Extract columns
    ground_truth_velocity = data['Ground Truth Velocity']
    vo_velocity = data['Visual Odometry Velocity']
    velocity_error = data['Velocity Error']

    # Plotting the velocities
    plt.figure(figsize=(10, 6))
    plt.plot(ground_truth_velocity, label="Ground Truth Velocity", marker='o')
    plt.plot(vo_velocity, label="Visual Odometry Velocity", marker='s')
    plt.title("Velocity Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the velocity error
    plt.figure(figsize=(10, 6))
    plt.plot(velocity_error, label="Velocity Error", color='red', marker='x')
    plt.title("Velocity Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error (m/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Dual-axis plot for advanced visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot ground truth and VO velocity on the first axis
    ax1.plot(ground_truth_velocity, label="Ground Truth Velocity", color='blue', marker='o')
    ax1.plot(vo_velocity, label="Visual Odometry Velocity", color='green', marker='s')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create a second y-axis for velocity error
    ax2 = ax1.twinx()
    ax2.plot(velocity_error, label="Velocity Error", color='red', linestyle='--')
    ax2.set_ylabel("Error (m/s)")
    ax2.legend(loc="upper right")

    plt.title("Velocity and Error Comparison")
    plt.show()

def plot_trajectory(file_path="trajectory.txt", plane="XZ"):
    """
    Load and plot the trajectory from a file.

    Args:
        file_path (str): Path to the trajectory file. Default is "trajectory.txt".
        plane (str): Plane to plot. Options are "XZ", "XY", or "YZ". Default is "XZ".
    """
    # Load trajectory from the file
    trajectory = np.loadtxt(file_path)

    # Separate X, Y, and Z coordinates
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Select the plane to plot
    if plane.upper() == "XZ":
        x_data, y_data = x, z
        xlabel, ylabel = "X (meters)", "Z (meters)"
    elif plane.upper() == "XY":
        x_data, y_data = x, y
        xlabel, ylabel = "X (meters)", "Y (meters)"
    elif plane.upper() == "YZ":
        x_data, y_data = y, z
        xlabel, ylabel = "Y (meters)", "Z (meters)"
    else:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'XZ', 'XY', or 'YZ'.")

    # Plot the trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(x_data, y_data, label=f"Trajectory ({plane.upper()} plane)", linewidth=2)
    plt.title("Vehicle Trajectory")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()