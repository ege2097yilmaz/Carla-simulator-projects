import cv2, os
import numpy as np



class VisualOdometry:
    def __init__(self, camera_params):
        self.camera_matrix = camera_params['camera_matrix']
        self.dist_coeffs = camera_params['dist_coeffs']
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.trajectory = []
        self.prev_time = None

        # IMU state
        self.velocity = np.zeros(3)  
        self.translation = np.zeros(3)
        self.orientation = np.eye(3)

        self.pose = np.eye(4)  # Initial pose

    def process_frame(self, frame):
        # Feature detection
        kp, des = self.orb.detectAndCompute(frame, None)

        # Draw and display detected features
        frame_with_keypoints = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Feature Extraction", frame_with_keypoints)
        cv2.waitKey(1)

        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = frame
            return self.pose

        # Feature matching
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Visualize matches
        frame_matches = cv2.drawMatches(
            self.prev_frame, self.prev_kp, frame, kp, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow("Feature Matches", frame_matches)
        cv2.waitKey(1)

        # Extract points
        prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(curr_pts, prev_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

        # Update pose only visual odom
        # pose_increment = np.eye(4)
        # pose_increment[:3, :3] = R
        # pose_increment[:3, 3] = t.flatten()
        # self.pose = self.pose @ pose_increment
        
        ## fuse IMU-VO
        t[1] = 0 # assuming visual odometry cannot detect lateral movement
        self.translation += (self.orientation @ t.flatten())

        # # Update pose
        # self.translation[2] *= -1
        self.pose[:3, :3] = self.orientation  # IMU-based orientation
        self.pose[:3, 3] = self.translation  # Translation

        # Update trajectory
        self.trajectory.append(self.pose[:3, 3])

        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = frame

        return self.pose
    
    def process_imu(self, accel, gyro, timestamp):
        if self.prev_time is None:
            self.prev_time = timestamp
            return

        # Time delta
        dt = timestamp - self.prev_time
        self.prev_time = timestamp

        omega = gyro * dt  
        self.orientation = self.orientation @ self.rotation_matrix_from_gyro(omega)

        accel_world = self.orientation @ accel  
        self.velocity += accel_world * dt
        self.pose[:3, 3] += self.velocity * dt


    def rotation_matrix_from_gyro(self, gyro):
        """
        Compute rotation matrix from gyroscope data using small-angle approximation.
        """
        theta = np.linalg.norm(gyro)
        if theta < 1e-5:  # Avoid division by zero
            return np.eye(3)
        axis = gyro / theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cross_prod_matrix = np.array([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]])
        return cos_theta * np.eye(3) + sin_theta * cross_prod_matrix + (1 - cos_theta) * np.outer(axis, axis)


    def get_trajectory(self):
        return np.array(self.trajectory)