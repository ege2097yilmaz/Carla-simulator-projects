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

        # Update pose
        pose_increment = np.eye(4)
        pose_increment[:3, :3] = R
        pose_increment[:3, 3] = t.flatten()
        self.pose = self.pose @ pose_increment

        # Update trajectory
        self.trajectory.append(self.pose[:3, 3])

        # Save keypoints and descriptors for the next iteration
        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = frame

        return self.pose

    def get_trajectory(self):
        return np.array(self.trajectory)