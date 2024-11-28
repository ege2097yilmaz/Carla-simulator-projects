import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
data = pd.read_csv('trajectory_comparison.csv')

# Extract columns and convert them to numpy arrays
time = data['time'].values
gt_x = data['gt_x'].values
gt_y = data['gt_y'].values
ekf_x = data['ekf_x'].values
ekf_y = data['ekf_y'].values

# Plot the ground truth trajectory
plt.figure()
plt.plot(gt_x, gt_y, label='Ground Truth', linestyle='-', marker='o')
plt.plot(ekf_x, ekf_y, label='EKF Estimate', linestyle='--', marker='x')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid()
plt.show()
