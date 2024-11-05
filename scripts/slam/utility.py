import open3d as o3d
import os, carla
import matplotlib.pyplot as plt

def save_point_cloud_data(point_cloud_data, file_index, output_directory="point_cloud_data"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pcd = o3d.geometry.PointCloud()
    

    new_pcd = point_cloud_data.get_open3d_point_cloud(-1.0)
    if new_pcd is not None:
        pcd.points = new_pcd.points

        file_name = f"{output_directory}/point_cloud_{file_index}.pcd"
        o3d.io.write_point_cloud(file_name, pcd)
        print(f"Saved {file_name}")
        file_index += 1


def save_point_cloud_data2(point_cloud_data, start_time, current_time, file_index, output_directory="point_cloud_data"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pcd = o3d.geometry.PointCloud()
    

    new_pcd = point_cloud_data.get_open3d_point_cloud(0.1)
    if new_pcd is not None:
        pcd.points = new_pcd.points

        # print("//////////////////debuing")
        # print(current_time)
        # print(start_time)
        # print(abs(current_time - start_time))

        if abs(current_time - start_time) >= 1:
            file_name = f"{output_directory}/point_cloud_{file_index}.pcd"
            o3d.io.write_point_cloud(file_name, pcd)
            print(f"Saved {file_name}")
            start_time = current_time  # Reset the timer
            file_index += 1


def visualize_keypoints(scan, keypoints):
    """
    Visualizes the 3D point cloud and the extracted keypoints using Matplotlib.
    
    Args:
        scan (np.ndarray): Original point cloud data of shape (N, 3).
        keypoints (np.ndarray): Extracted keypoints of shape (M, 3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(f"Number of points in scan: {scan.shape[0]}")
    print(f"Number of keypoints: {keypoints.shape[0]}")

    # Plot the original point cloud
    ax.scatter(scan[:, 0], scan[:, 1], scan[:, 2], c='blue', s=1.5, label='Original Points')

    # Plot the keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='red', s=10, label='Keypoints')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def visualize_keypoints_in_carla(client, keypoints, life_time=1.0):
    """
    Visualizes keypoints in CARLA using the DebugHelper.
    
    Args:
        client (carla.Client): CARLA client instance.
        keypoints (np.ndarray): Extracted keypoints as a numpy array of shape (N, 3).
        life_time (float): How long the keypoints should be visible in seconds.
    """
    world = client.get_world()
    debug = world.debug

    for point in keypoints:
        location = carla.Location(x=point[0], y=point[1], z=point[2])
        debug.draw_point(location, size=0.05, color=carla.Color(0, 0, 255), life_time=life_time)

def visualize_pose_graph_in_carla(pose_graph, carla_world, initial_x=0.0, initial_y=0.0, initial_z=0.0):
    """
    Visualizes the pose graph in the CARLA simulator with an initial offset.
    
    Args:
        pose_graph (networkx.DiGraph): The pose graph containing the poses.
        carla_world (carla.World): The CARLA world instance.
        initial_x (float): The initial x-coordinate offset.
        initial_y (float): The initial y-coordinate offset.
        initial_z (float): The initial z-coordinate offset.
    """
    # Iterate over the nodes and draw them as points
    for node_index in pose_graph.nodes:
        pose = pose_graph.nodes[node_index]['pose']
        
        # Extract the (x, y, z) translation from the 4x4 transformation matrix
        x, y, z = pose[0, 3] + initial_x, pose[1, 3] + initial_y, pose[2, 3] + initial_z
        
        # Draw a point at this location in CARLA
        carla_world.debug.draw_point(
            carla.Location(x=x, y=y, z=z),
            size=0.1,  
            color=carla.Color(255, 255, 255), 
            life_time=5.0 
        )

    # Iterate over the edges and draw lines between connected nodes
    for edge in pose_graph.edges:
        start_index, end_index = edge
        start_pose = pose_graph.nodes[start_index]['pose']
        end_pose = pose_graph.nodes[end_index]['pose']

        # Extract the (x, y, z) translations from the 4x4 transformation matrices
        start_x, start_y, start_z = start_pose[0, 3] + initial_x, start_pose[1, 3] + initial_y, start_pose[2, 3] + initial_z
        end_x, end_y, end_z = end_pose[0, 3] + initial_x, end_pose[1, 3] + initial_y, end_pose[2, 3] + initial_z

        # Draw a line between the two points in CARLA
        carla_world.debug.draw_line(
            carla.Location(x=start_x, y=start_y, z=start_z),
            carla.Location(x=end_x, y=end_y, z=end_z),
            thickness=0.05,  
            color=carla.Color(0, 255, 0),  
            life_time=5.0 
        )