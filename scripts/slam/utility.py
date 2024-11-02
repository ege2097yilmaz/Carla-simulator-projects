import open3d as o3d
import os

def save_point_cloud_data(point_cloud_data, file_index, output_directory="point_cloud_data"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pcd = o3d.geometry.PointCloud()
    

    new_pcd = point_cloud_data.get_open3d_point_cloud()
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
    

    new_pcd = point_cloud_data.get_open3d_point_cloud()
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

