# IMPORT PACKAGES AND MODULES
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation as R
from scipy import stats

# IMPORT USER-DEFINED MODULES
from detectMarkers import detect

def getimage(cloud):

    # Translate the point cloud so all 'x' and 'y' values are positive
    min_x = cloud.points['x'].min()
    min_y = cloud.points['y'].min()
    cloud.points['x'] -= min_x
    cloud.points['y'] -= min_y

    # Create an empty list to store the indices - note that x and y axis are flipped 
    # to get the correct image
    indices = np.empty((int(cloud.points['y'].max()) + 1, 
                        int(cloud.points['x'].max()) + 1), 
                        dtype=object)

    # Now create the image and store the indices - note that x and y axis are flipped 
    # to get the correct image
    image = np.zeros((int(cloud.points['y'].max()) + 1, 
                      int(cloud.points['x'].max()) + 1, 
                      3), dtype=np.uint8)

    # Loop through the points, save the color values and store the indices
    for index, point in cloud.points.iterrows():
        img_x, img_y = int(point['x']), int(point['y'])
        image[img_y, img_x] = [point['blue'], point['green'], point['red']]
        indices[img_y, img_x] = index

    return image, indices


def sortpoints(ptc_corners, ptc_ids, pattern, aruco_count):
    # Throw out the aruco ids that are above the aruco count
    ptc_ids = ptc_ids[ptc_ids < aruco_count]

    # Define source and target points structure
    ptc_corners_det = np.ones((len(ptc_ids)*4, 2))
    target_ptc = np.ones((len(ptc_ids)*4, 3))

    # Sort detected corners and target points according to detected marker IDs
    for i, id in enumerate(ptc_ids):
        ptc_corners_det[i*4:i*4+4] = ptc_corners[i*4:i*4+4]
        target_ptc[i*4:i*4+4] = 0.1 * pattern[id*4:id*4+4]

    # Define corners in 3D as source points
    source_ptc = np.hstack((ptc_corners_det, np.zeros((len(ptc_ids) * 4, 1), 
                                                      dtype=ptc_corners.dtype)))

    return source_ptc, target_ptc

def calculate_transformation(points_3d_source, points_3d_target):
    # Calculate centroids
    centroid_source = np.mean(points_3d_source, axis=0)
    centroid_target = np.mean(points_3d_target, axis=0)

    # Center the points
    centered_source = points_3d_source - centroid_source
    centered_target = points_3d_target - centroid_target

    # Compute the cross-covariance matrix
    H = np.dot(centered_source.T, centered_target)

    # Compute the singular value decomposition of H
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Correct for reflection case
    if np.linalg.det(R) < 0:
       Vt[-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = centroid_target - np.dot(centroid_source, R.T)

    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix

def transform(cloud, M, basis_length, R, t):
    # Create an empty DataFrame with the same columns as the original point cloud's points
    df_new = pd.DataFrame(columns=cloud.points.columns)

    # Apply the transformation matrix to the point cloud
    x_rast, y, z, _ = np.dot(M, np.array([cloud.points['x'], cloud.points['y'], 
                                          cloud.points['z'], np.ones(cloud.points.shape[0])]))

    # Mirror the y-coordinates
    y_rast = 1000*basis_length - y

    # Inverse the z-coordinates
    z_rast = -z

    # Rotate the pointcloud to the wanted world frame
    x_world, y_world, z_world = np.dot(R, np.array([x_rast, y_rast, z_rast]))

    # Translate the pointcloud to the wanted world frame [mm]
    x_world += 1000*t[0]
    y_world += 1000*t[1]
    z_world += 1000*t[2]

    # Fill the new point cloud with the transformed coordinates in higher resolution (0.1mm) -
    # except z-values stay in [mm]
    df_new['x'] = 10*x_world
    df_new['y'] = 10*y_world
    df_new['z'] = z_world
    df_new['red'] = cloud.points['red']
    df_new['green'] = cloud.points['green']
    df_new['blue'] = cloud.points['blue']

    # Create a new point cloud with the new DataFrame
    cloud_corr = PyntCloud(df_new)

    return cloud_corr

def rasterize(cloud, raster_size, x_min, x_max, y_min, y_max):
    # Compute the number of bins for the x and y axes in [0.1 mm]
    x_bins = np.arange(10000*x_min, 10000*x_max, 10000*raster_size)
    y_bins = np.arange(10000*y_min, 10000*y_max, 10000*raster_size)

    # Compute the maximum z-coordinate in each bin
    max_z, x_edges, y_edges, _ = stats.binned_statistic_2d(cloud.points['x'], 
                                                           cloud.points['y'], 
                                                           cloud.points['z'], 
                                                           statistic='max', 
                                                           bins=[x_bins, y_bins])

    return max_z, x_edges, y_edges
    
def convertimage(img_thr, img_min_x, img_max_x, img_min_y, img_max_y):
    # Change the indeces from [m] to pixel coordinates [0.1mm]
    img_min_x = int(10000*img_min_x)
    img_max_x = int(10000*img_max_x)
    img_min_y = int(10000*img_min_y)
    img_max_y = int(10000*img_max_y)

    # Rotate the 2D array 90 degrees clockwise
    rotated_img_thr = np.rot90(img_thr, -1)

    # Get the needed part of the image
    cut_img_thr = rotated_img_thr[img_min_x:img_max_x, img_min_y:img_max_y]

    # Define the z-values of the image and set the background to NaN
    img_z = np.where(cut_img_thr == 0, np.nan, 0)

    # Define x and y edges of raster by adjustng to raster size
    img_x = np.linspace(img_min_x, img_max_x-1, num = img_max_x - img_min_x)
    img_y = np.linspace(img_min_y, img_max_y-1, num = img_max_y - img_min_y)

    return img_z, img_x, img_y

def export(max_z, x_edges, y_edges, raster_size, output_path):
    # Define the header of the output data
    header = f'# Units: 0.1mm\n'
    header = f'ncols {max_z.shape[0]}\n'
    header += f'nrows {max_z.shape[1]}\n'
    header += f'xllcorner {min(x_edges/10000)}\n'
    header += f'yllcorner {min(y_edges/10000)}\n'
    header += f'cellsize {raster_size}\n'
    header += f'NODATA_value -9999\n'

    # Rotate the 2D array 90 degrees counterclockwise to match the orientation after exporting
    rotated_max_z = np.rot90(max_z, 1)

    # Convert all z-values to [m]
    meters_max_z = rotated_max_z / 10000

    # Flatten the rotated 2D array and replace NaN values with the NODATA value
    flat_max_z = np.where(np.isnan(meters_max_z), -9999, rotated_max_z).flatten()

    # Write the header and the flattened array to the ASC file
    with open(output_path, 'w') as f:
        f.write(header)
        np.savetxt(f, flat_max_z, fmt='%1.4f')

    return


if __name__ == '__main__':
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define the path to the data
    data_path = 'data/'

    # Define experiment to be aligned
    exp = 'f_r2-pa_d41_h148_16'

    # Define positions of the Aruco codes as 3D coordinates in 0.1mm in clockwise order
    pattern = 100 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], # Code ID 0
                              [9.12, 8.92, 0], [1.05, 8.92, 0], # clockwise from left top
                              [50.7, 0.91, 0], [58.77, 0.94, 0], # Code ID 1
                              [58.75, 9.04, 0], [50.66, 9, 0], # clockwise from left top
                              [50.57, 51.09, 0], [58.63, 50.99, 0], # Code ID 2
                              [58.73, 59.07, 0], [50.66, 59.14, 0], # clockwise from left top
                              [1.08, 51.02, 0], [9.15, 51.02, 0], # Code ID 3
                              [9.12, 59.13, 0], [1.04, 59.11, 0]]) # clockwise from left top
   
    #######################################
    # SCANNER OPTIONS
    
    # Define how many aruco codes are used on the scanned area
    aruco_count = 4 # [] Number of aruco codes used on the scanned area

    # Define the length of the basis in [m] as the y-distance between the image and raster
    # coordinate frame
    basis_length = 0.6 # [m] Length of the basis in y direction

    # Define raster size in [m] for rasterization of scanned pointcloud
    raster_size = 0.001 # [m] Size of one bin in the raster in x and y directionn
    
    # Define angle of inclined base plane in [deg]
    angle = 0 # [deg] Angle of the inclined base plane - if codes are on runout set to 0

    # Define the location of the raster frame in the wanted world frame in [m]
    # This moves the pointcloud to the wanted world frame
    # If everything is set to zero, the raster frame and world frame align
    trans_x = 0.0 # [m] Translation of the wanted world frame in x direction
    trans_y = 0.0 # [m] Translation of the wanted world frame in y direction
    trans_z = 0.0 # [m] Translation of the wanted world frame in z direction

    # Define raster min and max values in [m] for x and y direction in world frame
    # This defines what part of the pointcloud in the world frame is rasterized and exported
    world_min_x = 0.1 # [m] Minimum value of the raster in x direction
    world_max_x = 0.5 # [m] Maximum value of the raster in x direction
    world_min_y = 0.1 # [m] Minimum value of the raster in y direction
    world_max_y = 0.5 # [m] Maximum value of the raster in y direction

    ####################################################################################
    
    # Load the .ply file
    cloud = PyntCloud.from_file(f'{data_path}{exp}.ply')

    # Get the image and indices
    image, indices = getimage(cloud)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    # Try making the directory for the pointcloud images
    try:
        os.mkdir(f'{data_path}ptc_images')
        print(f'Directory scanner/images created. All pointcloud images saved there.')
    except FileExistsError:
        pass
    
    # Save the image
    cv2.imwrite(f'{data_path}ptc_images/{exp}_ptc.png', image)

    # Detect ArUco markers
    marker = 'DICT_4X4_50'
    ptc_det, ptc_corners, ptc_ids = detect(image, marker)

    # Show the image with detected markers
    cv2.imshow('image detected', ptc_det)
    cv2.waitKey(0)
    
    # Save the image with detected markers
    cv2.imwrite(f'{data_path}ptc_images/{exp}_det.png', ptc_det)

    # Sort the detected corners and target points
    source, target = sortpoints(ptc_corners, ptc_ids, pattern, aruco_count)    

    # Calculate the transformation matrix from the detected corners and the measured pattern
    M = calculate_transformation(source, target)

    # Define the rotation angle in radians
    theta = np.radians(angle)

    # Define the additional rotation matrix for the inclined base plane
    R = np.array([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])

    # Define the translation vector for the wanted world frame
    t = np.array([trans_x, trans_y, trans_z])

    # Transform the pointcloud to correct origin
    cloud_corr = transform(cloud, M, basis_length, R, t)
    
    # Plot the initial point cloud
    fig = plt.figure(num='Initial Point Cloud')
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=0, azim=0, roll=0) # Set the view to y-z plane
    #ax.view_init(elev=0, azim=-90, roll=0) # Set the view to x-z plane
    ax.view_init(elev=90, azim=-90, roll=0) # Set the view to x-y plane
    ax.scatter(
        cloud.points['x'], 
        cloud.points['y'], 
        cloud.points['z'], 
        c=cloud.points[['red', 'green', 'blue']] / 255.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    # Plot the corrected point cloud
    fig = plt.figure(num='Corrected Point Cloud')
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=0, azim=0, roll=0) # Set the view to y-z plane
    #ax.view_init(elev=0, azim=-90, roll=0) # Set the view to x-z plane
    ax.view_init(elev=90, azim=-90, roll=0) # Set the view to x-y plane
    ax.scatter(
        cloud_corr.points['x'], 
        cloud_corr.points['y'], 
        cloud_corr.points['z'], 
        c=cloud_corr.points[['red', 'green', 'blue']] / 255.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # Rasterise the corrected point cloud
    max_z, x_edges, y_edges = rasterize(cloud_corr, raster_size, 
                                        world_min_x, world_max_x, 
                                        world_min_y, world_max_y)
    # Define the output path
    output_path = f'output/{exp}_raster.asc'

    # Export data as ascii file
    export(max_z, x_edges, y_edges, raster_size, output_path)