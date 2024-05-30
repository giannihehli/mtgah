# IMPORT PACKAGES AND MODULES
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from scipy import stats

# IMPORT USER-DEFINED MODULES
from detectMarkers import detect

def rasterize(cloud, raster_size):
    # Compute the number of bins for the x and y axes
    x_bins = np.arange(0, 6000, raster_size)
    y_bins = np.arange(0, 6000, raster_size)

    # Compute the maximum z-coordinate in each bin
    max_z, x_edges, y_edges, binnumber = stats.binned_statistic_2d(
        cloud.points['x'], 
        cloud.points['y'], 
        cloud.points['z'], 
        statistic='max', 
        bins=[x_bins, y_bins])
    
    print(max_z.shape)
    print(max_z[3000][3000])
    print(x_edges)
    print(y_edges)

    return max_z

def transform(cloud, M):
    # Create an empty DataFrame with the same columns as the original point cloud's points
    df_new = pd.DataFrame(columns=cloud.points.columns)

    # Apply the transformation matrix to the point cloud
    x, y, z, _ = np.dot(M, np.array([cloud.points['x'], cloud.points['y'], cloud.points['z'], np.ones(cloud.points.shape[0])]))

    # Mirror the y-coordinates
    y = 600 - y

    # Inverse the z-coordinates
    z = -z

    # Fill the new point cloud with the transformed coordinates in higher resolution (0.1mm)
    df_new['x'] = 10*x
    df_new['y'] = 10*y
    df_new['z'] = 10*z
    df_new['red'] = cloud.points['red']
    df_new['green'] = cloud.points['green']
    df_new['blue'] = cloud.points['blue']

    # Create a new point cloud with the new DataFrame
    cloud_corr = PyntCloud(df_new)

    return cloud_corr

def getimage(cloud):

    # Translate the point cloud so all 'x' and 'y' values are positive
    min_x = cloud.points['x'].min()
    min_y = cloud.points['y'].min()
    cloud.points['x'] -= min_x
    cloud.points['y'] -= min_y

    # Create an empty list to store the indices - note that x and y axis are flipped to get the
    # correct image
    indices = np.empty((int(cloud.points['y'].max()) + 1, int(cloud.points['x'].max()) + 1), dtype=object)

    # Now create the image and store the indices - note that x and y axis are flipped to get the
    # correct image
    image = np.zeros((int(cloud.points['y'].max()) + 1, int(cloud.points['x'].max()) + 1, 3), dtype=np.uint8)

    # Loop through the points, save the color values and store the indices
    for index, point in cloud.points.iterrows():
        img_x, img_y = int(point['x']), int(point['y'])
        image[img_y, img_x] = [point['blue'], point['green'], point['red']]
        indices[img_y, img_x] = index
 
    """ # Show the image
    cv2.imshow('Point cloud', cv2.resize(image, (1080, 1080)))
    cv2.waitKey(0)  """

    # Save the image
    cv2.imwrite('H:/data/cloudcompare/test/imagefrompointcloud.jpg', image)

    return image, indices

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


if __name__ == "__main__":

    # Define pattern as 3D coordinates in 0.1mm
    pattern = 100 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], [9.12, 8.92, 0], [1.05, 8.92, 0],
                            [50.7, 0.91, 0], [58.77, 0.94, 0], [58.75, 9.04, 0], [50.66, 9, 0],
                            [50.57, 51.09, 0], [58.63, 50.99, 0], [58.73, 59.07, 0], [50.66, 59.14, 0],
                            [1.08, 51.02, 0], [9.15, 51.02, 0], [9.12, 59.13, 0], [1.04, 59.11, 0]])

    # Load the .ply file
    cloud = PyntCloud.from_file('H:/data/cloudcompare/test/sand_minus.ply')

    # Get the image and indices
    image, indices = getimage(cloud)

    # Detect ArUco markers
    marker = 'DICT_4X4_50'
    ptc_det, ptc_corners, ptc_ids = detect(image, marker)

    """ # Show the image with detected markers
    cv2.imshow('image', ptc_det)
    cv2.waitKey(0) """

    # Save the image with detected markers
    cv2.imwrite('H:/data/cloudcompare/test/imagefrompointcloud_det.jpg', ptc_det)

    # Define corners_3d array
    corners_3d = np.hstack((ptc_corners, np.zeros((ptc_corners.shape[0], 1), dtype=ptc_corners.dtype)))

    # Calculate the transformation matrix from the detected corners and the measured pattern
    M = calculate_transformation(corners_3d, pattern)

    # Transform the pointcloud to correct origin
    cloud_corr = transform(cloud, M)

    """ # Plot the corrected point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=0, azim=0, roll=0) # Set the view to y-z plane
    ax.view_init(elev=0, azim=-90, roll=0) # Set the view to x-z plane
    #ax.view_init(elev=90, azim=-90, roll=0) # Set the view to x-y plane
    ax.scatter(
        cloud_corr.points['x'], 
        cloud_corr.points['y'], 
        cloud_corr.points['z'], 
        c=cloud_corr.points[['red', 'green', 'blue']] / 255.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show() """

    # Define raster size in 0.1mm
    raster_size = 1

    # Rasterise the corrected point cloud
    max_z = rasterize(cloud_corr, raster_size)
