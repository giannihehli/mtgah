# IMPORT PACKAGES AND MODULES
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


# IMPORT USER-DEFINED MODULES
from detectMarkers import detect


def calculate_transformation_matrix(points_3d_source, points_3d_target):
    # Calculate centroids
    centroid_source = np.mean(points_3d_source, axis=0)
    centroid_target = np.mean(points_3d_target, axis=0)

    # Center the points
    centered_source = points_3d_source - centroid_source
    centered_target = points_3d_target - centroid_target

    # Compute the matrix H
    H = np.dot(centered_source.T, centered_target)

    # Perform singular value decomposition on H
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Ensure the rotation matrix is right-handed
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation_vector = centroid_target - np.dot(centroid_source, rotation_matrix)

    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix



# Define pattern as 3D coordinates in mm
pattern = 10 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], [9.12, 8.92, 0], [1.05, 8.92, 0],
                          [50.7, 0.91, 0], [58.77, 0.94, 0], [58.75, 9.04, 0], [50.66, 9, 0],
                          [50.57, 51.09, 0], [58.63, 50.99, 0], [58.73, 59.07, 0], [50.66, 59.14, 0],
                          [1.08, 51.02, 0], [9.15, 51.02, 0], [9.12, 59.13, 0], [1.04, 59.11, 0]])

# Load the .ply file
cloud = PyntCloud.from_file('H:/data/cloudcompare/test/sand_plus.ply')

# Translate the point cloud so all 'x' and 'y' values are positive
min_x = cloud.points['x'].min()
max_y = cloud.points['y'].max()
cloud.points['x'] -= min_x
cloud.points['y'] -= max_y

print('Point cloud size: ', cloud.points.shape)
print(f'Min x: {min_x}, Max y: {max_y}')
print(f'Max x: {cloud.points["x"].max()}, Min y: {-cloud.points["y"].min()}')

# Create an empty list to store the indices
indices = np.empty((int(-cloud.points['y'].min()) + 1, int(cloud.points['x'].max()) + 1), dtype=object)

# Now create the image and store the indices
image = np.zeros((int(-cloud.points['y'].min()) + 1, int(cloud.points['x'].max()) + 1, 3), dtype=np.uint8)
for index, point in cloud.points.iterrows():
    x_img, y_img = int(point['x']), -int(point['y'])
    image[y_img, x_img] = [point['blue'], point['green'], point['red']]
    indices[y_img, x_img] = index

print('Image shape: ', image.shape)

cv2.imshow('Point cloud', cv2.resize(image, (1080, 1080)))
cv2.waitKey(0) 

# Detect ArUco markers
marker = "DICT_4X4_50"
img_det, corners, ids = detect(image, marker)

cv2.imshow("image", img_det)
cv2.waitKey(0)

# Define source array
src = np.empty([0, 3], dtype=np.float32)

# Define distance array
distances_min = np.empty([16], dtype=np.float32)

# Get the 3D coordinates of the source points by calculating the shortes distance to the point cloud
for i in range(len(corners)):
    # set the corner as x, y coordinates
    corner_x, corner_y = corners[i][0], corners[i][1]

    # Calculate the Euclidean distance between (x, corner_y) and each point in the point cloud
    distances = cloud.points.apply(lambda point: distance.euclidean((corner_x, corner_y), (point['x'], point['y'])), axis=1)
    
    # Get the index of the closest point to the detected corner
    closest_point_index = distances.idxmin()

    # Get the closest point from the point cloud
    closest_point = cloud.points.iloc[closest_point_index]

    # Append the closest point to the source array
    src = np.append(src, [[closest_point['x'], closest_point['y'], closest_point['z']]], axis=0)

    # Save the minimum distance
    distances_min[i] = distances.min()

print(f'Distance: {distances_min}')
print(f'Mean of all errors: {distances_min.mean()}')

print(f'Source shape: {src.shape}')
print(f'Pattern shape: {pattern.shape}')

print(f'Source: {src}')
print(f'Pattern: {pattern}')

# Initialize the transformation matrix to the identity matrix
M = np.eye(4)

# Define the maximum number of iterations and the convergence threshold
max_iterations = 100
convergence_threshold = 1e-6

# Create a nearest neighbors object for the target points
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pattern)

for i in range(max_iterations):
    # Apply the current transformation to the source points
    src_transformed = np.dot(M, np.vstack((src.T, np.ones(src.shape[0]))))[:3].T

    # Find the closest points in the target point cloud
    distances, indices = nbrs.kneighbors(src_transformed)

    # Calculate the new transformation that best aligns the transformed source points to the closest target points
    M_new = calculate_transformation_matrix(src_transformed, pattern[indices.flatten()])

    # Update the transformation matrix
    M = np.dot(M_new, M)

    # Check for convergence
    if np.linalg.norm(M_new - np.eye(4)) < convergence_threshold:
        break

print('Transformation matrix: ', M)

# Create an empty DataFrame with the same columns as the original point cloud's points
df_new = pd.DataFrame(columns=cloud.points.columns)

# Apply the transformation matrix to the point cloud
x, y, z, _ = np.dot(M, np.array([cloud.points['x'], cloud.points['y'], cloud.points['z'], np.ones(cloud.points.shape[0])]))

# Fill the new point cloud with the transformed coordinates in higher resolution (0.1mm)
df_new['x'] = 10*x
df_new['y'] = 10*y
df_new['z'] = 10*z
df_new['red'] = cloud.points['red']
df_new['green'] = cloud.points['green']
df_new['blue'] = cloud.points['blue']

# Create a new point cloud with the new DataFrame
cloud_new = PyntCloud(df_new)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0, azim=-90, roll=0)

ax.scatter(cloud_new.points['x'], cloud_new.points['y'], cloud_new.points['z'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

""" # Translate the point cloud so all 'x' and 'y' values are positive
min_x = cloud.points['x'].min()
min_y = cloud.points['y'].min()
cloud.points['x'] -= min_x
cloud.points['y'] -= min_y """

print('Point cloud size: ', cloud_new.points.shape)

# Create an empty list to store the indices
indices = np.empty((int(-cloud_new.points['y'].min()) + 1, int(cloud_new.points['x'].max()) + 1), dtype=object)

# Now create the image and store the indices
image = np.zeros((int(-cloud_new.points['y'].min()) + 1, int(cloud_new.points['x'].max()) + 1, 3), dtype=np.uint8)
for index, point in cloud_new.points.iterrows():
    x_img, y_img = int(point['x']), -int(point['y'])
    image[y_img, x_img] = [point['blue'], point['green'], point['red']]
    indices[y_img, x_img] = index

print('Image shape: ', image.shape)

cv2.imshow('Point cloud repositioned', cv2.resize(image, (1080, 1080)))
cv2.waitKey(0) 