# IMPORT PACKAGES AND MODULES
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
from scipy import stats
from sklearn.neighbors import NearestNeighbors


# IMPORT USER-DEFINED MODULES
from detectMarkers import detect

def calculate_transformation_matrix_horn(points_3d_source, points_3d_target):
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

def calculate_transformation_matrix_svd(points_3d_source, points_3d_target):
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

""" test1 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]))
print(f'Rotation around x-axis: \n{test1}')
test2 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]))
print(f'Rotation around y-axis: \n{test2}')
test3 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]))
print(f'Rotation around z-axis: \n{test3}')
test4 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]]))
print(f'Translation along x-axis: \n{test4}')
test5 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 0], [1, 1, 0], [0, 2, 0]]))
print(f'Translation along y-axis: \n{test5}')
test6 = calculate_transformation_matrix_horn(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]))
print(f'Translation along z-axis: \n{test6}') """

# Define pattern as 3D coordinates in mm
pattern = 10 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], [9.12, 8.92, 0], [1.05, 8.92, 0],
                          [50.7, 0.91, 0], [58.77, 0.94, 0], [58.75, 9.04, 0], [50.66, 9, 0],
                          [50.57, 51.09, 0], [58.63, 50.99, 0], [58.73, 59.07, 0], [50.66, 59.14, 0],
                          [1.08, 51.02, 0], [9.15, 51.02, 0], [9.12, 59.13, 0], [1.04, 59.11, 0]])

# Load the .ply file
cloud = PyntCloud.from_file('H:/data/cloudcompare/test/sand_minus.ply')

# Translate the point cloud so all 'x' and 'y' values are positive
min_x = cloud.points['x'].min()
min_y = cloud.points['y'].min()
cloud.points['x'] -= min_x
cloud.points['y'] -= min_y
max_y = cloud.points['y'].max()

print('Point cloud size: ', cloud.points.shape)

# Create an empty list to store the indices
indices = np.empty((int(cloud.points['y'].max()) + 1, int(cloud.points['x'].max()) + 1), dtype=object)

# Now create the image and store the indices
image = np.zeros((int(cloud.points['y'].max()) + 1, int(cloud.points['x'].max()) + 1, 3), dtype=np.uint8)

for index, point in cloud.points.iterrows():
    img_x, img_y = int(point['x']), int(point['y'])
    image[img_y, img_x] = [point['blue'], point['green'], point['red']]
    indices[img_y, img_x] = index
print('Image shape: ', image.shape)

cv2.rectangle(image, (0, 0), [600, 600], (255, 0, 0), 5) 
cv2.imshow('Point cloud', cv2.resize(image, (1080, 1080)))
cv2.waitKey(0) 
cv2.imwrite('H:/data/cloudcompare/test/image.jpg', image)

# Detect ArUco markers
marker = 'DICT_4X4_50'
img_det, corners, ids = detect(image, marker)

print(f'Corners shape: {corners.shape}')
print(f'Pattern shape: {pattern.shape}')
print(f'Corners: {corners}')
print(f'Pattern: {pattern}')

# Draw the pattern points on the detected image
i =0
for point in pattern:
    i += 1
    cv2.circle(img_det, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    if i == 8:
        break

cv2.circle(img_det, tuple(corners[6].astype(int)), 5, (0, 0, 255), -1)
cv2.imshow('image', img_det)
cv2.waitKey(0)
cv2.imwrite('H:/data/cloudcompare/test/detected.jpg', img_det)

# Define corners_3d array
corners_3d = np.hstack((corners, np.zeros((corners.shape[0], 1), dtype=corners.dtype)))

src = corners_3d

""" # Define the source array
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
print(f'Mean of all errors: {distances_min.mean()}') """

print(f'Source shape: {src.shape}')
print(f'Source: {src}')


# Initialize the transformation matrix to the identity matrix
M = calculate_transformation_matrix_horn(src, pattern)

# Define the maximum number of iterations and the convergence threshold
max_iterations = 100
convergence_threshold = 1e-6

""" # Create a nearest neighbors object for the target points
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pattern)

iteration = 0

for i in range(max_iterations):
    # Apply the current transformation to the source points
    src_transformed = np.dot(M, np.vstack((src.T, np.ones(src.shape[0]))))[:3].T

    # Find the closest points in the target point cloud
    distances, indices = nbrs.kneighbors(src_transformed)

    # Calculate the new transformation that best aligns the transformed source points to the closest target points
    M_new = calculate_transformation_matrix_horn(src_transformed, pattern[indices.flatten()])

    # Update the transformation matrix
    M = np.dot(M_new, M)

    iteration += 1

    # Check for convergence
    if np.linalg.norm(M_new - np.eye(4)) < convergence_threshold:
        print(f'Exited after {iteration} iterations')
        break
 """
print('Transformation matrix: ', M)

# Create an empty DataFrame with the same columns as the original point cloud's points
df_new = pd.DataFrame(columns=cloud.points.columns)
print(f'df_new: {df_new}')

# Apply the transformation matrix to the point cloud
x, y, z, _ = np.dot(M, np.array([cloud.points['x'], cloud.points['y'], cloud.points['z'], np.ones(cloud.points.shape[0])]))

# Shift the y-coordinates to negative
y = 600 - y
z = -z

# Turn all coordinates around x-axis
#x, y, z = np.dot(R.from_euler('x', 180, degrees=True).as_matrix(), np.array([x, y, z]))

# Fill the new point cloud with the transformed coordinates in higher resolution (0.1mm)
df_new['x'] = 10*x
df_new['y'] = 10*y
df_new['z'] = 10*z
df_new['red'] = cloud.points['red']
df_new['green'] = cloud.points['green']
df_new['blue'] = cloud.points['blue']

print(f'df_new: {df_new.shape}')

# Create a new point cloud with the new DataFrame
cloud_new = PyntCloud(df_new)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.view_init(elev=0, azim=0, roll=0) # Set the view to y-z plane
ax.view_init(elev=0, azim=-90, roll=0) # Set the view to x-z plane
#ax.view_init(elev=90, azim=-90, roll=0) # Set the view to x-y plane

ax.scatter(cloud_new.points['x'], cloud_new.points['y'], cloud_new.points['z'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Export the point cloud as a PLY and asc file
cloud_new.to_file('H:/data/cloudcompare/test/output.ply', also_save=['points'])
df_new.to_csv('H:/data/cloudcompare/test/output.asc', sep=' ', index=False, header=False)

# Define the size of the bins
bin_size = 1

# Bin the x and y coordinates
x_bins = np.arange(df_new['x'].min(), df_new['x'].max(), bin_size)
y_bins = np.arange(df_new['y'].min(), df_new['y'].max(), bin_size)

# Use np.histogram2d to bin the x and y coordinates and compute the mean z-coordinate in each bin
counts, x_edges, y_edges, binnumber = stats.binned_statistic_2d(df_new['x'], df_new['y'], df_new['z'], statistic='mean', bins=[x_bins, y_bins])

# Replace NaN values with a default value
counts = np.nan_to_num(counts, nan=0.0)

# Display the height map
plt.imshow(counts, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

# Add a colorbar to show the height scale
plt.colorbar(label='Height')

# Show the plot
plt.show()

# Create an empty list to store the indices
indices = np.empty((int(cloud_new.points['y'].max()) + 1, int(cloud_new.points['x'].max()) + 1), dtype=object)

# Now create the image and store the indices
image = np.zeros((int(cloud_new.points['y'].max()) + 1, int(cloud_new.points['x'].max()) + 1, 3), dtype=np.uint8)

for index, point in cloud_new.points.iterrows():
    img_x, img_y = int(point['x']), int(point['y'])
    image[img_y, img_x] = [point['blue'], point['green'], point['red']]
    indices[img_y, img_x] = index
print('Image shape: ', image.shape)

cv2.rectangle(image, (0, 0), [6000, 6000], (255, 0, 0), 5) 
cv2.imshow('Point cloud repositioned', cv2.resize(image, (1080, 1080)))
cv2.waitKey(0) 
cv2.imwrite('H:/data/cloudcompare/test/repositioned.jpg', image)