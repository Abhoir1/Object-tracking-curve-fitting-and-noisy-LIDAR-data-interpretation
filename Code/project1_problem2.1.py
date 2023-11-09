# Import the necessary libaries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the csv file
df1 = pd.read_csv("/home/ab/enpm673/project1/pc1.csv", header= None)

# Extract x, y and z data from the csv file
X_data1 = df1.iloc[:, 0]
Y_data1 = df1.iloc[:, 1]
Z_data1 = df1.iloc[:, 2]

X_data1 = np.array(X_data1)
Y_data1 = np.array(Y_data1)
Z_data1 = np.array(Z_data1)

mean_X = df1.iloc[:, 0].mean()
# print(mean_X)

mean_Y = df1.iloc[:, 1].mean()
# print(mean_Y)

mean_Z = df1.iloc[:, 2].mean()
# print(mean_Z)

for i in range (0, 300):
    Sx = (df1.iloc[:,0] - mean_X) * (df1.iloc[:,0] - mean_X)
Sx = Sx.sum()
S_X = Sx/len(df1)
# print(S_X)

for i in range (0, 300):
    Sy = (df1.iloc[:,1] - mean_Y) * (df1.iloc[:,1] - mean_Y)
Sy = Sy.sum()
S_Y = Sy/len(df1)
# print(S_Y)

for i in range (0, 300):
    Sz = (df1.iloc[:,2] - mean_Z) * (df1.iloc[:,2] - mean_Z)
Sz = Sz.sum()
S_Z = Sz/len(df1)
# print(S_Z)

for i in range (0, 300):
    S_XY = (df1.iloc[:,0] - mean_X) * (df1.iloc[:,1] - mean_Y)
S_XY = S_XY.sum()
S_XY = S_XY/len(df1)
# print(S_XY)

for i in range (0, 300):
    S_YZ = (df1.iloc[:,1] - mean_Y) * (df1.iloc[:,2] - mean_Z)
S_YZ = S_YZ.sum()
S_YZ = S_YZ/len(df1)
# print(S_YZ)

for i in range (0, 300):
    S_XZ = (df1.iloc[:,0] - mean_X) * (df1.iloc[:,2] - mean_Z)
S_XZ = S_XZ.sum()
S_XZ = S_XZ/len(df1)
# print(S_XZ)

Covariance_matrix = np.array([[S_X, S_XY, S_XZ], [S_XY, S_Y, S_YZ], [S_XZ, S_YZ, S_Z]])

print("Covariance_matrix is :")
print(Covariance_matrix)

eigenvalues1, eigenvectors1 = np.linalg.eig(Covariance_matrix)

# Find the index belonging to smallest eigenvalue
index_min_eigenvalue1 = np.argmin(eigenvalues1)

# Find eigenvector belonging to smallest eigenvalue
min_eigenvector1 = eigenvectors1[:, index_min_eigenvalue1]

# Calculate magnitude of surface normal
mangnitude_normal = np.linalg.norm(min_eigenvector1)

# Calculate direction of normal vector
direction_normal = min_eigenvector1 / mangnitude_normal

dir = np.arctan2(direction_normal[1], direction_normal[0])

print("Mangintude and direction of surface normal are: ", mangnitude_normal, "," ,direction_normal)
print("Angle theta in radinas is: ", dir)
