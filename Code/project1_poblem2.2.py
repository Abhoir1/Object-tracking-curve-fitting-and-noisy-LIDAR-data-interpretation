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


# LEAST SQUARE METHOD 


# Form the matrix B and Yfrom equation BX = Y
B = np.vstack((X_data1, Y_data1, np.ones(len(X_data1)))).T
Y = Z_data1

# The solution of this equation is given by:
# coefficients = (B^T y)^(-1) @ B^T @ Y

inverse = np.linalg.inv(B.T @ B)
K = inverse @ B.T @ Y

# Extract the coefficients of plane
m, n, o = K

# Generate sequence of equally spaced points
xx, yy = np.meshgrid(np.linspace(X_data1.min(), X_data1.max(), 10),
                     np.linspace(Y_data1.min(), Y_data1.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = m * xx + n * yy + o

# Plot the ftted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_data1, Y_data1, Z_data1)
ax.plot_surface(xx, yy, zz)
plt.title("Least square method for pc1")
plt.show()

# Calculate the squared error
squared_error = np.sum((m * X_data1 + n * Y_data1 + o - Z_data1) ** 2)
print("Error for least square method for pc1 is: ", squared_error)


# TOTAL LEAST SQUARE METHOD


# Find the centriod of x y and z coordinates
centroid = np.array([np.mean(X_data1), np.mean(Y_data1), np.mean(Z_data1)])

# Shift the data to centriod
X_data1 = X_data1 - centroid[0]
Y_data1 = Y_data1 - centroid[1]
Z_data1 = Z_data1 - centroid[2]

# Stack the data in a matrix
A = np.vstack((X_data1, Y_data1, Z_data1)).T

# Calculate A^T@A
X = A.T @ A

# Calculate eigen values and eigen vectprs
eigenvalues, eigenvectors = np.linalg.eig(X)

# Find the index belonging to smallest eigenvalue
index_min_eigenvalue = np.argmin(eigenvalues)

# Find eigenvector belonging to smallest eigenvalue
min_eigenvector = eigenvectors[:, index_min_eigenvalue]

# The x y and y components give the values of coefficent a b and c
a = min_eigenvector[0]
b = min_eigenvector[1]
c = min_eigenvector[2]

# Calculate d using this formula
d = a*np.mean(X_data1) + b*np.mean(Y_data1) + c*np.mean(Z_data1)

# Generate sequence of equally spaced points
xx, yy = np.meshgrid(np.linspace(X_data1.min(), X_data1.max(), 10),
                     np.linspace(Y_data1.min(), Y_data1.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = (d - a*xx - b*yy)/c

# Plot the fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_data1, Y_data1, Z_data1)
ax.plot_surface(xx, yy, zz)
plt.title("Total Least Square Method for pc1")
plt.show()

# Calculate the mean squared error
squared_error = np.sum((a * X_data1 + b * Y_data1 + c * Z_data1 - d) ** 2)
print("Error for total least square method for pc1 is: ", squared_error)


# RANSAC METHOD


# Define the parameters for RANSAC Method

# Extract x, y and z data from the csv file
X_data1 = df1.iloc[:, 0]
Y_data1 = df1.iloc[:, 1]
Z_data1 = df1.iloc[:, 2]

X_data1 = np.array(X_data1)
Y_data1 = np.array(Y_data1)
Z_data1 = np.array(Z_data1)

num_iterations = 100000
threshold_distance = 0.6 # distance to consider a point to be an inlier
best_num_inliners = -1    # atleast 0 points can be inliner


for i in range(num_iterations):

    random_indices = np.random.choice(X_data1.size, 3, replace=False)  # choose 3 random points

    X_sample1 = X_data1[random_indices]
    Y_sample1 = Y_data1[random_indices]
    Z_sample1 = Z_data1[random_indices]

    A = np.vstack([X_sample1, Y_sample1, np.ones(3)]).T                  # Creating matrix A and B from equation Ax = B
    B = Z_sample1

    K = np.linalg.solve(A, B)                                          # Solving the three linear equations

    m, n, o = K                                                        # Coefficients m, n, o

    distance_from_plane = np.abs( m*X_data1 + n *Y_data1 - Z_data1 + o)/np.sqrt( m**2 + n**2 + 1)  # Calculatong the distance of all the points from the plane found above

    inliners = (distance_from_plane < threshold_distance).sum()        # If the distance from the plane is lesser than the threshold distance that point is consider as inliner

    if inliners > best_num_inliners:                                   # If the number of inliners is greater than previous number, it is considered as new best number
        best_num_inliners = inliners                                   # Coefficients corresponding to that best num are considered
        best_coefficient = K

m, n, o = best_coefficient 

# Plot the fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_data1, Y_data1, Z_data1)

# Generate sequence of equally spaced points
xx, yy= np.meshgrid(np.linspace(X_data1.min(), X_data1.max(), 10), np.linspace(Y_data1.min(), Y_data1.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = m*xx + n*yy + o

ax.plot_surface(xx, yy, zz)
plt.title("RANSAC Method for pc1")
plt.show()

# Calculate the mean squared error
squared_error = np.sum((m * X_data1 + n * Y_data1 + o - Z_data1) ** 2)
print("Error for RANSAC method for pc1 is: " , squared_error)


# Read the csv file
df2 = pd.read_csv("/home/ab/enpm673/project1/pc2.csv", header= None)

# Extract x, y and z data from the csv file
X_data2 = df2.iloc[:, 0]
Y_data2 = df2.iloc[:, 1]
Z_data2 = df2.iloc[:, 2]

X_data2 = np.array(X_data2)
Y_data2 = np.array(Y_data2)
Z_data2 = np.array(Z_data2)



# LEAST SQUARE METHOD 


# Form the matrix B and Yfrom equation BX = Y
B = np.vstack((X_data2, Y_data2, np.ones(len(X_data2)))).T
Y = Z_data2

# The solution of this equation is given by:
# coefficients = (B^T y)^(-1) @ B^T @ Y

inverse = np.linalg.inv(B.T @ B)
K = inverse @ B.T @ Y

# Extract the coefficients of plane
m, n, o = K

# Generate sequence of equally spaced points
xx, yy = np.meshgrid(np.linspace(X_data2.min(), X_data2.max(), 10),
                     np.linspace(Y_data2.min(), Y_data2.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = m * xx + n * yy + o

# Plot the ftted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_data2, Y_data2, Z_data2)
ax.plot_surface(xx, yy, zz)
plt.title("Least Square Method for pc2")
plt.show()

# Calculate the squared error
squared_error = np.sum((m * X_data2 + n * Y_data2 + o - Z_data2) ** 2)
print("Error for the least square method for pc2 is: ", squared_error)


# TOTAL LEAST SQUARE METHOD


# Find the centriod of x y and z coordinates
centroid = np.array([np.mean(X_data2), np.mean(Y_data2), np.mean(Z_data2)])

# Shift the data to centriod
X_data2 = X_data2 - centroid[0]
Y_data2 = Y_data2 - centroid[1]
Z_data2 = Z_data2 - centroid[2]

# Stack the data in a matrix
A = np.vstack((X_data2, Y_data2, Z_data2)).T

# Calculate A^T@A
X = A.T @ A

# Calculate eigen values and eigen vectprs
eigenvalues, eigenvectors = np.linalg.eig(X)

# Find the index belonging to smallest eigenvalue
index_min_eigenvalue = np.argmin(eigenvalues)

# Find eigenvector belonging to smallest eigenvalue
min_eigenvector = eigenvectors[:, index_min_eigenvalue]

# The x y and y components give the values of coefficent a b and c
a = min_eigenvector[0]
b = min_eigenvector[1]
c = min_eigenvector[2]

# Calculate d using this formula
d = a*np.mean(X_data2) + b*np.mean(Y_data2) + c*np.mean(Z_data2)

# Generate sequence of equally spaced points
xx, yy = np.meshgrid(np.linspace(X_data2.min(), X_data2.max(), 10),
                     np.linspace(Y_data2.min(), Y_data2.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = (d - a*xx - b*yy)/c

# Plot the fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_data2, Y_data2, Z_data2)
ax.plot_surface(xx, yy, zz)
plt.title("Total Least Square Method for pc2")
plt.show()

# Calculate the mean squared error
squared_error = np.sum((a * X_data2 + b * Y_data2 + c * Z_data2 - d) ** 2)
print("Error for total least squared method for pc2 is: " , squared_error)


# RANSAC METHOD 

df2 = pd.read_csv("/home/ab/enpm673/project1/pc2.csv", header= None)

# Extract x, y and z data from the csv file
X_data2 = df2.iloc[:, 0]
Y_data2 = df2.iloc[:, 1]
Z_data2 = df2.iloc[:, 2]

X_data2 = np.array(X_data2)
Y_data2 = np.array(Y_data2)
Z_data2 = np.array(Z_data2)


# Define the parameters for RANSAC Method

num_iterations = 1000
threshold_distance = 0.1  # distance to consider a point to be an inlier
best_num_inliners = -1    # atleast 0 points can be inliner


for i in range(num_iterations):

    random_indices = np.random.choice(X_data2.size, 3, replace=False)  # choose 3 random points

    X_sample2 = X_data2[random_indices]
    Y_sample2 = Y_data2[random_indices]
    Z_sample2 = Z_data2[random_indices]

    A = np.vstack([X_sample2, Y_sample2, np.ones(3)]).T                  # Creating matrix A and B from equation Ax = B
    B = Z_sample2

    K = np.linalg.solve(A, B)                                          # Solving the three linear equations

    m, n, o = K                                                        # Coefficients m, n, o

    distance_from_plane = np.abs( m*X_data2 + n *Y_data2 - Z_data2 + o)/np.sqrt( m**2 + n**2 + 1)  # Calculatong the distance of all the points from the plane found above

    inliners = (distance_from_plane < threshold_distance).sum()        # If the distance from the plane is lesser than the threshold distance that point is consider as inliner

    if inliners > best_num_inliners:                                   # If the number of inliners is greater than previous number, it is considered as new best number
        best_num_inliners = inliners                                   # Coefficients corresponding to that best num are considered
        best_coefficient = K

m, n, o = best_coefficient 

# Plot the fitted plane
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_data2, Y_data2, Z_data2)

# Generate sequence of equally spaced points
xx, yy= np.meshgrid(np.linspace(X_data2.min(), X_data2.max(), 10), np.linspace(Y_data2.min(), Y_data2.max(), 10))

# Find the z coordinate for these points using equation of plane
zz = m*xx + n*yy + o

ax.plot_surface(xx, yy, zz)
plt.title("RANSAC Method for pc2")
plt.show()

# Calculate the mean squared error
squared_error = np.sum((m * X_data2 + n * Y_data2 + o - Z_data2) ** 2)
print("Error for RANSAC method for pc2 is: ", squared_error)