import numpy as np
import scipy

# Camera Coordinates
image_points = [[757, 213], [758, 415], [758, 686], [759, 966], 
      [1190, 172], [329, 1041], [1204, 850], [340, 159]]

#creating an nx3 matrix for image points by adding a column with ones
image_points = np.hstack((image_points, np.ones((len(image_points), 1))))
# print(image_points)
# print(np.shape(image_points))

# World Coordinates
world_points = [[0, 0, 0], [0, 3, 0], [0, 7, 0], [0, 11, 0], 
    [7, 1, 0], [0, 11, 7], [7, 9, 0], [0, 1, 7]]
world_points = np.hstack((world_points, np.ones((len(world_points), 1))))
# print(np.shape(world_points))

#A Matrix calculated as the formula provided in the lectures
A_mat = []
rows = len(image_points)
for i in range (0, rows):
    x_s,y_s,w = image_points[i]
    x_calc,y_calc,z,a = world_points[i]
    A_mat.append([0, 0, 0, 0, -w*x_calc, -w*y_calc, -w*z, -w, y_s*x_calc, y_s*y_calc, y_s*z, y_s])
    A_mat.append([w*x_calc, w*y_calc, w*z, w, 0, 0, 0, 0, -x_s*x_calc, -x_s*y_calc, -x_s*z, -x_s]) 

#creating a numpy array
A_mat = np.asarray(A_mat)
print("___________________________________________________")
print("A Matrix is")
print(A_mat)
print(A_mat.shape)
print("___________________________________________________")

# Using SVD to compute P matrix. P matrix is the last column on V mat after performing SVD
U,D,Vt = np.linalg.svd(A_mat)

#choosing the last row of Vt which is the last column of V
P = Vt[-1,:]

# #Normalizing this vector with the smallest singular value of original matrix A
# P = Vt[-1,:] / Vt[-1,-1]

#Reshaping to get a 3x4 matrix
P_mat = np.reshape(P,(3,4))
print("The obtained projection matrix (P) is ")
print(P_mat)
print(np.shape(P_mat))
print("___________________________________________________")

#The left 3x3 matrix of the Projection matrix is a product of the 
# upper triangular calibartion matrix and the orthogonal rotaion matrix denoted by M
#Performing the RQ factorization will decompose M into both matrices
M = P_mat[:, :3].tolist()
#converting to numpy array
# M = np.asarray(M).T
print("The matrix M is:")
print(M)
print("___________________________________________________")

# rot_mat,K = np.linalg.qr(M)
K_mat, rot_mat = scipy.linalg.rq(M)
K = K_mat / K_mat[2, 2] #Normalizing K matrix such that last element is 1.
print("The Intrinsic matrix for the camera calibration is: ")
print(K)
print("___________________________________________________")

#Rotation Matrix
print("The Rotation Matrix is: ")
print(rot_mat)
print("___________________________________________________")

#Finding the Extrinsix Matrix which is K_inv*P. P is a product is intrinsic and extrinsic matrices
print("The Extrinsic is: ")
extrinsic_mat = np.linalg.inv(K_mat) @ P_mat
# rot = extrinsic_mat[:, :3]
# print(rot)
print(extrinsic_mat)
print("___________________________________________________")

#Translation Matrix
print("The Translation Vector is ")
trans_mat = np.resize(extrinsic_mat[:,3].T,(3,1))
print(trans_mat)
print("___________________________________________________")

#Finding reprojection error
reprojection_error = 0
for i in range(8):
    X_s, Y_s, Z_s, _ = world_points[i]
    x_calc, y_calc, z = P_mat @ np.array([X_s, Y_s, Z_s, a])    
    x_calc /= z
    y_calc /= z
    x_s, y_s, _ = image_points[i]    
    error = np.sqrt((x_calc-x_s)**2 + (y_calc-y_s)**2)
    reprojection_error += error
    print(f"Reprojection error for point {i+1} is {error} \n")

mean_reprojection_error = reprojection_error/8
print("The average reprojection error is: ", mean_reprojection_error)
