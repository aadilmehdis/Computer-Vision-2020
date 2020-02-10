from scipy import linalg
import numpy as np
import random 

def calculate_projection_error(P, world_coords, img_coords):
    # Convert points into homogeneous coordinates
    world_coords_homogeneous = np.concatenate((world_coords, np.ones((len(world_coords), 1))), axis=1)
    image_coords_homogeneous = np.concatenate((img_coords, np.ones((len(img_coords), 1))), axis=1)

    # Calculate the projected points and normalize them
    projected_coords = P @ world_coords_homogeneous.T
    projected_coords[0, :] = projected_coords[0, :] / projected_coords[2, :]
    projected_coords[1, :] = projected_coords[1, :] / projected_coords[2, :]
    projected_coords[2, :] = projected_coords[2, :] / projected_coords[2, :]
    projected_coords = projected_coords.T

    # Calculate the projection error
    projection_error =  np.mean(np.sum(np.square(image_coords_homogeneous - projected_coords), axis=1))

    return projection_error

def construct_DLT_matrix(X, x):
    M = np.zeros((0, 12))
    for i in range(len(X)):
        m = np.array([
            [-X[i][0], -X[i][1], -X[i][2], -1, 0, 0, 0, 0, x[i][0]*X[i][0], x[i][0]*X[i][1], x[i][0]*X[i][2], x[i][0]],
            [0, 0, 0, 0, -X[i][0], -X[i][1], -X[i][2], -1, x[i][1]*X[i][0], x[i][1]*X[i][1], x[i][1]*X[i][2], x[i][1]],
            ])
        M = np.concatenate((M, m))
    return M

def RANSAC(world_coords, img_coords, max_iterations=5000):

    min_proj_error = 999999999

    # Best estimate of Projection matrix by far
    _P = np.zeros((3, 4))

    for i in range(max_iterations):
                
        # Randomly select 6 world points and the corresponding image points
        idx = random.sample(range(0, 50), 6)        
        X = world_coords[idx]
        x = img_coords[idx]

        # Perform DLT and get the Transformation Matrix
        P = DLT(X, x)

        # Calculate projection error
        projection_error = calculate_projection_error(P, world_coords, img_coords)
        # Check if projection error is lesser than the minimum projection error so far
        if projection_error < min_proj_error:
            min_proj_error = projection_error
            _P = P

        # Repeat for a maximum number of iterations
    
    return _P

def DLT(X, x):
    # Construct the DLT Matrix
    M = construct_DLT_matrix(X, x)

    # Perform SVD on the Matrix
    U, s, Vh = np.linalg.svd(M)

    # Extract the 12th row and Normalize it
    P = Vh[-1, :] / Vh[-1, -1]

    # Reshape the row and get the projection matrix
    P = P.reshape(3, 4)

    return P

def decompose_P(P):

    # H = KR
    H = P[:,0:3]

    # h = -KRT
    h = P[:,3]

    # Get Translation vector
    t = -np.linalg.inv(H)@h 

    # Do QR decomposition of inv(H)
    # This gives R.T and inv(K)
    R, K = np.linalg.qr(np.linalg.inv(H))

    # Obtain the R and K matrices
    R = R.T
    K = np.linalg.inv(K)

    # Normalize K due to homogeneity
    K = K / K[-1,-1]

    return K, R, t

def main():
    pass 
    # Load the world coordinates
    world_coords = np.load('./resources/world_coordinates.npy')

    # Load the pixel coordinates
    pixel_coords = np.load('./resources/pixel_coordinates.npy')

    # Run RANSAC
    P = RANSAC(world_coords, pixel_coords, max_iterations=50000)

    # Output the Tranformation Matrix
    print("The Projection Matrix:")
    print(P)

    # Decompose Projection Matrix into Intrinsic, Rotation and Translation
    K, R, t = decompose_P(P)
    
    print("Camera Intrinsic Matrix:")
    print(K)

    print("Camera Rotation Matrix:")
    print(R)

    print("Camera Translation Vector:")
    print(t)

    print("Projection Error (MSE) for the Above Projection Matrix: {}".format(calculate_projection_error(P, world_coords, pixel_coords)))

if __name__ == '__main__':
    main()