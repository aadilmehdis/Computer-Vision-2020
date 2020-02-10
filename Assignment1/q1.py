from scipy import linalg as LA
import numpy as np
import random 

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

def QR():
    pass

def main():
    pass 
    # Load the world coordinates
    world_coords = np.load('./resources/world_coordinates.npy')

    # Load the pixel coordinates
    pixel_coords = np.load('./resources/pixel_coordinates.npy')

    # Run RANSAC
    P = RANSAC(world_coords, pixel_coords, max_iterations=50000)

    # Output the Tranformation Matrix

    # Convert points into homogeneous coordinates
    world_coords_homogeneous = np.concatenate((world_coords, np.ones((len(world_coords), 1))), axis=1)
    image_coords_homogeneous = np.concatenate((pixel_coords, np.ones((len(pixel_coords), 1))), axis=1)

    # Calculate the projected points and normalize them
    projected_coords = P @ world_coords_homogeneous.T
    projected_coords[0, :] = projected_coords[0, :] / projected_coords[2, :]
    projected_coords[1, :] = projected_coords[1, :] / projected_coords[2, :]
    projected_coords[2, :] = projected_coords[2, :] / projected_coords[2, :]
    projected_coords = projected_coords.T

    print(P)
    print(projected_coords)
    print(image_coords_homogeneous)


    # Perform QR decomposition and get R and T

    # Output R and T

if __name__ == '__main__':
    main()