import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from glob import glob

def construct_H_matrix(x, xs):
    A = np.zeros((0, 9))
    for i in range(len(x)):
        a = np.array([
            [x[i][0], x[i][1], 1, 0, 0, 0, -xs[i][0]*x[i][0], -xs[i][0]*x[i][1], -xs[i][0]],
            [0, 0, 0, x[i][0], x[i][1], 1, -xs[i][1]*x[i][0], -xs[i][1]*x[i][1], -xs[i][1]],
            ])
        A = np.concatenate((A, a))
    return A


def transform_points(H, x):
    # Convert points into homogeneous coordinates
    x_homogeneous = np.concatenate((x, np.ones((len(x), 1))), axis=1)

    # Calculate the transformed points and normalize them
    xs = H @ x_homogeneous.T
    xs[0, :] = xs[0, :] / xs[2, :]
    xs[1, :] = xs[1, :] / xs[2, :]
    xs[2, :] = xs[2, :] / xs[2, :]
    xs = xs.T

    return xs   


def calculate_transformation_error(H, x, xs):
    # Convert points into homogeneous coordinates
    xs_homogeneous = np.concatenate((xs, np.ones((len(xs), 1))), axis=1)

    # Calculate the projected points
    transformed_coords = transform_points(H, x)

    # Calculate the projection error
    transformation_error =  np.linalg.norm(xs_homogeneous - transformed_coords, axis=1)

    return transformation_error  


def RANSAC(img_coords_1, img_coords_2, num_points, max_iterations=20000, thresh=5.0):

    min_transform_error = 999999999
    max_inliers = -999999999

    # Best estimate of Projection matrix by far
    _H = np.zeros((3, 3))

    inliers = []

    for i in range(max_iterations):

        # Randomly select 4 world points and the corresponding image points
        idx = random.sample(range(0, num_points), 4)        
        x  = img_coords_1[idx]
        xs = img_coords_2[idx]

        # Perform DLT and get the Transformation Matrix
        H = DLT(x, xs)

        # Calculate projection error
        transformation_error = calculate_transformation_error(H, img_coords_1, img_coords_2)


        inliers = 0
        for i in transformation_error<=5:
            if i == True:
                inliers+=1

        print(inliers, max_inliers)
        if inliers > max_inliers:
            max_inliers = inliers
            _H = H

        # # Check if transformation error is lesser than the minimum transformation error so far
        # if transformation_error < min_transform_error:
        #     min_transform_error = transformation_error
        #     _H = H

        # Repeat for a maximum number of iterations
    
    return _H


def DLT(x, xs):
    # Construct the DLT Matrix
    A = construct_H_matrix(x, xs)


    # Perform SVD on the Matrix
    U, s, Vh = np.linalg.svd(A.T @ A)

    # Extract the 9th row and Normalize it
    A = Vh[-1, :] / Vh[-1, -1]

    # Reshape the row and get the projection matrix
    A = A.reshape(3, 3)

    return A


def find_matching_points(image1, image2, num_points=100):
    
    # Initiate sift detector
    # sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    
    # Create BF Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Find keypoints and descriptors 
    kp1, desc1 = orb.detectAndCompute(image1, None)    
    kp2, desc2 = orb.detectAndCompute(image2, None)

    # Match descriptors
    matches = bf.match(desc1, desc2) 

    # Sort the matches in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)[:num_points]

    print("Number of matches found", len(matches))
    print()
    # Draw first 10 matches.
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)

    plt.imshow(img3),plt.show()


       
    if len(matches) >= 4:
        x = np.array([ kp1[m.queryIdx].pt for m in matches ])#.reshape(-1,1,2)
        xs = np.array([ kp2[m.trainIdx].pt for m in matches ])#.reshape(-1,1,2)
    return x, xs

def warp_image(image_1, image_2, H):
    im_out = cv2.warpPerspective(image_2, H, (image_1.shape[1] + image_2.shape[1], image_2.shape[0]))
    # plt.subplot(122)
    # plt.imshow(im_out)
    # plt.show()
    return im_out

def stitch_images(images):

    for i in range(len(images)-1):
        x, xs = find_matching_points(images[0], images[i+1])
        H = RANSAC(xs, x, len(xs))
        h, status = cv2.findHomography(xs.reshape(-1,1,2), x.reshape(-1,1,2), cv2.RANSAC, 5.0)
        print("Estimated Homography Matrix using RANSAC\n",H)
        print("Homography matrix using in-builts\n",h)
        print("Tranformation Error(1): ", calculate_transformation_error(H, x, xs))
        print("Tranformation Error(2): ", calculate_transformation_error(h, x, xs))
        dst1 = warp_image(images[0], images[i+1], H)
        dst2 = warp_image(images[0], images[i+1], h)

        # dst1[0:images[i].shape[0], 0:images[i].shape[1]] = images[0]
        # dst2[0:images[i].shape[0], 0:images[i].shape[1]] = images[0]

        # plt.show()
        plt.imshow(dst1)
        plt.show()

        plt.imshow(dst2)
        plt.show()

def main():
    image_files = sorted(glob('./image_mosaicing/img1/*'))
    images = []
    for im in image_files:
        images.append(cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY))
    
    stitch_images(images)
    pass


main()
