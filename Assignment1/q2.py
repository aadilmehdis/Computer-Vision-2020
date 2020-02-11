import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import math
import random
import scipy.linalg.null_space

def euler_angles_to_rotation(theta) :
    R_x = np.array([[1,0,0],[0,math.cos(theta[0]),-math.sin(theta[0])],[0,math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],[0,1,0],[-math.sin(theta[1]),0,math.cos(theta[1])]])             
    R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],[math.sin(theta[2]),math.cos(theta[2]),0],[0,0,1]])
    R = np.dot(R_z,np.dot(R_y,R_x))
    return R

def calibrate_camera(path, width, height, cell_length, rng):
    x, y = np.meshgrid(range(width),range(height))
    num_points = width*height
    world_coords = np.hstack((cell_length*x.reshape(num_points, 1), cell_length*y.reshape(num_points, 1), np.zeros((num_points, 1)))).astype(np.float32)
    _3d_points = []
    _2d_points = []

    imgs = []
    for i in range(rng[0],rng[1]):
        im=cv2.imread(path + str(i) + '.jpeg')
        print(im)
        imgs.append(im)
        ret, corners = cv2.findChessboardCorners(im, (width, height))
        
        if ret: 
            _2d_points.append(corners) 
            _3d_points.append(world_coords) 

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]), None, None)

    return ret, mtx, dist, rvecs, tvecs, imgs

def get_projections(image_number, mtx, dist, rvecs, tvecs, width, height, cell_length):
    R = euler_angles_to_rotation(rvecs[image_number])

    x, y = np.meshgrid(range(width),range(height))
    num_points = width*height
    world_coords = np.hstack((cell_length*x.reshape(num_points,1),cell_length*y.reshape(num_points,1),np.zeros((num_points,1)),np.ones((num_points,1)))).astype(np.float32)

    RnT = np.zeros((3,4))
    RnT[0:3, 0:3] = R[0:3, 0:3]
    RnT[:,3] = tvecs[image_number][:,0]
    P = mtx @ RnT
    print(scipy.linalg.null_space(P))
    P = P/P[2,3]

    projected_coords = P @ world_coords.T
    projected_coords = projected_coords / projected_coords[2]
    projected_coords = projected_coords[0:2]

    return projected_coords

def plot_wireframe(projected_coords, width, height):
    idx = [7, 15, 23, 31, 39, 47]
    p=0
    num_points = height*width
    
    projected_coords = projected_coords.T

    for i in range(projected_coords.shape[0]):
        if i == idx[p]:
            p += 1
            continue
        plt.plot( [projected_coords[i][0], projected_coords[i+1][0]], [projected_coords[i][1], projected_coords[i+1][1]], 'ro-')

    for i in range(width):
        k = i 
        j = i + width
        while j < num_points:
            plt.plot( [projected_coords[k][0],projected_coords[j][0]], [projected_coords[k][1],projected_coords[j][1]], 'ro-')
            k = j
            j += width

    plt.title('Wireframe from reprojected points')
    plt.show() 

def main():
    # ret, mtx, dist, rvecs, tvecs, imgs = calibrate_camera('./Camera_calibration_data/IMG_', 8, 6, 29, (5456, 5471))
    ret, mtx, dist, rvecs, tvecs, imgs = calibrate_camera('./resources/iPhone_Zhang/images/z', 8, 6, 29, (1, 12))

    print(mtx)
    print(dist)
    print(rvecs)
    print(tvecs)
    for i in range(len(imgs)):
        projected_coords = get_projections(i, mtx, dist, rvecs, tvecs, 8, 6, 29)
        plt.imshow(imgs[i])
        plot_wireframe(projected_coords, 8, 6)
main()