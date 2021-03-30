import numpy as np
from cv2 import cv2 # careful, cv2 uses BGR convention
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm

from preprocessing import prepro


def find_intrinsec_params(cam_path):
    """Returns intrinsic parameters of the fisheye camera, thanks to calibration images

    Args:
        cam_path (str): path to the folder containing the images for calibration

    Returns:
        K, D: intrinsic parameters of the camera
        DIM: dimension of the images used for calibration
    """
    CHECKERBOARD = (10,15)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = cam_path + "\\*.jpg"
    images = glob(path)


    for fname in tqdm(images):
        image = cv2.imread(fname)
        img = prepro(image, 480, outside_val=0.)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    _, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    DIM = _img_shape[::-1]
    print("DIM=" + str(DIM))
    return K, D, DIM

def find_undistort_maps(cam_path):
    """Save the undistortion maps, using calibration images

    Args:
        cam_path (Path): path to the folder containing images with checkerboards, used for calibration. Please use \\
    """
    K, D, DIM = find_intrinsec_params(cam_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    
    num_cam = cam_path.split("\\")[-2]
    save_path = "../../processed_data/MapsSquare/" + num_cam

    np.save(save_path + "map1.npy", map1)
    np.save(save_path + "map2.npy", map2)



def undistort(img_path):
    """return a plane projection of a fisheye image

    Args:
        img_path (Path): path to the original image

    Returns:
        numpy array: "undistorted image"
    """
    cam_num = img_path.split("/")[-2]
    map1 = np.load("../../Solais_Data/MapsSquare/" + cam_num + "map1.npy")
    map2 = np.load("../../Solais_Data/MapsSquare/" + cam_num + "map2.npy")

    img = cv2.imread(img_path)
    square_img = prepro(img, 480, outside_val=0.)
    undistorted_img = cv2.remap(square_img, 
                                map1, 
                                map2, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
                                
    #cv2 works with GBR images, but RGB is standard in many other libraries (like matplotlib)
    im_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    return im_rgb/255





