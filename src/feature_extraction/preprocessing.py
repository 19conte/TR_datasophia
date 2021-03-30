import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import cv2
from img_undistort import undistort


"""preprocess RGB images : circular mask, square images"""


def downscale(img, new_size):
    dim = (new_size, new_size)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    return resized


def prepro(img, radius, outside_val, new_size=None):
    """preprocesses the fisheye image : square crop, circular mask, and downscale.

    Args:
        img (numpy array): the input image
        radius (int): radius of the centered mask applied on the picture, in pixels
        new_size (int): new size of the square image after downscaling
    """
    image = img.copy()
    h, w = image.shape[:2]

    # Mask the date in the upper-right corner
    image[:30, 700:] = [outside_val, outside_val , outside_val]

    # Square crop
    image = image[:, w//2 - h//2 : w//2 + h//2]
   
    # Circular mask
    mask = create_circular_mask(h, h, radius=radius)
    masked_img = image.copy()
    masked_img[~mask] = [outside_val, outside_val , outside_val]

    # Downsample 
    if new_size:
        masked_img = downscale(masked_img, new_size) #FINDBETTER
    return masked_img


"""Then, from an image file path, we extract two images (grayscale and R-B normalized ratio image)"""


def split_two_canals(image):
    """creates two images from an RGB image (grayscale and R-B normalized ratio image)

    Args:
        image (numpy array): original RGB image

    Returns:
        tuple: the grayscale image, and the R-B normalized ratio image
    """
    # get the average of each pixel
    grayImage = np.mean(image, axis=2)
    # get the R-B/R+B ratio of each pixel
    eps = 0.00001
    r_bImage = (image[:,:,0] - image[:,:,2])/(image[:,:,0] + image[:,:,2] + eps)

    return grayImage, r_bImage


def path_to_canals(image_path, with_undistortion = False, outside_val=np.nan, min_max=False):
    """from an image file path, we extract two images (grayscale and R-B normalized ratio image)

    Args:
        image_path (numpy array): RGB image
        with_undistortion (bool, optional): If True extract images from a plane projection of the image. Defaults to False.
        outside_val (float, optional): Value of pixels outside the sky zone. Defaults to np.nan.
        min_max (bool, optional): Display min and max pixel value of the two images. Defaults to False.

    Returns:
        list: the original image and the two images created
    """
    if with_undistortion:
        image = undistort(image_path)
    else:
        img = np.array(Image.open(image_path))/255
        image = prepro(img, 390, outside_val=outside_val)
    
    gray, r_b = split_two_canals(image)

    if min_max:
        for name, img in zip(["gray", "r_b"], [gray, r_b]): 
            print(f"min de {name} = {np.min(img)} | max de {name} = {np.max(img)}")
    if (~with_undistortion) and (outside_val == 0.):
        # we ensure that pixels outside the zone of interest are black, on the r_b image.
        h = r_b.shape[0]
        mask = create_circular_mask(h, h, radius=390)
        masked_r_b = r_b.copy()
        masked_r_b[~mask] = outside_val
        return image, gray, masked_r_b
    return image, gray, r_b