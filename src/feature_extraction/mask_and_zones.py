import numpy as np
import matplotlib.pyplot as plt
import mahotas
from skimage.measure import perimeter


"""In this first section we define useful masks (matrix of booleans)"""

def square_mask(image, size):
    h, w = image.shape
    Y, X = np.ogrid[:h, :w]
    center = (int(w/2), int(h/2))
    dist_from_center = np.maximum(np.abs(X - center[0]), np.abs(Y-center[1]))
    mask = dist_from_center <= size
    return mask


def create_circular_mask(h, w, radius, center=None):
    """create a circular mask, masking pixels further from center 
    than the radius

    Args:
        h (int): image height
        w (int): image width
        radius (int): radius of the applied mask
        center (tuple, optional): Mask center coordinates. Defaults to None.

    Returns:
        ndarray: the circular mask
    """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def percentile_mask(image, radius=390, p1: int = 0, p2: int = 100):
    """create a boolean array(mask) of the size of the image. Pixel values between
        the p1 and p2 percentile are set to True. Other are set to False

    Args:
        image (numpy array): input image 
        radius (int): for distorted image, radius of valid pixels zone. Default to 390.
        p1 (int, optional): lower percentile. Defaults to 0.
        p2 (int, optional): upper percentile. Defaults to 100.

    Returns:
        numpy array: a mask with True values for pixels values between p1 and p2 percentiles.
    """
    assert 0 <= p1 and p1 <= 100, "percentiles must be in [0, 100]"
    assert 0 <= p2 and p2 <= 100, "percentiles must be in [0, 100]"
    assert p1 < p2, "p2 must be stricly greater than p1"
    p1_value = np.nanpercentile(image, p1)
    p2_value = np.nanpercentile(image, p2)
    mask = (p1_value <= image) & (image <= p2_value)
    if radius:
        h, w = image.shape[:2]
        border_mask = create_circular_mask(h, w, radius)
        mask[~border_mask] = np.nan
    return mask


"""We also need to extract circular crops from images. For each crop, pixels outside the zone of interest are set to NaNs""" 


def concentric_crops(image, center=None, num_bins=4):
    """Create zenith-centered zones of an image.

    Args:
        image (numpy array): an image 
        center (tuple, optional): Center coordinates to tell the concetric zones center. Defaults to None.
        num_bins (int, optional): Number of zones created. Defaults to 4.

    Returns:
        list: list of the concentric zones of the image
    """
    h, _ = image.shape
    R = [130*k for k in range(num_bins)]
    # Create a list of dense circular mask
    mask_list = [create_circular_mask(h, h, r, center=center) for r in R[1:]]
    # Extract a list of hollow mask
    concentric_zones = [~mask_list[k]*mask_list[k+1] for k in range(len(mask_list)-1)]
    # add center square and whole image square to the list
    concentric_zones = [mask_list[0]] + concentric_zones
    # Apply these masks to the image
    image_crops = []
    for zone in concentric_zones:
        img = image.copy()
        img[~zone] = np.nan
        image_crops.append(img)
    return image_crops


def circumsolar_crops(gray, to_crop, with_sun_mask=False):
    """Create circumsolar crops of an image. That is to say concentric zones centered on the sun

    Args:
        gray (numpy array): grayscale version of the image
        to_crop (numpy array): image from which crops will be made
        with_sun_mask (bool, optional): if True, sun pixels are set to NaN in the first crop. Defaults to False.

    Returns:
        list: list of circumsolar crops
    """
    # find sun coordinates
    try :
        i, j, sun_mask = sun_center(gray, False)
    except TypeError :
        return None
    circum_zones = concentric_crops(to_crop, center=(j, i), num_bins=4)
    if with_sun_mask:
        circum_zones[0][sun_mask] = np.nan
    return circum_zones


"""The next function finds sun center coordinates"""


def sun_center(image, mode_viz=False):
    """returns the sun center coordinates and create a mask of the sun disk.

    Args:
        image (numpy array): sky image 
        mode_viz (bool, optional): If True, returns the original image with the sun masked. Defaults to False.

    Returns:
        (i, j): sun center coordinates. Returned only if mode_viz=False
        sun_mask: mask of the sun, as a matrix of booleans
    """
    sun_mask = ~percentile_mask(image, p2=99)
    # find the sun center coordinates
    I, J = np.where(sun_mask==True)
    if (len(I) == 0) & (len(J) == 0):
        print("Sun not found")
        return None
    try:
        i, j = int(np.mean(I)), int(np.mean(J))
    except ValueError:
        # In case of error, returns the image center
        i, j = len(image)//2, len(image)//2
    if mode_viz:
        # mark the sun center, and colorize the sun disc in black
        sun_mask[i-1:i+2, j-1:j+2] = 0.
        masked_img = image.copy()
        masked_img[~sun_mask] = 0.
        return masked_img, sun_mask
    return i, j, sun_mask