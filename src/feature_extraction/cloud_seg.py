import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import watershed, find_boundaries
from skimage.filters import gaussian
from skimage.measure import perimeter
import mahotas

from mask_and_zones import create_circular_mask, circumsolar_crops


def sun_segmentation(img_to_modif, based_on_img, sun_val):
    """Modify the img_to_modif according to sun position on the based_on_img

    Args:
        img_to_modif (numpy array): image whose pixels in the sun zone will be modified
        based_on_img (numpy array): image on which the sun disk will be identified
        sun_val (int): value that will be attributed to pixels identified as sun pixels on the based_on_img
    """
    sun_thres = np.nanpercentile(based_on_img, 98)
    img_to_modif[based_on_img > sun_thres] = sun_val

def high_filter(image, param=1.6):
    """Computes the high-pass filtered version of an image

    Args:
        image (numpy array): grayscale image
        param (float, optional): filter intensity parameter. Defaults to 1.6.

    Returns:
        numpy array: high filtered image
    """
    nan_pos = np.isnan(image)
    img = image.copy()
    mean = np.nanmean(img)
    img[nan_pos] = mean

    low_filtered_image = gaussian(img, sigma = 4)
    gau = gaussian(img, sigma = 4/param)
    high_filtered_image = gau - low_filtered_image

    high_filtered_image[nan_pos] = np.nan
    return high_filtered_image

def target(hist, bins, t):
    """Computes the cross-entropy between two parts of an histogram

    Args:
        hist (list): an histogram
        bins (list): list of the bins of the histogram
        t (float): threshold value cutting the histogram in two groups

    Returns:
        float: cross-entropy value
    """
    m1 = np.sum(hist[bins <= t]*bins[bins <= t])
    m2 = np.sum(hist[bins >= t]*bins[bins >= t])
    s1, s2 = np.sum(hist[bins <= t]), np.sum(hist[bins >= t])
    mu1, mu2 = m1/s1, m2/s2
    res = -m1 * np.log(mu1) - m2 * np.log(mu2)
    return res


def tar(t, his, bin_edges):
    """Apply the function target to each element of t, with the fixed parameters hist and bins

    Args:
        t (numpy array): list of possible threshold values
        his (list): an histogram
        bin_edges (list): list of the bins of the histogram

    Returns:
        numpy array: cross-entropy values for each separations of the histogram
    """
    get_target = lambda x: target(his, bin_edges, x)
    return np.vectorize(get_target)(t)


def adaptative_threshold(image):
    """Find the best segmentation threshold for the (R-B)/(R+B) image, based on its histogram

    Args:
        image (numpy array): R-B normalized ratio image

    Returns:
        float: threshold value between cloud and sky pixels
    """
    img = image[~np.isnan(image)]
    his, bin_edges = np.histogram(img, bins = 100)
    starter = np.percentile(img, 5)
    finisher = np.percentile(img, 95)
    t = np.linspace(starter, finisher, 100)
    targets = tar(t, his, bin_edges[:-1])
    return t[np.argmin(targets)]


def cloud_segmentation(gray_image, rb):
    """Perform the cloud segmentation of a sky image. Uses a watershed algorithm (requires markers and an elevation map).
        Markers are found thanks to cross-entropy maximization thresholds and a simple decision tree.
        The elevation map is obtained by high-filtering the grayscale image.

    Args:
        gray_image (numpy array): grayscale version of the image
        rb (numpy array): R-B normalized ratio version of the image

    Returns:
        : [description]
    """
    from feature_extractor import sun_isoperimetric_ratio
    
    r_b_img = rb.copy()
    r_b_img -= np.nanmin(r_b_img)
    #initialize markers
    sun_mask = np.zeros_like(r_b_img)
    markers = np.zeros_like(r_b_img)
    mask = create_circular_mask(r_b_img.shape[0], r_b_img.shape[0], radius=390)
    mean = np.nanmean(r_b_img)
    std = np.nanstd(r_b_img)

    #sun segmentation, put sun pixels to 3 in the sun_mask matrix
    sun_segmentation(sun_mask, gray_image, sun_val=3)

    #high pass filter, for the filtering to make sense, sun pixels are set to the mean
    r_b_img[sun_mask == 3] = mean
    filtered_image = high_filter(r_b_img)
    fil_std = np.nanstd(filtered_image)

    # Put the sun pixels back to NaNs
    sun_segmentation(r_b_img, gray_image, sun_val=np.nan)

    # Define functions for the decision tree
    def is_not_uniform(fil_std):
        return fil_std > 0.0043

    def not_round_sun(gray_image):
        return sun_isoperimetric_ratio(gray_image) < 0.48

    #Decision tree
    if is_not_uniform(fil_std):
        markers[r_b_img < mean - 0.1*std] = 1 # sky pixels
        crops = circumsolar_crops(gray_image, r_b_img, True)
        for crop in crops:
            # Use cross-entropy thresholds
            thres = adaptative_threshold(crop)
            markers[crop > 1.*thres] = 2
    else:
        if not_round_sun(gray_image):
            markers[r_b_img > mean - 0.1*std] = 2 # 2 for cloud pixels
        else:
            markers[r_b_img < mean - 0.1*std] = 1 # 1 for sky pixels

    # Perform watershed 
    segmentation = watershed(filtered_image, markers)
    segmentation[~mask] = 0.
    sun_segmentation(segmentation, gray_image, sun_val=3)
    # Remodify border pixels for vizualization :
    r_b_img[~mask] = np.nan

    return segmentation


def cloud_isoperimetric_ratio(cloud_mask, mode_viz=False):
    """Compute the cloud isoperimetric ratio of an image. It indicates the circularity of the cloud zone. This is done to detect 
        mis-segmentation in the circumsolar area, caused by aerosols.
    Args:
        cloud_mask (numpy array): segmented image obtained thanks to the cloud_segmentation function
        mode_viz (bool, optional): Display the cloud border pixels. Defaults to False.

    Returns:
        float: the cloud isoperimetric ratio
    """
    blurred_mask = mahotas.gaussian_filter(cloud_mask, 4)
    blurred_mask = (blurred_mask > blurred_mask.mean())
    # Obtain a binary image with the cloud borders in white pixels
    cloud_perim_mask = mahotas.labeled.bwperim(blurred_mask, 8)
    # Compute the perimeter in pixels
    cloud_perim = int(perimeter(cloud_perim_mask))
    # Compute the surface in pixels
    cloud_surface = np.nansum(blurred_mask)
    # ratio = 4*pi*S/(P^2). Is in [0,1], equals 1 for a circle
    ratio = 4*np.pi*cloud_surface/(cloud_perim**2)
    if mode_viz:
        # Plot
        print(f"perimeter = {cloud_perim} | surface = {cloud_surface}")
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax1.imshow(blurred_mask, cmap="gray")
        ax2 = fig.add_subplot(122)
        ax2.imshow(cloud_perim_mask, cmap="gray")
        plt.show()
    return ratio


def cloud_features(gray, r_b, radius=390):
    """Calculate 4 cloud-related features of an image

    Args:
        gray (numpy array): grayscale version of the image
        r_b (numpy array): normalized R-B ratio version of the image
        radius (int, optional): radius of the sky zone. Defaults to 390.

    Returns:
        list: 4 cloud-related features
    """
    h = gray.shape[0]
    # Perform the cloud segmentation
    seg = cloud_segmentation(gray, r_b)
    # Identify cloud pixels
    cloud_mask = np.where(seg==2., True, False).reshape(h, h)
    cloud_pixels = np.nansum(cloud_mask)
    # Compute the cloud ratio
    cloud_ratio = cloud_pixels/(np.pi*radius**2)
    if True not in cloud_mask:
        # In case of no cloud pixels
        cloud_disp = 0.
        cloud_brightness = 0.
        cloud_isoperimetric = 0.
    else:
        # Compute the cloud disparity (ratio between clouds' border pixels and inner cloud pixels)
        boundary_pixels = find_boundaries(cloud_mask)
        cloud_disp = 10*np.nansum(boundary_pixels)/cloud_pixels

        cloud_isoperimetric = cloud_isoperimetric_ratio(cloud_mask)

        # Compute the cloud brightness (mean value of cloud pixels)
        masked_gray = np.ma.array(gray, mask=~cloud_mask)
        cloud_brightness = masked_gray.mean()

    return cloud_ratio, cloud_disp, cloud_isoperimetric, cloud_brightness


