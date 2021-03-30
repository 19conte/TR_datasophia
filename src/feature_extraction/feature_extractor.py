import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import wiener
from skimage.filters import gaussian
from scipy.stats import skew
import mahotas
from skimage.measure import perimeter

from mask_and_zones import concentric_crops, square_mask, circumsolar_crops, sun_center
from cloud_seg import cloud_features
from preprocessing import path_to_canals


""" First let's define functions to extract statistical features from crops of images (zenith- or sun-centered crops)"""

def empty_stats(prefix, n_dicts=3):
    """generates n_dicts statistical features records filled with NaNs. This is used when, for whatever reason, it is not possible to 
        compute statistical features.
    Args:
        prefix (str): prefix which characterizes the type of crop
        n_dicts ([type]): number of NaNfilled-records to generate

    Returns:
        dict: n-dicts statistical records filled with NaNs
    """
    stats_dict = {}
    for k in range(n_dicts):
        empty_dict = {"_mean": np.nan, "_std": np.nan, "_smoothness": np.nan, 
            "_skewness": np.nan, "_entropy": np.nan}
        tmp_dict = empty_dict.copy()
        # Adapt dict keys
        for old_key in tmp_dict.keys():
            empty_dict[prefix + str(k) + old_key] = empty_dict.pop(old_key)
        stats_dict = {**stats_dict, **empty_dict}
    return stats_dict


def get_crop_statistical_features(img, prefix : str): # Use distorted grayscale images, outside val: NaN
    """Extract statistical features from an image crop

    Args:
        img (numpy array): grayscale image
        prefix (str): prefix which characterizes the type of image

    Returns:
        dict: statistical features of the given crop 
    """
    image = img[~np.isnan(img)]     
    # moment features
    _mean = np.mean(image) 
    _std = np.std(image)
    _smoothness = 1 - 1/(1 + np.var(image))
    _skewness = skew(image, axis=None)
    # entropy
    marg = np.histogramdd(np.ravel(image), bins = 256)[0]/image.size
    # We filter bins of the histogramm that are equal to 0
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    # Apply Schannon entropy formula
    _entropy = -np.sum(np.multiply(marg, np.log2(marg)))

    stats_features = {"_mean": _mean, "_std": _std, "_smoothness": _smoothness, 
                    "_skewness": _skewness, "_entropy": _entropy}
    tmp_dict = stats_features.copy()
    for old_key in tmp_dict.keys():
        stats_features[prefix + old_key] = stats_features.pop(old_key)

    return stats_features


def crops_to_features(crops, prefix):
    """extract features from a list of image's crops

    Args:
        crops (list): list of image's crops
        prefix (str): prefix which characterizes the type of image
    Returns:
        dict: dictionary containing statistical features of each crop
    """
    stats_dict = {}
    for k, crop in enumerate(crops):
        crop_stats = get_crop_statistical_features(crop, prefix=prefix + str(k))
        stats_dict = {**stats_dict, **crop_stats}
    return stats_dict


def get_statistical_features(gray):
    """Extract features from 3 versions of a grayscale image : original one, high-filtered one and low-filtered one.

    Args:
        gray (numpy array): grayscale image

    Returns:
        dict: statistical records for every crops of the 3 images
    """
    low, high = low_high_filter(gray)
    labels = ["gray", "gray_low_pass", "high_pass"]
    stats = {}
    for image, label in zip([gray, low, high], labels):
        crops = concentric_crops(image)
        concent_stats = crops_to_features(crops, prefix=label + "_concentric")
        circum_crops = circumsolar_crops(image, image)
        if circum_crops == None:
            circum_stats = empty_stats(prefix=label + "_circum")
        else:
            circum_stats = crops_to_features(circum_crops, prefix=label + "_circum")
        img_stats_features = {**concent_stats, **circum_stats}
        stats = {**stats, **img_stats_features}
    return stats


"""Then we extract spectral features from the power spectrum of an image"""


def get_energy_bins(image, num_bins=4):
    """compute the sum of pixel values in concentric square parts of the image. The first 
        square is dense, and the next are emptied in the center, forming concentric squares.

    Args:
        image (numpy array): image, must be square and with on nonegative values
        num_bins (int, optional): number of parts on which a sum is calculated. Defaults to 4.

    Returns:
        energy_bins (numpy array): energy repartition in each bin
        image_bins (list): crops of the input image 
    """
    h, _ = image.shape
    C = np.linspace(0, h//2, num=num_bins+1, dtype="int")
    # Create a list of dense square mask
    mask_list = [square_mask(image, c) for c in C[1:]]
    # Extract a list of hollow mask
    square_zones = [~mask_list[k]*mask_list[k+1] for k in range(len(mask_list)-1)]
    # add center square and whole image square to the list
    square_zones = [mask_list[0]] + square_zones + [mask_list[-1]]
    # Apply these masks to the image
    image_bins = []
    for zone in square_zones:
        img = image.copy()
        img[~zone] = np.nan
        image_bins.append(img)
    # Compute the energy of each masked image
    energy_bins = [np.nansum(_bin) for _bin in image_bins]
    total_energy = energy_bins[-1]
    # normalize the energy repartition
    energy_bins /= total_energy
    return energy_bins[:-1], image_bins[:-1]


def spectral_power(img, avg_window_size=None, log=True): #COMPLETE spectrum generator
    """ Compute spectrum for a distorted image
    """
    image = img.copy()
    # to avoid large spectral power at the 0 frequency :
    image -= np.mean(image)
    # wiener filter to reduce non physical variability in the spectral power
    if avg_window_size:
        N = avg_window_size
        image = wiener(image, (N, N))
    # compute the spectral power function. Place the 0 frequency-component in the center
    fshift = np.fft.fftshift(np.fft.fft2(image))
    spectrum = np.abs(fshift)**2
    if log:
        spectrum = 10*np.log(spectrum)
    return spectrum


def low_high_filter(image, param=1.6):
    """Computes the low-pass and high-pass filtered versions of an image

    Args:
        image (numpy array): grayscale image
        param (float, optional): filter intensity parameter. Defaults to 1.6.

    Returns:
        tuple of arrays: low- and high-filtered images
    """
    nan_pos = np.isnan(image)
    img = image.copy()
    mean = np.nanmean(img)
    img[nan_pos] = mean

    low_filtered_image = gaussian(img, sigma = 4)
    gau = gaussian(img, sigma = 4/param)
    high_filtered_image = gau - low_filtered_image

    low_filtered_image[nan_pos] = np.nan
    high_filtered_image[nan_pos] = np.nan
    return low_filtered_image, high_filtered_image


def get_spectral_features(image, img_type: str): # Apply on the whole image
    """Extract spectral features from an image

    Args:
        image (numpy array): an image
        img_type (str): string used as a prefix for the features' names. Characterize the type of image used as input.

    Returns:
        dict: dict of spectral features
    """
    spectrum = spectral_power(image, log=False)
    bins_energy, _ = get_energy_bins(spectrum, num_bins=3) # Use images with outside val: NaN
    spectral_features = {img_type + "low_freq_energy": bins_energy[0],
                        img_type + "mid_freq_energy": bins_energy[1], 
                        img_type + "high_freq_energy": bins_energy[2]}
    return spectral_features


"""We extract one feature related to the sun"""


def sun_isoperimetric_ratio(image, mode_viz=False): # On distorted image
    """compute the isoperimetric ratio of a sun disc. This ratio indicates the disc circularity


    Args:
        image (numpy array): grayscale image
        mode_viz (bool, optional): If True, displays the sun disc surface and border. Defaults to False.

    Returns:
        int: isoperimetric ratio of the sun disc
    """
    try :
        _, _, sun_mask = sun_center(image)
    except TypeError :
        return np.nan
    # We blurr the image and re-binarize it
    blurred_mask = mahotas.gaussian_filter(sun_mask, 0.7)
    blurred_mask = (blurred_mask > blurred_mask.mean())
    # Obtain a binary image with the sun border in white pixels
    sun_perim_mask = mahotas.labeled.bwperim(blurred_mask, 8)
    # Compute the perimeter in pixels
    sun_perim = int(perimeter(sun_perim_mask))
    # Compute the surface in pixels
    sun_surface = np.sum(blurred_mask)
    # ratio = 4*pi*S/(P^2). Is in [0,1], equals 1 for a circle
    ratio = 4*np.pi*sun_surface/(sun_perim**2)
    if mode_viz:
        # Plot
        # print(f"perimeter = {sun_perim} | surface = {sun_surface}")
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax1.imshow(blurred_mask, cmap="gray")
        ax2 = fig.add_subplot(122)
        ax2.imshow(sun_perim_mask, cmap="gray")
        plt.show()
    return np.round(ratio, 3)


def get_sun_features(image): # Use grayscale images, outside val: NaN
    """Extract sun-related features from an image

    Args:
        image (numpy array): an image

    Returns:
        dict: dict containing sun-related features
    """
    ratio = sun_isoperimetric_ratio(image)
    sun_features = {"sun_circularity_ratio": ratio}
    return sun_features


"""We use the cloud segmentation made in cloud_seg.py to extract cloud-related features"""


def get_cloud_features(gray, rb): # Use grayscale and r_b images, outside val : 0.
    """Extract cloud-related features from an image

    Args:
        gray (numpy array): grayscale version of the image
        rb (numpy array): normalized R-B ratio of the image

    Returns:
        dict: dict containing cloud-related features
    """
    c_r, c_d, cloud_iso, c_b = cloud_features(gray, rb)
    cloud_f = {"cloud_ratio": c_r, "cloud_disparity": c_d,
                        "cloud_isoperimetric_ratio":cloud_iso,
                        "cloud_brightness": c_b}    
    return cloud_f


"""We regroup these features in a dictionary, which will be useful to create dataframes"""


def path_to_feature_dict(path):
    """Create dict of features, from an image

    Args:
        path (Path or string): path of the original image

    Returns:
        dict: dictionary containing features values of the image
    """
    _, gray0, rb0 = path_to_canals(path, with_undistortion=False, outside_val=0.)
    _, grayN, rbN = path_to_canals(path, with_undistortion=False, outside_val=np.nan)

    stats_f = get_statistical_features(grayN)
    gray_spectral_f = get_spectral_features(gray0, img_type="gray_")
    r_b_spectral_f = get_spectral_features(rb0, img_type="rb_")
    sun_f = get_sun_features(grayN)
    cloud_f = get_cloud_features(grayN, rbN)

    features = {**stats_f, **gray_spectral_f, **r_b_spectral_f, **sun_f, **cloud_f}
    return features
    

# # Charge the image (cloudy sky)
# path = "../Solais_Data/FTP/2020-10-01/mobotix1/09_48_00.871.jpg"

# records = [path_to_feature_dict(path), path_to_feature_dict(path)]
# df = pd.DataFrame(records)
# print(df.columns)
