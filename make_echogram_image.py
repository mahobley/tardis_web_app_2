import numpy as np
from scipy.signal import convolve


from PIL import Image
import matplotlib.pyplot as plt

import os


def numpy_to_redblue(array):
    """
    Maps a value from -1 to 1 to an RGB color gradient transitioning
    from red (-1) to white (0) to blue (1).

    Args:
        value (float): A number between -1 and 1.

    Returns:
        tuple: (R, G, B) values as integers in the range [0, 255].
    """
    # Ensure the value is clamped between -1 and 1
    # value = max(-1, min(1, value))
    cmapped_pos = np.stack(
        [255 * np.ones_like(array), 255 * (1 + array), 255 * (1 + array)], axis=-1
    )
    cmapped_neg = np.stack(
        [
            255 * (1 - array),
            255 * (1 - array) * 0.5 + 128 * np.ones_like(array),
            255 * np.ones_like(array),
        ],
        axis=-1,
    )
    array_stacked = np.stack([array, array, array], axis=-1)
    cmapped = np.where(
        array_stacked <= 0,
        cmapped_pos,
        cmapped_neg,
    )
    cmapped = cmapped.astype(np.float32) / 255

    cmapped = np.clip(
        cmapped,
        0,
        1,
    )
    return cmapped


def make_echogram_image(
    echograms,
    echogram_pop=False,
    filter_kernel=0,
    filter_tol=0.5,
    do_bgs=False,
    colour_power=3,
    colour_intensity_cutoff=0.0,
    colour_mask_power=1,
    given_bgr=False,
):
    """

    Args:
        echograms (numpy.ndarray): The exchogram data in [time, height, 2], the last dimension is the magnitude
                                of the echogram and the angle the max was found at
        echogram_pop (bool): whether to do background subtract (per row) on the echogram

    Returns:
        np.array: [time, height, 3] (R, G, B) values as integers in the range [0, 255] of the echogram,
                    with a colour scheme going from red to blue (left to right).
    """
    # Copy to writable float arrays so in-place normalization is always safe
    # even when input originates from read-only image buffers.
    if given_bgr:
        ec_mag = np.array(echograms[:, :, 2], dtype=np.float32, copy=True)  # magnitude
    else:
        ec_mag = np.array(echograms[:, :, 0], dtype=np.float32, copy=True)  # magnitude
    ec_angle = np.array(echograms[:, :, 1], dtype=np.float32, copy=True)  # angle

    ec_mag -= ec_mag.min()
    ec_mag_max = ec_mag.max()
    if ec_mag_max > 0:
        ec_mag /= ec_mag_max

    # find the areas of the echogram that are brighter than the average for that row
    if do_bgs:
        ec_mag_bgs = ec_mag - np.mean(ec_mag, axis=0)
        ec_mag_bgs[ec_mag_bgs < 0] = 0
        ec_mag_bgs_max = ec_mag_bgs.max()
        if ec_mag_bgs_max > 0:
            ec_mag_bgs /= ec_mag_bgs_max
    else:
        ec_mag_bgs = ec_mag
    colmapped = numpy_to_redblue(
        ec_angle * colour_power
    )  # the squared makes the colours more extreme
    if filter_kernel != 0:
        kernel = np.ones((filter_kernel, filter_kernel))
        ec_mag_bgs_bin = np.where(ec_mag_bgs > filter_tol, 1, 0)
        neighbourhood = convolve(ec_mag_bgs_bin, kernel, mode="same")
        neighbourhood_bin = np.where(neighbourhood > 2, 1, 0.5)
        ec_mag_bgs *= neighbourhood_bin
    if echogram_pop:
        # take the average subtracted echogram
        stacked_echogram_image = np.stack([ec_mag_bgs, ec_mag_bgs, ec_mag_bgs], axis=-1)
    else:
        # take the raw echogram
        stacked_echogram_image = np.stack([ec_mag, ec_mag, ec_mag], axis=-1)

    colour_mask = np.stack([ec_mag_bgs, ec_mag_bgs, ec_mag_bgs], axis=-1)
    if colour_intensity_cutoff > 0:
        colour_mask = np.where(colour_mask > colour_intensity_cutoff, 1, 0)
    colour_mask = colour_mask**colour_mask_power

    output_image = stacked_echogram_image * (1 - colour_mask) + colmapped * (
        colour_mask
    )
    # output_image = output_image.transpose(1, 0, 2)

    return output_image
