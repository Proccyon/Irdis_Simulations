#-----Header-----#
#This file contains some relevant functions for bad pixel correction from Irdap.
#The code will be used in Irdis and SCExAO calibration.
#--/--Header--/--#


#-----Imports-----#

import numpy as np
import scipy.stats as stats
from scipy import ndimage
#--/--Imports--/--#

#-----Functions-----#

def create_bpm_darks(frame_dark):
    '''
    Create a bad pixel map from DARK(,BACKGROUND)-files based on bias offsets and
    sigma filtering.
    Input:
        list_frame_dark: list of (mean-combined) DARK(,BACKGROUND)-frames
    Output:
        frame_bpm_dark: bad pixel map created from darks
    Function written by Rob van Holstein; constructed from functions by Christian Ginski
    Function status: verified
    '''

    # Create initial bad pixel map with only 1's
    frame_bpm_dark = np.ones(frame_dark.shape)


    # Remove outliers from dark frame and compute median and standard deviation
    frame_dark_cleaned = stats.sigmaclip(frame_dark, 5, 5)[0]
    stddev = np.nanstd(frame_dark_cleaned)
    median = np.nanmedian(frame_dark_cleaned)

    # Subtract median from dark frame and take absolute value
    frame_dark = np.abs(frame_dark - median)

    # Initialize a bad pixel array with 1 as default pixel value
    frame_bpm = np.ones(frame_dark.shape)

    # Set pixels that deviate by more than 3.5 sigma from the frame median value
    # to 0 to flag them as bad pixels
    frame_bpm[frame_dark > 3.5*stddev] = 0

    # Add bad pixels found to master bad pixel map
    frame_bpm_dark *= frame_bpm

    return frame_bpm_dark


#
def process_dark_flat_frames(DarkImage,FlatImage,ExpTime):
    '''
    Process DARK(,BACKGROUND)- and FLAT-files to create a master flat frame and
    a bad pix map. The number of dark and flat frames provided must be the same,
    and they must have matching exposure times. Generally a sequence of darks and
    flats with exposure times 1, 2, 3, 4, 5 s or 2, 4, 6, 8, 10 s is used. The bad
    pixel mask contains flags for all pixels that have a strong bias offset or that
    respond non-linearly.
    Input:
        path_dark_files: list of paths to raw DARK(,BACKGROUND)-files
        path_flat_files: list of paths to raw FLAT-files
        indices_to_remove_dark: list of 1-D arrays with indices of frames to
            be removed for each DARK(,BACKGROUND)-file. If no frames are to be
            removed the array is empty.
        indices_to_remove_flat: list of 1-D arrays with indices of frames to
            be removed for each FLAT-file. If no frames are to be removed the
            array is empty.
    Output:
        frame_master_flat: master flat frame
        frame_master_bpm: master bad pixel map (1 indicates good pixel;
                                                0 indicates bad pixel)
    File written by Christian Ginski; adapted by Rob van Holstein
    Function status: verified
    '''

     # Dark-subtract flat frame
    frame_flat_dark_subtracted = FlatImage - DarkImage

    # Filter dark-subtracted flat for zeros and NaN's
    frame_flat_dark_subtracted = np.nan_to_num(frame_flat_dark_subtracted)
    frame_flat_dark_subtracted[frame_flat_dark_subtracted <= 0] = 1


    frame_flat = frame_flat_dark_subtracted / ExpTime

    # Select the left and right detector area that actualy receives signal
    frame_flat_left = frame_flat[11:1024, 36:932]
    frame_flat_right = frame_flat[5:1018, 1062:1958]

    # Normalize left and right side of the flat with the respective median values
    frame_flat_left_norm = frame_flat_left / np.median(frame_flat_left)
    frame_flat_right_norm = frame_flat_right / np.median(frame_flat_right)

    # Create a baseline flat image that encompases left and right detector side
    # All values are 1, i.e. if applied this baseline flat does not influence the reduction
    # This is mainly to prevent later edge areas from containing blown-up data values
    frame_master_flat = np.ones(frame_flat.shape)

    # Construct the final full detector flat
    frame_master_flat[11:1024, 36:932] = frame_flat_left_norm
    frame_master_flat[5:1018, 1062:1958] = frame_flat_right_norm

    return frame_master_flat


def remove_bad_pixels(cube, frame_master_bpm, sigma_filtering=True):
    '''
    Remove bad pixels from an image cube or frame using the bad pixel map
    followed by optional repeated sigma-filtering
    Input:
        cube: image data cube or frame to filtered for bad pixels
        frame_master_bpm: frame indicating location of bad pixels with 0's and good
            pixels with 1's
        sigma_filtering: if True remove bad pixels remaining after applying
            master bad pixel map using repeated sigma-filtering (default = True)
    Output:
        cube_filtered: image data cube with bad pixels removed
    File written by Rob van Holstein
    Function status: verified
    '''

    cube_ndim = cube.ndim

    # Define size of side of kernel for median filter
    filter_size_median = 5

    # Round filter size up to nearest odd number for a symmetric filter kernel
    filter_size_median = 2*(filter_size_median // 2) + 1

    # Remove bad pixels using the bad pixel map
    cube_median = ndimage.filters.median_filter(cube, size=(1, filter_size_median, \
                                                            filter_size_median))
    cube_filtered = cube_median + frame_master_bpm * (cube - cube_median)

    if sigma_filtering == True:
        # Define threshold factor for sigma filtering
        factor_threshold = 5

        # Define maximum number of iterations and counters for while-loop
        maximum_iterations = 10
        number_pixels_replaced = 1
        iteration_counter = 0

        # Prepare weights to compute mean without central pixel using convolution
        filter_size = 7
        kernel = np.ones((1, filter_size, filter_size)) / (filter_size**2 - 9)
        for i in range(-1, 2):
            for j in range(-1, 2):
                kernel[0, filter_size//2 + i, filter_size//2 + j] = 0

        while number_pixels_replaced > 0 and iteration_counter < maximum_iterations:
            # Calculate local standard deviation using convolution
            cube_mean = ndimage.filters.convolve(cube_filtered, kernel)
            cube_EX2 = ndimage.filters.convolve(cube_filtered**2, kernel)
            cube_std = np.sqrt(cube_EX2 - cube_mean**2)

            # Compute threshold map for removal of pixels
            cube_threshold = factor_threshold*cube_std

            # Determine difference of image data with the local median
            cube_difference = np.abs(cube_filtered - cube_median)

            # Replace bad pixels by values in median filtered images
            pixels_to_replace = cube_difference > cube_threshold
            cube_filtered[pixels_to_replace] = cube_median[pixels_to_replace]

            # Compute number of pixels replaced and update number of iterations
            number_pixels_replaced = np.count_nonzero(pixels_to_replace)
            iteration_counter += 1
            print(str(iteration_counter)+":"+str(number_pixels_replaced))


    return cube_filtered

#--/--Functions--/--#

