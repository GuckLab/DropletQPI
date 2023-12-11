import skimage.filters as fil

def flatten_background(image, sigma=25):
    """
    Function to flatten the background for image segmentation employing a gaussian
    filter. The absolute intensity values of the output image are changed with respect
    to the inout.
    Parameters
    ----------
    image: Input image
    sigma: Standard deviation for Gaussian kernel

    Returns
    -------
    Difference of the input image and the blurred image, resulting in a smoothened
    background
    """
    blurred = fil.gaussian(image, sigma=25)
    return image - blurred

def filter_features(region_props):
    pass