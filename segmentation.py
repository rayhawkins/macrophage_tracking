# Importing all necessary toolboxes
import matplotlib.pyplot as plt
import numpy
from skimage import io, filters
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_opening, disk


# Defining segmentation function 
# Input is raw images (4D grayscale images in x,y,z, and time), output is masks (3D binary images in x,y, and time)
def segmentation(im_raw):
    # Initialize the 3D array for the binarized images
    im_binarize = []
    
    # Read in raw 4D grayscale images
    im_4d = io.imread(im_raw)
    
    # Obtaining Maximum Intensity Projection (MIP) images
    im_mip = numpy.max(im_4d, axis=1)
    im_range = len(im_mip)
    
    # Initialize the structuring element for binary erosion (to separate the elements)
    se = numpy.zeros((9, 9), dtype=numpy.int16) # Creating a 9x9 square of 0s
    se[4, :] = 1
    
    # Creating loop to binarize the images
    for t in range(0,im_range):
        # Segmentation of original image using Otsu's Method
        t_otsu = threshold_otsu(im_mip[t])
        mask = im_mip[t] > t_otsu
        
        # Performing binary erosion on the mask to separate the nuclei
        im_bin_erode = binary_opening(mask, footprint=disk(5))
        
        im_binarize.append(im_bin_erode)
        
    return numpy.array(im_binarize)  # Return the 3D array of binarized images




