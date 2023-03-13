import matplotlib.pyplot as plt
import numpy
from skimage import io
from skimage import exposure, filters, feature, io, morphology, measure
from skimage.measure import label, regionprops

img=io.imread('n6-n8.tif')
segmented_image=segment(img)
# Add code here for finding centroids of segmented nuclei and tracking nuclei through time
# Input is masks (3D binary images in x,y, and time) and output is nuclei tracks (n by t by 2 numpy arrays where n is 
# the number of macrophages, t is the number of timepoints, and 2 corresponds to (x,y) coordinates of that nucleus
# at that timepoint

def track(segmented_image):
    
    centroids=[[] for _ in segmented_image[1]]
    for t, this_image in enumerate(segmented_image):
        labelled_image=label(this_image)
        regions=regionprops(labelled_image)
        for this_region in regions:
            centroids[t].append(this_regions.centroid)
    return centroids         
                   

result=track(img)
