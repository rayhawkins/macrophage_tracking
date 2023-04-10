from skimage import io, filters
from skimage.morphology import disk, opening
from skimage.measure import regionprops, label


def NucleiCount(raw, mask):
    """
    Counts the number of macrophage nuclei that are within the region defined by mask for each timepoint.
    :param raw: t by rows by columns image array specifying coordinates of images at timepoints t.
    :param mask: binary image specifying region within which counting will occur.
    :return: t x (nuclei count) array containing number of nuclei within the region at each timepoint.
    """
    
    # Initializing variables
    nuclei_count = []

    raw_range = len(raw)
    
    # Initializing the structuring element
    raw_se = disk(3)
    
    # Loop to process the images and count the number of nuclei
    for t in range(0, raw_range):
        # Initializing local variable to count number of nuclei for each image
        count = 0
        
        # Gaussian smoothing to remove noise
        raw_smooth = filters.gaussian(raw[t], sigma=2, preserve_range=True)

        # Opening to break gaps between nuclei
        raw_open = opening(raw_smooth, raw_se)
        
        # Using the Sobel edge detection filter to isolate the nuclei
        raw_sobel = filters.sobel(raw_open)
        
        # Segmenting the Sobel images using Li's Method
        nuclei_li = raw_sobel > filters.threshold_li(raw_sobel)
        
        #Applying the mask passed to the function to the segmented images to isolate an area of interest 
        nuclei_li_masked = nuclei_li * mask

        # Labelling the segmented images to get region measurements
        labelled_nuclei = label(nuclei_li_masked)
        nuclei_regions = regionprops(labelled_nuclei)
            
        # Nested loop to count the number of nuclei
        for this_region in nuclei_regions:
            if this_region.area > 100 and this_region.area < 800: # Counts single nuclei
                count += 1
            elif this_region.area >= 800 and this_region.area < 1000: # Anticipates nuclei are overlapping and counts 2
                count += 2
            elif this_region.area >= 1000: # Anticipates nuclei are overlapping and counts 3
                count += 3
        nuclei_count.append(count)
        
    return nuclei_count





