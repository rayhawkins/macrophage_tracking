# Add code here for finding centroids of segmented nuclei and tracking nuclei through time
# Input is masks (3D binary images in x,y, and time) and output is nuclei tracks (n by t by 2 numpy arrays where n is 
# the number of macrophages, t is the number of timepoints, and 2 corresponds to (x,y) coordinates of that nucleus
# at that timepoint
