import segmentation
import tracking
import measurements
import numpy as np
import scipy.io as sio
import skimage.io as skio
import matplotlib.pyplot as plt

# seg = segmentation.segmentation("n6-n8/n6-n8/n6-n8.tif")
# skio.imsave("test_segmentation.tif", seg)

seg = skio.imread("test_segmentation.tif")

tracks = tracking.track(seg)
print(tracks.shape)

speeds = measurements.measure_speed(tracks)

goal = np.array([512/2, 512/2])
directedness = measurements.measure_directedness(tracks, goal)

mask_img = np.ones(seg[0].shape)
number = measurements.count_objects(tracks, mask_img)

measurements.set_plot_params()
measurements.line_plot(speeds, speeds, 30, ["group_1", "group_2"], "time", "speed")
# measurements.line_plot(directedness, directedness, 30, ["group_1", "group_2"], "time", "directedness")
measurements.tracks_plot(tracks, x_label="AP position (microns)", y_label="ML position (microns)", im_shape=[512, 512],
                         n_ticks=[4, 4])




