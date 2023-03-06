import segmentation
import tracking
import measurements
import numpy as np
import scipy.io as sio

tracks = sio.loadmat('test_tracks.mat')
tracks = tracks.get('tracks')

goal = np.array([512/2, 512/2])
mask_img = np.ones([512, 512])
speeds = measurements.measure_speed(tracks)
directedness = measurements.measure_directedness(tracks, goal)
print(directedness)
number = measurements.count_objects(tracks, mask_img)

# measurements.line_plot(speeds, speeds, 30, ["group_1", "group_2"], "time", "speed")
measurements.line_plot(directedness, directedness, 30, ["group_1", "group_2"], "time", "directedness")


