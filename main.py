import measurements
import numpy as np
import scipy.io as sio
import skimage.io as skio
import skimage.draw as sd
import pandas as pd
import os
from nuclei_count import NucleiCount

control_files = ["C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_1/max_project_1to18_C488D-rh.tif",
                 "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_2/max_project_1to19_C488D-rh.tif",
                 "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/h2o/20230326_embryo_3/max_project_1to18_C488D-rh.tif"]
exp_files = ["C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_1/max_project_1to19_C488D-rh.tif",
             "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_2/max_project_1to19_C488D-rh.tif",
             "C:/Users/Ray/Documents/MASc/BME1462/Project/macrophage_tracking/macrophage_nuclei/macrophage_nuclei/20230327-nlsgfp-50mMcbx/20230327_embryo_3/max_project_1to20_C488D-rh.tif"]

control_goals = [[236, 236], [306, 257], [287, 273]]
exp_goals = [[273, 267], [320, 300], [320, 326]]

ntp = 220  # number of timepoints in each movie
wound_slice = 40  # first slice index after laser is fired
dx = 0.267  # length of one pixel in microns
dt = 0.25  # time resolution in minutes
step = 10  # Number of slices to skip for trackpy measurements

# Measure speed and directedness in control and experiment videos using PIV and trackpy
control_speed = [[] for _ in control_files]
control_directedness = [[] for _ in control_files]
control_number = [[] for _ in control_files]
control_trackpy_speed = [[] for _ in control_files]
control_trackpy_directedness = [[] for _ in control_files]
control_trackpy_number = [[] for _ in control_files]

for f, file in enumerate(control_files):
    print(f"Processing control file {f + 1}")
    img = skio.imread(file)

    # Create mask that defines circle of radius 100 px around the wound
    mask_selem = sd.draw.disk(control_goals[f], 100, shape=img.shape[1:])
    mask = np.full(img.shape[1:], False)
    mask[mask_selem] = True

    # Perform trackpy measurements from manual annotations for comparison
    trackpy_csv = pd.read_csv(os.path.join(os.path.split(file)[0], 'manual_tracking.csv'))
    this_trackpy_displacement, control_trackpy_directedness[f], control_trackpy_number[f]\
        = measurements.trackpy_measurements(trackpy_csv, control_goals[f], mask)
    control_trackpy_speed[f] = np.array(this_trackpy_displacement)*dx/(step*dt)  # Trackpy is measured in steps of 10

    # Perform PIV measurements
    this_displacement, control_directedness[f] = measurements.piv_measurements(img, control_goals[f])
    control_number[f] = NucleiCount(img, mask)
    control_speed[f] = np.array(this_displacement)*dx/dt

exp_speed = [[] for _ in exp_files]
exp_directedness = [[] for _ in exp_files]
exp_number = [[] for _ in exp_files]
exp_trackpy_speed = [[] for _ in exp_files]
exp_trackpy_directedness = [[] for _ in exp_files]
exp_trackpy_number = [[] for _ in exp_files]

for f, file in enumerate(exp_files):
    print(f"Processing experiment file {f + 1}")
    img = skio.imread(file)

    # Create mask that defines circle of radius 100 px around the wound
    mask_selem = sd.draw.disk(exp_goals[f], 100, shape=img.shape[1:])
    mask = np.full(img.shape[1:], False)
    mask[mask_selem] = True

    # Perform trackpy measurements from manual annotations for comparison
    trackpy_csv = pd.read_csv(os.path.join(os.path.split(file)[0], 'manual_tracking.csv'))
    this_trackpy_displacement, exp_trackpy_directedness[f], exp_trackpy_number[f] \
        = measurements.trackpy_measurements(trackpy_csv, exp_goals[f], mask)
    exp_trackpy_speed[f] = np.array(this_trackpy_displacement) * dx / (step * dt)  # Trackpy is measured in steps of 10

    # Perform PIV measurements
    this_displacement, exp_directedness[f] = measurements.piv_measurements(img, exp_goals[f])
    exp_number[f] = NucleiCount(img, mask)
    exp_speed[f] = np.array(this_displacement) * dx / dt


# Save results using savemat so that tracking doesn't have to be run multiple times
dictio = {'control_speed': control_speed,
          'control_directedness': control_directedness,
          'control_number': control_number,
          'exp_speed': exp_speed,
          'exp_directedness': exp_directedness,
          'exp_number': exp_number
          }
sio.savemat('measurement_results.mat', dictio)

dictio = {'control_trackpy_speed': control_trackpy_speed,
          'control_trackpy_directedness': control_trackpy_directedness,
          'control_trackpy_number': control_trackpy_number,
          'exp_trackpy_speed': exp_trackpy_speed,
          'exp_trackpy_directedness': exp_trackpy_directedness,
          'exp_trackpy_number': exp_trackpy_number
          }
sio.savemat('manual_measurement_results.mat', dictio)








