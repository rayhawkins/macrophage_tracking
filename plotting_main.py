import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import measurements
from copy import copy

fname = 'measurement_results.mat'
manual_fname = 'manual_measurement_results.mat'
dictio = sio.loadmat(fname)
manual_dictio = sio.loadmat(manual_fname)
measurements.set_plot_params()

dt = 0.25  # time resolution in seconds
wound_slice = 40  # slice after which laser is fired
step = 10  # number of slices skipped between manual annotations

# Load in results from tracking and remove some outliers, outliers are created in the PIV measurements by
# correlations between noise and the laser scar between the frames wound_slice and wound_slice + 1
control_speed = np.array(dictio['control_speed'])
control_speed = measurements.remove_outliers(control_speed, 1.5)  # remove outlier in control speeds
control_directedness = dictio['control_directedness']
control_number = dictio['control_number']
control_trackpy_speed = np.array(manual_dictio['control_trackpy_speed'])
control_trackpy_directedness = manual_dictio['control_trackpy_directedness']
control_trackpy_number = manual_dictio['control_trackpy_number']

exp_speed = np.array(dictio['exp_speed'])
exp_speed = measurements.remove_outliers(exp_speed, 1.5)  # remove outlier in experiment speeds
exp_directedness = dictio['exp_directedness']
exp_number = dictio['exp_number']
exp_trackpy_speed = np.array(manual_dictio['exp_trackpy_speed'])
exp_trackpy_directedness = manual_dictio['exp_trackpy_directedness']
exp_trackpy_number = manual_dictio['exp_trackpy_number']

# Plot mean speeds from PIV
labels = ['control', 'carbenoxelone']
control_speed_normalized = measurements.normalize_plots(control_speed, [0, int(wound_slice/2)], 'mean')
exp_speed_normalized = measurements.normalize_plots(exp_speed, [0, int(wound_slice/2)], 'mean')
# measurements.line_plot(control_speed_normalized, exp_speed_normalized, dt, wound_slice, labels,
#                        x_label='time (min)', y_label='normalized speed', title='PIV speed')

# Plot mean directedness from PIV
control_directedness_normalized = measurements.normalize_plots(control_directedness, [0, int(wound_slice/2)], 'mean')
exp_directedness_normalized = measurements.normalize_plots(exp_directedness, [0, int(wound_slice/2)], 'mean')
# measurements.line_plot(control_directedness_normalized, exp_directedness_normalized, dt, wound_slice, labels,
#                        x_label='time (min)', y_label='normalized directedness', title='PIV directedness')

# Plot mean nuclear counts from PIV
# measurements.line_plot(control_number, exp_number, dt, wound_slice, labels,
#                        x_label='time (min)', y_label='nuclear count')

# Plot mean speeds from trackpy
control_trackpy_speed_normalized = measurements.normalize_plots(control_trackpy_speed, [0, 2], 'mean')
exp_trackpy_speed_normalized = measurements.normalize_plots(exp_trackpy_speed, [0, 2], 'mean')
# measurements.line_plot(control_trackpy_speed_normalized, exp_trackpy_speed_normalized, dt*step, wound_slice//step, labels,
#                        x_label='time (min)', y_label='normalized speed', title='Manual speed')

# Plot mean directedness from trackpy
control_trackpy_directedness_normalized = measurements.normalize_plots(control_trackpy_directedness, [0, 2], 'mean')
exp_trackpy_directedness_normalized = measurements.normalize_plots(exp_trackpy_directedness, [0, 2], 'mean')
# measurements.line_plot(control_trackpy_directedness_normalized, exp_trackpy_directedness_normalized, dt*step, wound_slice//step, labels,
#                        x_label='time (min)', y_label='normalized directedness', title='Manual directedness')

# Plot mean nuclear counts from segmentation
measurements.line_plot(control_trackpy_number, exp_trackpy_number, dt*step, wound_slice//step, labels,
                       x_label='time (min)', y_label='nuclear count')

# Give bar plots for speed before and after in PIV controls
pre_labels = ["before-wounding", "after wounding"]
control_speed_before = [np.mean(this_speed[:wound_slice]) for this_speed in control_speed]
control_speed_after = [np.mean(this_speed[wound_slice:]) for this_speed in control_speed]
measurements.bar_plot(control_speed_before, control_speed_after, pre_labels, "speed (um/s)", method='PT')

# Give bar plots for change in speed before and after from PIV
control_delta_speed = [np.mean(this_speed[wound_slice:]) for this_speed in control_speed_normalized]
exp_delta_speed = [np.mean(this_speed[wound_slice:]) for this_speed in exp_speed_normalized]
# measurements.bar_plot(control_delta_speed, exp_delta_speed, labels, "% change in speed", method='T', title='Change in speed for PIV')

# Give bar plots for change in directedness before and after from PIV
control_delta_directedness = [np.nanmax(this_directedness[wound_slice:]) for this_directedness in control_directedness_normalized]
exp_delta_directedness = [np.nanmax(this_directedness[wound_slice:]) for this_directedness in exp_directedness_normalized]
# measurements.bar_plot(control_delta_directedness, exp_delta_directedness, labels, "% change in directedness", method='T', title='Change in directedness for PIV')

# Give bar plots for directedness before and after in PIV controls
pre_labels = ["before-wounding", "after wounding"]
control_directedness_before = [np.mean(this_directedness[:wound_slice]) for this_directedness in control_directedness]
control_directedness_after = [np.mean(this_directedness[wound_slice:]) for this_directedness in control_directedness]
measurements.bar_plot(control_directedness_before, control_directedness_after, pre_labels, "directedness", method='PT')

# Give bar plots for change in number before and after from PIV
control_delta_number = [np.nanmean(this_number[wound_slice//step:]) - np.nanmean(this_number[:wound_slice//step])
                        for this_number in control_number]
exp_delta_number = [np.nanmean(this_number[wound_slice//step:]) - np.nanmean(this_number[:wound_slice//step])
                    for this_number in exp_number]
# measurements.bar_plot(control_delta_number, exp_delta_number, labels, "change in nuclear count", method='T', title='Change in nuclear count for PIV')

# Give bar plots for speed before and after in PIV controls
pre_labels = ["before-wounding", "after wounding"]
control_count_before = [np.mean(this_count[:wound_slice]) for this_count in control_number]
control_count_after = [np.mean(this_count[wound_slice:]) for this_count in control_number]
measurements.bar_plot(control_count_before, control_count_after, pre_labels, "nuclear count", method='PT')

# Give bar plots for change in speed before and after from manual annotations
control_trackpy_delta_speed = [np.nanmean(this_speed[wound_slice//step:]) for this_speed in control_trackpy_speed_normalized]
exp_trackpy_delta_speed = [np.nanmean(this_speed[wound_slice//step:]) for this_speed in exp_trackpy_speed_normalized]
measurements.bar_plot(control_trackpy_delta_speed, exp_trackpy_delta_speed, labels, "% change in speed", method='T')

# Give bar plots for change in directedness before and after from manual annotations
control_trackpy_delta_directedness = [np.nanmax(this_directedness[wound_slice//step:]) for this_directedness in control_trackpy_directedness_normalized]
exp_trackpy_delta_directedness = [np.nanmax(this_directedness[wound_slice//step:]) for this_directedness in exp_trackpy_directedness_normalized]
measurements.bar_plot(control_trackpy_delta_directedness, exp_trackpy_delta_directedness, labels, "% change in directedness", method='T')

# Give bar plots for change in number before and after from manual annotations
control_trackpy_delta_number = [np.nanmean(this_number[wound_slice//step:]) - np.nanmean(this_number[:wound_slice//step])
                                for this_number in control_trackpy_number]
exp_trackpy_delta_number = [np.nanmean(this_number[wound_slice//step:]) - np.nanmean(this_number[:wound_slice//step])
                            for this_number in exp_trackpy_number]
measurements.bar_plot(control_trackpy_delta_number, exp_trackpy_delta_number, labels, "change in nuclear count", method='T', title='Change in nuclear count for manual annotations')