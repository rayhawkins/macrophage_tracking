import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from tqdm import tqdm
import skimage.registration as skr
import trackpy as tp
from copy import deepcopy


def piv_measurements(img, goal):
    # compute the optical flow: for every slice in the stack
    nr, nc = img.shape[1:]
    v = np.empty([img.shape[0] - 1, nr, nc])
    u = np.empty([img.shape[0] - 1, nr, nc])

    speeds = [None for _ in range(img.shape[0] - 1)]
    directedness = [None for _ in range(img.shape[0] - 1)]
    for slice_index in tqdm(range(img.shape[0] - 1), ascii=True, desc='PIV measurements'):
        # compute the optical flow between the current and the next slice.
        v[slice_index], u[slice_index] = skr.optical_flow_ilk(img[slice_index], img[slice_index + 1], radius=15)
        flow_magnitude = np.sqrt(u[slice_index] ** 2 + v[slice_index] ** 2)

        # get speed by magnitude above low threshold
        speeds[slice_index] = np.mean(flow_magnitude[flow_magnitude > 0])

        # get directedness by dot product with goal
        this_directedness = []
        for r in range(nr):
            for c in range(nc):
                if flow_magnitude[r, c] < 0.5:
                    continue
                displacement_goal = [r - goal[0], c - goal[1]]
                u_ = u[slice_index, r, c]
                v_ = v[slice_index, r, c]
                dot_product = np.dot([v_, u_], displacement_goal)
                norms = np.linalg.norm([v_, u_]) * np.linalg.norm(displacement_goal)
                alpha = np.arccos(dot_product / norms)
                this_directedness.append(1 - abs((alpha - np.pi) / np.pi))
        directedness[slice_index] = np.nanmean(this_directedness)

    return speeds, directedness


def trackpy_measurements(csv, goal, mask=np.ones([512, 512])):
    tp.quiet()
    frames = np.unique(csv['frame'])
    counts = [0 for _ in range(frames.size)]

    t = tp.link(csv, 15, memory=11)
    t1 = tp.filter_stubs(t, 4)

    # tp.plot_traj(t1)

    speeds = [None for _ in range(frames.size - 1)]
    directedness = [None for _ in range(frames.size - 1)]

    for s, slice_num in tqdm(enumerate(frames[:-1]), ascii=True, desc='Trackpy measurements'):
        slice_index = slice_num//10
        these_particles = t1[t1['frame'] == slice_num]
        next_particles = t1[t1['frame'] == frames[s+1]]
        these_particle_ids = np.unique(these_particles['particle'])
        next_particle_ids = np.unique(next_particles['particle'])

        these_speeds = []
        these_directedness = []
        for this_particle_id in these_particle_ids:
            this_particle = these_particles[these_particles['particle'] == this_particle_id]
            this_coord = [this_particle.iloc[0]['y'], this_particle.iloc[0]['x']]
            if mask[this_coord[0], this_coord[1]]:
                counts[slice_index] += 1
            if this_particle_id in next_particle_ids:
                next_particle = next_particles[next_particles['particle'] == this_particle_id]
                next_coord = [next_particle.iloc[0]['y'], next_particle.iloc[0]['x']]
                displacement = np.sqrt((this_coord[0] - next_coord[0]) ** 2 + (this_coord[1] - next_coord[1]) ** 2)
                these_speeds.append(displacement)
                displacement_goal = [this_coord[0] - goal[0], this_coord[1] - goal[1]]

                dot_product = np.dot(displacement, displacement_goal)
                norms = np.linalg.norm(displacement) * np.linalg.norm(displacement_goal)
                alpha = np.arccos(dot_product / norms)
                these_directedness.append(1 - abs((alpha - np.pi) / np.pi))

        speeds[slice_index] = np.nanmean(these_speeds)
        directedness[slice_index] = np.nanmean(these_directedness)

    return speeds, directedness, counts


def set_plot_params():
    """
    Sets the global matplotlib parameters for pretty plotting.
    :return: True when completed.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.figsize'] = [9, 6]
    return True


def bar_plot(v1, v2, x_labels, y_label, title: str = None, method: str = None, n_colors: int = 2):
    """
    Plots a box plot comparing two groups, including the result of the statistical test specified by method.
    :param v1: n x 1 array containing values for group 1.
    :param v2: n x 1 array containing values for group 2.
    :param x_labels: list of two strings containing labels for group v1 and v2.
    :param y_label: string of label for the y-axis.
    :param title: optional string to set a title for the plot.
    :param method: statistical test to be used, MW for two-sided Mann-Whitney, None for no testing.
    :param n_colors: number of colors to be used when generating the palette.
    :return: True if completed.
    """

    if method is not None:
        if method == 'MW':  # Mann-whitney U-test
            s, p = stats.mannwhitneyu(v1, v2)
        elif method == 'T':  # Independent t-test
            s, p = stats.ttest_ind(v1, v2, equal_var=False)
        elif method == 'PT':  # Paired t-test
            s, p = stats.ttest_rel(v1, v2)

    labels = np.expand_dims(np.array([x_labels[0] for _ in v1] + [x_labels[1] for _ in v2]), axis=1)
    all_data = np.expand_dims(np.concatenate((v1, v2), axis=0), axis=1)
    df = pd.DataFrame(np.concatenate((all_data, labels), axis=1),
                      columns=['value', 'label'])
    df['value'] = df['value'].astype('float')
    a_palette = sns.color_palette('bright', n_colors)
    ax = sns.boxplot(x='label', y='value', data=df, hue='label', palette=a_palette, dodge=False, whis=100000)
    ax = sns.stripplot(x='label', y='value', data=df, color='k', alpha=0.75, dodge=False, jitter=0.05)
    if method is not None and p < 0.05:
        plt.axhline(y=np.max(all_data)*1.05, xmin=0.25, xmax=0.75, color='k')
        plt.scatter([0.5], [np.max(all_data)*1.07], marker='*', color='k')
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0)
    ax.legend().set_visible(False)
    if title is not None:
        plt.title(title)
    plt.show()

    print(p)
    return True


def line_plot(v1, v2, dt, wound_slice, group_labels, x_label, y_label, title: str = None):
    """
    Plots a box plot comparing two groups, including the result of the statistical test specified by method.
    :param v1: n x t array containing values for group 1.
    :param v2: n x t array containing values for group 2.
    :param dt: temporal resolution in seconds.
    :param wound_slice: timepoint at which wound occurs.
    :param group_labels: list of two strings containing labels for group v1 and v2.
    :param x_label: string of label for the x-axis.
    :param y_label: string of label for the y-axis.
    :param title: optional string to set a title for the plot.
    :return: True if completed.
    """
    a_palette = sns.color_palette('bright', 2)
    labels = np.expand_dims(np.array([group_labels[0] for _ in np.ravel(v1)] +
                                     [group_labels[1] for _ in np.ravel(v2)]), axis=1)
    all_data = np.expand_dims(np.ravel(np.concatenate((v1, v2), axis=0)), axis=1)
    timepoints = np.expand_dims(np.ravel(np.array([np.arange(-wound_slice*dt, dt*(v1.shape[1] - wound_slice), dt) for _ in v1] +
                                [np.arange(-wound_slice*dt, dt*(v1.shape[1] - wound_slice), dt) for _ in v2])), axis=1)
    df = pd.DataFrame(np.concatenate((all_data, timepoints, labels), axis=1),
                      columns=['value', 'time', 'label'])
    df['value'] = df['value'].astype('float')
    df['time'] = df['time'].astype('float')

    ax = sns.lineplot(x='time', y='value', hue='label', data=df, errorbar=('ci', 68), err_style='band',
                      lw=3, legend=False, palette=a_palette)
    plt.axvline(0, linestyle='--', c='k', alpha=0.7)
    ax.legend(labels=[group_labels[0], '_no_legend_', group_labels[1]], frameon=False)
    ax.set_xlim([timepoints[0], timepoints[-1]])
    # ax.set_ylim(bottom=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if title is not None:
        plt.title(title)
    plt.show()

    return True


def remove_outliers(arr, th):
    new_arr = deepcopy(arr)
    # Get rid of an outlier in control speed
    for t, this_experiment in enumerate(new_arr):
        for s, this_val in enumerate(this_experiment[:-1]):
            if this_val > th * this_experiment[s + 1]:
                new_arr[t][s] = this_experiment[s + 1]
    return new_arr


def normalize_plots(arr, slice_nums, method='mean'):
    new_arr = deepcopy(arr)
    for t, this_experiment in enumerate(new_arr):
        if method == 'mean':
            new_arr[t] = new_arr[t]/np.mean(new_arr[t][slice_nums[0]:slice_nums[1]])
        elif method == 'median':
            new_arr[t] = new_arr[t] / np.median(new_arr[t][slice_nums[0]:slice_nums[1]])
        elif method == 'subtract':
            new_arr[t] = new_arr[t] - int(np.mean(new_arr[t][slice_nums[0]:slice_nums[1]]))
    return new_arr
