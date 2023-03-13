# Add code here for determining speed, directedness, number, etc. from tracked macrophages
# Input is nuclei tracks (n by t by 2 numpy arrays where n is the number of macrophages, t is the number of timepoints, 
# and 2 corresponds to (x,y) coordinates of that nucleus at that timepoint
# Output is measurements in arrays and plots that allow visualization of the measured parameters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


def count_objects(tracks, mask):
    """
    Counts the number of objects given in tracks that are within the region defined by mask for each timepoint.
    :param tracks: n by t by 2 array specifying coordinates of n objects at timepoints t.
    :param mask: binary image specifying region within which counting will occur.
    :return: 1 x t array containing number of objects within the region at each timepoint.
    """

    counts = np.zeros([tracks.shape[1]])
    for this_nucleus in tracks:
        for t, this_timepoint in enumerate(this_nucleus):
            if this_timepoint is None:
                continue
            if mask[int(this_timepoint[0]), int(this_timepoint[1])] == 1:
                counts[t] += 1

    return counts


def measure_speed(tracks, dt: float = 30., dx: float = 0.178):
    """
    Measures speed as the pythagorean distance travelled in one timepoint divided by the time resolution.
    :param tracks: n by t by 2 array specifying coordinates of n objects at timepoints t.
    :param dt: time resolution in seconds.
    :param dx: spatial resolution in microns/pixel.
    :return: n x t array containing object speed at each timepoint in microns/second.
    """
    speeds = np.zeros([tracks.shape[0], tracks.shape[1] - 1])
    for n, this_nucleus in enumerate(tracks):
        for t, this_timepoint in enumerate(this_nucleus[:-1]):
            if this_timepoint is None or this_nucleus[t + 1] is None:
                speeds[n, t] = None
                continue
            delta_x = (this_timepoint[1] - this_nucleus[t + 1][1]) * dx
            delta_y = this_timepoint[0] - this_nucleus[t + 1][0] * dx
            distance = np.sqrt(delta_x**2 + delta_y**2)
            speeds[n, t] = distance / dt

    return speeds


def measure_directedness(tracks, goals: np.ndarray):
    """
    Measures the difference in angle between the direction of motion of the objects defined in tracks and the
    vector that points from the object's location to the coordinate defined by goal.
    Normalizes the difference in angle
    by abs[angle (in rad) - pi] / pi so that a degree difference of 180 returns a directedness of 0 and a degree
    difference of 0, 360 returns a directedness of 1.
    :param tracks: n by t by 2 array specifying coordinates of n objects at timepoints t.
    :param goals: n by 2 array specifying (x,y) goals for each object, or tuple defining one goal for all objects.
    :return: directedness measure for each object at each timepoint.
    """
    directedness = np.zeros([tracks.shape[0], tracks.shape[1] - 1])
    for n, this_nucleus in enumerate(tracks):
        for t, this_timepoint in enumerate(this_nucleus[:-1]):
            if this_timepoint is None or this_nucleus[t + 1] is None:
                directedness[n, t] = None
                continue
            displacement = [this_timepoint[1] - this_nucleus[t + 1][1], this_timepoint[0] - this_nucleus[t + 1][0]]
            if goals.ndim == 2:  # Different goal for each object
                displacement_goal = [goals[n, 1] - this_timepoint[1], goals[n, 0] - this_timepoint[0]]
            elif goals.ndim == 1:  # Same goal for each object
                displacement_goal = [goals[1] - this_timepoint[1], goals[0] - this_timepoint[0]]
            dot_product = np.dot(displacement, displacement_goal)
            norms = np.linalg.norm(displacement)*np.linalg.norm(displacement_goal)
            alpha = np.arccos(dot_product/norms)
            directedness[n, t] = abs((alpha - np.pi) / np.pi)

    return directedness


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


def bar_plot(v1, v2, x_labels, y_label, title: str = None, method: str = None):
    """
    Plots a box plot comparing two groups, including the result of the statistical test specified by method.
    :param v1: n x 1 array containing values for group 1.
    :param v2: n x 1 array containing values for group 2.
    :param x_labels: list of two strings containing labels for group v1 and v2.
    :param y_label: string of label for the y-axis.
    :param title: optional string to set a title for the plot.
    :param method: statistical test to be used, MW for two-sided Mann-Whitney, None for no testing.
    :return: True if completed.
    """

    if method is not None:
        if method == 'MW':
            s, p = stats.mannwhitneyu(v1, v2)

    labels = np.expand_dims(np.array([x_labels[0] for _ in v1] + [x_labels[1] for _ in v2]), axis=1)
    all_data = np.expand_dims(np.concatenate((v1, v2), axis=0), axis=1)
    df = pd.DataFrame(np.concatenate((all_data, labels), axis=1),
                      columns=['value', 'label'])
    df['value'] = df['value'].astype('float')
    a_palette = sns.color_palette('bright', 2)
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

    return True


def line_plot(v1, v2, dt, group_labels, x_label, y_label, title: str = None):
    """
    Plots a box plot comparing two groups, including the result of the statistical test specified by method.
    :param v1: n x t array containing values for group 1.
    :param v2: n x t array containing values for group 2.
    :param dt: temporal resolution in seconds.
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
    timepoints = np.expand_dims(np.ravel(np.array([np.arange(0, dt*v1.shape[1], dt) for _ in v1] +
                                [np.arange(0, dt*v1.shape[1], dt) for _ in v2])), axis=1)
    df = pd.DataFrame(np.concatenate((all_data, timepoints, labels), axis=1),
                      columns=['value', 'time', 'label'])
    df['value'] = df['value'].astype('float')
    df['time'] = df['time'].astype('float')

    ax = sns.lineplot(x='time', y='value', hue='label', data=df, ci=68, err_style='band',
                      lw=3, legend=False, palette=a_palette)
    ax.legend(labels=[group_labels[0], '_no_legend_', group_labels[1]], frameon=False)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if title is not None:
        plt.title(title)
    plt.show()

    return True


def tracks_plot(tracks, dx: float = 0.178, im_shape: tuple = None, x_label: str = None, y_label: str = None,
                title: str = None, n_ticks: tuple = None):
    """
    Plots the position of centroids specified by tracks over time.
    :param tracks: n by t by 2 array specifying coordinates of n objects at timepoints t.
    :param dx: spatial resolution in microns.
    :param im_shape: tuple with image shape in (rows, cols) to set axes limits.
    :param x_label: string to be used as x-label.
    :param y_label: string to be used as y-label.
    :param title: string to be used for plot title.
    :param n_ticks: tuple with (number of y_ticks, number of x_ticks).
    :return: True when completed.
    """

    palette = plt.colormaps.get('jet')
    colors = palette(np.linspace(0, 1, tracks.shape[0]))  # Different color for each object
    fig, ax = plt.subplots(1, 1)
    for n, this_nucleus in enumerate(tracks):
        for t, this_timepoint in enumerate(this_nucleus):
            if this_timepoint is None:
                continue
            else:
                ax.scatter([this_timepoint[0]], [this_timepoint[1]], c=[colors[n]], s=30)

    xticks = np.arange(0, im_shape[1] * dx + dx, im_shape[1] * dx / n_ticks[1])
    yticks = np.arange(0, im_shape[0] * dx + dx, im_shape[1] * dx / n_ticks[0])
    ax.set_xlim([0, im_shape[1]])
    ax.set_ylim([0, im_shape[0]])
    ax.set_xticks(np.linspace(0, im_shape[1], n_ticks[1] + 1), labels=[str(int(x)) for x in xticks])
    ax.set_yticks(np.linspace(0, im_shape[0], n_ticks[0] + 1), labels=[str(int(y)) for y in yticks])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.title(title)
    plt.show()
    return True
