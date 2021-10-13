import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# file taken from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
from matplotlib import colors

# https://stackoverflow.com/questions/50192121/custom-color-palette-intervals-in-seaborn-heatmap
from matplotlib.colors import LinearSegmentedColormap


def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb = matplotlib.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict


def heatmap(data, row_label, col_label,cm,
            start_e,
            vmin, vmax, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.matshow(data,
                    cmap=cm,
                    # , norm=colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
                    )

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks((0, data.shape[1] - 1))
    filter_center = 2.5
    ax.set_xticklabels((2 + filter_center, data.shape[1] + 2 + filter_center))
    plt.xlabel(row_label)

    ax.set_yticks((0, data.shape[0] - 1))
    ax.set_yticklabels((start_e + data.shape[0] + filter_center, start_e + filter_center))
    plt.ylabel(col_label)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
