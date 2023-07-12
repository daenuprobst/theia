import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    y_rotation=0,
    group_lines=None,
    has_colorbar=True,
    title=None,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
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

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    data = np.nan_to_num(data)

    background = np.full((*data.shape, 3), 255, dtype=np.uint8)
    ax.imshow(background)

    # Plot the heatmap
    im = ax.imshow(
        data,
        interpolation="nearest",
        norm=LogNorm(),
        **kwargs,
    )

    # Create colorbar
    cbar = None
    if has_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    if title:
        ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    if y_rotation != 0:
        plt.setp(
            ax.get_xticklabels(),
            rotation=-y_rotation,
            ha="right",
            rotation_mode="anchor",
        )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle="-", linewidth=0)
    ax.tick_params(which="both", bottom=False, left=False, right=False, top=False)

    # Draw lines separating groups
    if group_lines:
        for line_pos in group_lines:
            l = ax.axvline(
                x=line_pos - 0.5, color="#555555", linestyle="-", linewidth=0.5
            )
            l.set_antialiased(False)
            l = ax.axhline(
                y=line_pos - 0.5, color="#555555", linestyle="-", linewidth=0.5
            )
            l.set_antialiased(False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    textcolors = ("black", "white")
    threshold = None

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = threshold
    else:
        threshold = data.max() / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0.0:
                kw.update(color=textcolors[int(data[i, j] > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None).lstrip("0"), **kw)
                texts.append(text)
            else:
                texts.append("")

    return texts
