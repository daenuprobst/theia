import os
import string
import pickle
import itertools
from typing import List, Optional
from collections import defaultdict
from pathlib import Path
from statistics import stdev
import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from scipy.spatial.distance import cosine as cosine_distance
from pycm import ConfusionMatrix

import seaborn as sns
import scienceplots

from heatmap_helpers import heatmap, annotate_heatmap
from helpers import index_subplots

plt.style.use(["science", "nature"])

SMALL_SIZE = 5
MEDIUM_SIZE = 7
BIGGER_SIZE = 10

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("figure", labelsize=SMALL_SIZE)  # fontsize of the figure title
plt.rc("figure", labelweight="normal")  # fontsize of the figure title
plt.rc("axes", titlesize=SMALL_SIZE + 0.01)

plt.rcParams.update({"mathtext.default": "regular"})


def main(
    cm_files: List[str], cols: int = 1, rows: int = 1, title: Optional[List[str]] = None
):
    fig, axs = plt.subplots(rows, cols, figsize=(3.54331, 2.65))

    if len(cm_files) != rows * cols:
        raise ValueError(
            f"Not enough rows and cols for the number of data files supplied. {len(cm_files)} files, {cols} columns, {rows} rows."
        )

    axs_flattened = []
    if rows * cols > 1:
        axs_flattened = axs.flat
    else:
        axs_flattened = [axs]

    pd.options.mode.chained_assignment = None

    modern_cmap = LinearSegmentedColormap.from_list(
        "modern_cmap", ["#ffffff", "#bc5090"], N=256
    )

    if title is None:
        title = []

    for i in range(len(cm_files) - len(title)):
        title.append("")

    for cm_file, ax, t in zip(cm_files, axs_flattened, title):
        cm: ConfusionMatrix = pickle.load(open(cm_file, "rb"))
        # print("Imbalance: ", cm.imbalance)
        # print("Recommendet params: ", cm.recommended_list)
        # print(cm.stat(summary=True))

        plt_cm = []
        for i in cm.classes:
            row = []
            for j in cm.classes:
                row.append(cm.table[i][j])
            plt_cm.append(row)

        plt_cm = np.array(plt_cm)
        plt_cm = plt_cm.astype("float") / plt_cm.sum(axis=1)[:, np.newaxis]

        # Create tick labels (group if too many)
        grouped_tick_labels = defaultdict(list)
        tick_labels = []
        for c in cm.classes:
            tick_labels.append("")
            grouped_tick_labels[str(c).split(".")[0]].append(c)

        group_lines = [0]
        offset = 0
        for key, value in grouped_tick_labels.items():
            tick_labels[offset + int(round(len(value) / 2.0))] = key
            offset += len(value)
            group_lines.append(offset)

        im, cbar = heatmap(
            plt_cm,
            tick_labels,
            tick_labels,
            ax=ax,
            cmap=modern_cmap,
            cbarlabel="",
            group_lines=group_lines,
            has_colorbar=False,
            title=t,
        )

        if len(cm.classes) < 10:
            texts = annotate_heatmap(
                im, data=plt_cm, valfmt="{x:.3f}", size=SMALL_SIZE - 1
            )

        # ax.set_xlabel("Predicted")
        # ax.set_ylabel("Ground Truth")

    index_subplots(axs.flat, font_size=MEDIUM_SIZE)
    fig.supxlabel("Predicted EC Class", x=0.54, y=0.05)
    fig.supylabel("Ground Truth EC Class", x=0.05, y=0.53)
    fig.tight_layout()

    plt.savefig(f"confusion-matrix.svg")
    plt.savefig(f"confusion-matrix.pdf")
    plt.savefig(f"confusion-matrix.png", dpi=300)


if __name__ == "__main__":
    typer.run(main)
