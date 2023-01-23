import pickle
from typing import List, Optional
import typer
import scienceplots
import pandas as pd
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix
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
    count_file: str,
    cm_files: List[str],
    cols: int = 1,
    rows: int = 1,
    outname: Optional[str] = "accuracy_vs_size",
    title: Optional[List[str]] = None,
):
    df_counts = pd.read_csv(count_file).set_index("ec")

    fig, axs = plt.subplots(
        rows, cols, figsize=(3.54331, 3.54331), sharex=False, sharey=True
    )

    if len(cm_files) != rows * cols:
        raise ValueError(
            f"Not enough rows and cols for the number of data files supplied. {len(cm_files)} files, {cols} columns, {rows} rows."
        )

    axs_flattened = []
    if rows * cols > 1:
        axs_flattened = axs.flat
    else:
        axs_flattened = [axs]

    # Pad titles
    for _ in range(rows * cols - len(title)):
        title.append("")

    data = {}

    for i in [1, 2, 3]:
        data[i] = {"f1_score": [], "n": []}

    for cm_file in cm_files:
        cm: ConfusionMatrix = pickle.load(open(cm_file, "rb"))
        for label, f1_score in cm.F1.items():
            level = df_counts.at[label, "level"]
            data[level]["f1_score"].append(f1_score)
            data[level]["n"].append(df_counts.at[label, "count"])

    for data, t, ax in zip(data.values(), title, axs_flattened):
        # ax.boxplot(
        #     f1_score,
        #     positions=[1, 2, 3, 4, 5, 6],
        #     widths=0.75,
        #     patch_artist=True,
        #     showmeans=True,
        #     flierprops={
        #         "marker": "D",
        #         "markersize": 2.0,
        #         "markerfacecolor": "black",
        #         "markeredgewidth": 0.0,
        #         "alpha": 0.25,
        #     },
        #     medianprops={"color": "#bc5090", "linewidth": 1.0},
        #     meanprops={
        #         "color": "#ffa600",
        #         "marker": "o",
        #         "markersize": 2.0,
        #     },
        #     boxprops={
        #         "facecolor": "white",
        #         "edgecolor": "black",
        #         "linewidth": 0.5,
        #     },
        #     whiskerprops={"color": "black", "linewidth": 0.5},
        #     capprops={"color": "black", "linewidth": 0.5},
        # )
        ax.scatter(
            data["n"],
            data["f1_score"],
            color="black",
            s=[2.5] * len(data["n"]),
            alpha=0.5,
            linewidths=0.0,
        )

        ax.set_title(t)

        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
        )
        ax.set_axisbelow(True)
        # ax.set_yticks([0, 0.5, 1], ["0.0", "0.5", "1.0"])
        # ax.set_xticks([1, 2, 3, 4, 5, 6], ["1", "2", "3", "4", "5", "6"])
        ax.yaxis.grid(True, which="both", color="#cccccc", linestyle="dashed")
        ax.yaxis.grid(True, which="minor", color="#eeeeee", linestyle="dashed")

    index_subplots(axs.flat, font_size=MEDIUM_SIZE)
    fig.supxlabel("Sample Count", x=0.544, y=0.05)
    fig.supylabel("F1 Score", x=0.05, y=0.515)
    fig.tight_layout()

    plt.savefig(f"{outname}.svg")
    plt.savefig(f"{outname}.pdf")
    plt.savefig(f"{outname}.png", dpi=300)

    # for data_file, t, ax in zip(data_files, title, axs_flattened):
    #     metrics = pickle.load(open(data_file, "rb"))
    #     ax.plot(
    #         metrics["train_loss"],
    #         label="Training Loss",
    #         linestyle="--",
    #         color="#003f5c",
    #     )
    #     ax.plot(
    #         metrics["valid_loss"],
    #         label="Validation Loss",
    #         linestyle="-",
    #         color="#ffa600",
    #     )
    #     ax.set_title(t)
    #     ax.legend()

    # fig.tight_layout()

    # plt.savefig(f"training-metrics.svg", dpi=300)
    # plt.savefig(f"training-metrics.png", dpi=300)


if __name__ == "__main__":
    typer.run(main)
