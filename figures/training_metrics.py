import pickle
from typing import List, Optional
import typer
import scienceplots
from matplotlib import pyplot as plt
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
    data_files: List[str],
    cols: int = 1,
    rows: int = 1,
    title: Optional[List[str]] = None,
):
    fig, axs = plt.subplots(rows, cols)
    fig.set_figheight(3.54331)
    fig.set_figwidth(7.20472)

    if len(data_files) != rows * cols:
        raise ValueError(
            f"Not enough rows and cols for the number of data files supplied. {len(data_files)} files, {cols} columns, {rows} rows."
        )

    axs_flattened = []
    if rows * cols > 1:
        axs_flattened = axs.flat
    else:
        axs_flattened = [axs]

    # Pad titles
    for _ in range(rows * cols - len(title)):
        title.append("")

    for data_file, t, ax in zip(data_files, title, axs_flattened):
        metrics = pickle.load(open(data_file, "rb"))
        ax.plot(
            metrics["train_loss"],
            label="Training Loss",
            linestyle="--",
            color="#003f5c",
        )
        ax.plot(
            metrics["valid_loss"],
            label="Validation Loss",
            linestyle="-",
            color="#ffa600",
        )
        ax.set_title(t)
        ax.legend()

    fig.supxlabel("Epoch", x=0.5275, y=0.05)
    fig.supylabel("Loss", x=0.02, y=0.5)
    index_subplots(axs.flat, font_size=MEDIUM_SIZE)
    fig.tight_layout()

    plt.savefig(f"training-metrics.pdf")
    plt.savefig(f"training-metrics.svg")
    plt.savefig(f"training-metrics.png", dpi=300)


if __name__ == "__main__":
    typer.run(main)
