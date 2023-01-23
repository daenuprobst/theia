import typer
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import tmap as tm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from helpers import index_subplots

plt.style.use(["science", "nature"])

SMALL_SIZE = 5
MEDIUM_SIZE = 7
BIGGER_SIZE = 10

plt.rc("font", size=MEDIUM_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE - 1)
plt.rc("figure", titlesize=BIGGER_SIZE)
plt.rc("figure", labelsize=SMALL_SIZE)
plt.rc("figure", labelweight="normal")
plt.rc("patch", linewidth=0.5)


def main(input_path: str):
    fig = plt.figure(figsize=(3.54331, 1.5 * 3.54331))

    axs = []
    gs = GridSpec(3, 2, figure=fig)
    axs.append(fig.add_subplot(gs[:-1, :]))
    axs.append(fig.add_subplot(gs[2, 0]))
    axs.append(fig.add_subplot(gs[2, 1]))

    modern_cmap_3 = ListedColormap(
        [
            "#e60049",
            "#0bb4ff",
            "#50e991",
        ]
    )

    modern_cmap_6 = ListedColormap(
        [
            "#e60049",
            "#0bb4ff",
            "#50e991",
            "#e6d800",
            "#9b19f5",
            "#ffa300",
        ]
    )

    modern_cmap_9 = ListedColormap(
        [
            "#e60049",
            "#0bb4ff",
            "#50e991",
            "#e6d800",
            "#9b19f5",
            "#ffa300",
            "#dc0ab4",
            "#b3d4ff",
            "#00bfa0",
        ],
    )

    # # EC classes
    df_train = pd.read_csv(f"{input_path}/rheadb-0-ec1-train.csv")
    df_valid = pd.read_csv(f"{input_path}/rheadb-0-ec1-valid.csv")
    df_test = pd.read_csv(f"{input_path}/rheadb-0-ec1-test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    df = df.sort_values(by=["label"])

    le = LabelEncoder()
    df.label = le.fit_transform(df.label)

    X = df.fps.apply(lambda x: np.array(list(map(int, x.split(";"))))).to_numpy()
    y = df.label.tolist()

    te = tm.embed(
        X,
        layout_generator=tm.layout_generators.AnnoyLayoutGenerator(
            node_size=3, sl_repeats=2, mmm_repeats=2, n_trees=50
        ),
        keep_knn=True,
    )

    tm.plot(
        te,
        show=False,
        line_kws={"linestyle": "--", "color": "gray", "linewidth": 0.5},
        scatter_kws={"s": 0.25, "c": y, "cmap": modern_cmap_6},
        ax=axs[0],
    )

    legend_elements = axs[0].collections[0].legend_elements()

    legend = axs[0].legend(
        legend_elements[0],
        ["EC " + str(c) for c in le.classes_],
        loc="lower left",
        ncols=len(le.classes_),
        handletextpad=0.0,
        columnspacing=0.1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    axs[0].add_artist(legend)
    axs[0].tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        top=False,
        which="both",
    )

    # Detail of EC 2.x
    df_train = pd.read_csv(f"{input_path}/rheadb-0-ec12-train.csv")
    df_valid = pd.read_csv(f"{input_path}/rheadb-0-ec12-valid.csv")
    df_test = pd.read_csv(f"{input_path}/rheadb-0-ec12-test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    df = df.sort_values(by=["label"])

    df.label = df.label.astype(str)
    df = df[df.label.str.startswith("2.")]

    le = LabelEncoder()
    df.label = le.fit_transform(df.label)

    X = df.fps.apply(lambda x: np.array(list(map(int, x.split(";"))))).to_numpy()
    y = df.label.to_numpy()

    te = tm.embed(
        X,
        layout_generator=tm.layout_generators.AnnoyLayoutGenerator(
            node_size=2, sl_repeats=2, mmm_repeats=2, n_trees=50
        ),
        keep_knn=True,
    )

    tm.plot(
        te,
        show=False,
        line_kws={"linestyle": "--", "color": "gray", "linewidth": 0.5},
        scatter_kws={"s": 0.25, "c": y, "cmap": modern_cmap_9},
        ax=axs[1],
    )

    legend_elements = axs[1].collections[0].legend_elements()

    legend = axs[1].legend(
        legend_elements[0],
        ["EC " + str(c) for c in le.classes_],
        loc="lower left",
        ncols=5,
        handletextpad=0.0,
        columnspacing=0.1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    axs[1].add_artist(legend)
    axs[1].tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        top=False,
        which="both",
    )

    # Detail of EC 2.3.
    df_train = pd.read_csv(f"{input_path}/rheadb-0-ec123-train.csv")
    df_valid = pd.read_csv(f"{input_path}/rheadb-0-ec123-valid.csv")
    df_test = pd.read_csv(f"{input_path}/rheadb-0-ec123-test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    df = df.sort_values(by=["label"])

    df.label = df.label.astype(str)
    df = df[df.label.str.startswith("2.4.")]

    print(df.groupby(by=["label"]).count())

    le = LabelEncoder()
    df.label = le.fit_transform(df.label)

    X = df.fps.apply(lambda x: np.array(list(map(int, x.split(";"))))).to_numpy()
    y = df.label.to_numpy()

    te = tm.embed(
        X,
        layout_generator=tm.layout_generators.AnnoyLayoutGenerator(
            node_size=10, sl_repeats=2, mmm_repeats=2, n_trees=50
        ),
        keep_knn=True,
    )

    tm.plot(
        te,
        show=False,
        line_kws={"linestyle": "--", "color": "gray", "linewidth": 0.5},
        scatter_kws={"s": 1, "c": y, "cmap": modern_cmap_3},
        ax=axs[2],
    )

    legend_elements = axs[2].collections[0].legend_elements()

    legend = axs[2].legend(
        legend_elements[0],
        ["EC " + str(c) for c in le.classes_],
        loc="lower left",
        ncols=4,
        handletextpad=0.0,
        columnspacing=0.1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    axs[2].add_artist(legend)
    axs[2].tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
        top=False,
        which="both",
    )

    index_subplots(axs, font_size=MEDIUM_SIZE, x=0, y=1.02)
    fig.tight_layout()

    plt.savefig(f"tmap.svg")
    plt.savefig(f"tmap.pdf")
    plt.savefig(f"tmap.png", dpi=300)


if __name__ == "__main__":
    typer.run(main)
