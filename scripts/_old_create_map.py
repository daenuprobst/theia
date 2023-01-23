import gzip
import pickle
from pathlib import Path
import tmap as tm
import pandas as pd
from drfp import DrfpEncoder
import matplotlib.pyplot as plt


def load_data() -> pd.DataFrame:
    df = pd.read_csv(Path(Path(__file__).resolve().parent, "../data/rheadb.csv.gz"))
    return df


def calculate_cache_drfp(df) -> pd.DataFrame:
    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/rhea-drfp.cache.pkl")
    if cache_file.exists():
        df = pickle.load(open(cache_file, "rb"))
    else:
        df["fps"] = DrfpEncoder.encode(df.rxn)
        pickle.dump(df, open(cache_file, "wb+"))

    return df


def main():
    df = load_data()
    df = calculate_cache_drfp(df)
    te = tm.embed(
        df.fps.to_numpy(),
        layout_generator=tm.layout_generators.AnnoyLayoutGenerator(
            node_size=1 / 50, k=50, kc=50, sl_repeats=6, sl_extra_scaling_steps=5
        ),
    )

    model_path = Path(Path(__file__).resolve().parent, "../models")
    tmap_file = Path(model_path, f"tmap.pkl.gz")

    pickle.dump(
        {
            "x": list(map(lambda n: float(n) * 700, te.x)),
            "y": list(map(lambda n: float(n) * 700, te.y)),
            "z": [0.0 for _ in range(len(te.x))],
            "s": list(map(int, te.s)),
            "t": list(map(int, te.t)),
        },
        gzip.open(tmap_file, "wb+"),
    )

    tmap_preview = Path(model_path, f"tmap_preview.png")
    plt.scatter(te.x, te.y, marker=".", s=1)
    plt.gca().set_aspect("equal")
    plt.savefig(tmap_preview)


if __name__ == "__main__":
    main()
