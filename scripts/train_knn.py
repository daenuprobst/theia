import pickle
import gzip
from pathlib import Path

import pandas as pd
from drfp import DrfpEncoder
from annoy import AnnoyIndex
from tqdm import tqdm


def load_data() -> pd.DataFrame:
    df = pd.read_csv(Path(Path(__file__).resolve().parent, "../data/rheadb.csv.gz"))
    return df


def calculate_cache_drfp(df) -> pd.DataFrame:
    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/rhea-drfp.cache.pkl")
    if cache_file.exists():
        df = pickle.load(open(cache_file, "rb"))
    else:
        df["fps"] = DrfpEncoder.encode(
            df.rxn,
            root_central_atom=False,
            radius=2,
            include_hydrogens=True,
            n_folded_length=2048,
            show_progress_bar=True,
        )
        pickle.dump(df, open(cache_file, "wb+"))

    return df


def train_model(df) -> AnnoyIndex:
    # 2048 is the dimensionality of the fingerprint
    t = AnnoyIndex(2048, "angular")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        t.add_item(row.id, row.fps)

    t.build(10)
    return t


def main():
    model_path = Path(Path(__file__).resolve().parent, "../models")
    model_file = Path(model_path, f"rhea-drfp.ann")
    model_file_compressed = Path(model_path, f"rhea-drfp.ann.gz")
    df = load_data()
    df = calculate_cache_drfp(df)
    t = train_model(df)
    t.save(str(model_file))

    # Zip the model as it tends to be a huge memmap
    with open(model_file, "rb") as f_in:
        with gzip.open(model_file_compressed, "wb") as f_out:
            f_out.writelines(f_in)

    model_file.unlink()


if __name__ == "__main__":
    main()
