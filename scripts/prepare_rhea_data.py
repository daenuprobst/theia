import gzip
from pathlib import Path
import pandas as pd


def parse_rhea_reactions() -> pd.DataFrame:
    file_path = Path(__file__).resolve().parent
    rhea_uniprot_path = Path(file_path, "../data/rhea-reactions.txt.gz")
    items = []
    item = {}
    for line in gzip.open(rhea_uniprot_path, "rt"):
        vals = line.strip().split(maxsplit=1)
        if len(vals) < 2:
            continue
        if vals[0] == "ENTRY":
            item["RHEA_ID"] = vals[1].split(":")[1]
        if vals[0] == "DEFINITION":
            item["DEFINITION"] = vals[1]
        if vals[0] == "EQUATION":
            item["EQUATION"] = vals[1]
            items.append(item)
            item = {}

    return pd.DataFrame(items)


def main():
    file_path = Path(__file__).resolve().parent
    rhea_uniprot_s_path = Path(file_path, "../data/rhea2uniprot_sprot.tsv")
    rhea_uniprot_t_path = Path(file_path, "../data/rhea2uniprot_trembl.tsv.gz")

    df_rhea_uniprot_s = pd.read_csv(rhea_uniprot_s_path, sep="\t")
    df_rhea_uniprot_t = pd.read_csv(rhea_uniprot_t_path, sep="\t")
    df_rhea_reactions = parse_rhea_reactions()

    df_rhea_uniprot_s[["RHEA_ID", "ID"]].to_csv(
        Path(file_path, "../data/rhea2uniprot_sprot.csv"), index=None
    )

    df_rhea_uniprot_t = (
        df_rhea_uniprot_t[["RHEA_ID", "ID"]]
        .groupby(by="RHEA_ID")
        .count()
        .reset_index()
        .rename(columns={"ID": "COUNT"})
    )

    df_rhea_uniprot_t.to_csv(
        Path(file_path, "../data/rhea2uniprot_trembl.csv"), index=None
    )
    df_rhea_reactions.to_csv(Path(file_path, "../data/rhea-reactions.csv"), index=None)


if __name__ == "__main__":
    main()
