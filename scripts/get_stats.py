import pickle
from pathlib import Path
from statistics import mean, stdev
from typing import List
import typer
from pycm import ConfusionMatrix


def main(cm_files: List[str]):
    acc = []
    f1 = []

    for cm_file in cm_files:
        if not Path(cm_file).exists():
            continue

        cm: ConfusionMatrix = pickle.load(open(cm_file, "rb"))
        acc.append(cm.overall_stat["Overall ACC"])
        f1.append(cm.average("F1"))

    if len(acc) > 1 and len(f1) > 1:
        print("acc", round(mean(acc), 3), round(stdev(acc), 3))
        print("f1", round(mean(f1), 3), round(stdev(f1), 3))
        print(
            f"${round(mean(acc), 2)}\\pm{round(stdev(acc), 2)}$ (${round(mean(f1), 2)}\\pm{round(stdev(f1), 2)}$)"
        )
    else:
        print("acc", round(mean(acc), 3))
        print("f1", round(mean(f1), 3))


if __name__ == "__main__":
    typer.run(main)
