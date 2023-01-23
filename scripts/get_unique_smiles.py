import pickle
from collections import defaultdict
import typer


def main(file_path: str):
    data = pickle.load(open(file_path, "rb"))

    count = 0
    occupancy = defaultdict(int)
    for key, value in data.items():
        count += len(value)
        occupancy[len(value)] += 1

    print(count)
    print(occupancy)

    more_than_one = 0
    for key, value in occupancy.items():
        if key > 1:
            more_than_one += value

    print(more_than_one)


if __name__ == "__main__":
    typer.run(main)
