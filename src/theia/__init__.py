"""Explainable EC number predictions from reactions."""
__version__ = "0.5"

from .data_manager import DataManager

dm = DataManager()


def download():
    dm.download_file(
        [
            "https://github.com/daenuprobst/theia/blob/main/data/ecreact-nofilter-1.0.csv.gz?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/data/rhea-reactions.csv.gz?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/data/rheadb.csv.gz?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec1-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec1-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec1.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec12-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec12-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec12.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec123-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec123-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-0-ec123.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec1-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec1-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec1.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec12-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec12-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec12.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec123-background.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec123-le.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-0-ec123.pt?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/ecreact-map.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rheadb-map.pkl?raw=true",
            "https://github.com/daenuprobst/theia/blob/main/models/rhea-drfp.ann.gz?raw=true",
        ]
    )
