# theia
An extension to rhea-db enabling visualisations as well as reaction searching and classification.

## Development / Hacking
The easiest way to get started is creating a conda environtment. CPU only should be enough as the models are not complex (they can even be trained on the CPU within a few minutes). Next install some more requirements using pip. A `requirements.txt` file is provided in the root folder.

```bash
conda create -n theia -c conda-forge tensorflow-cpu
conda activate theia
pip install -r requirements.txt
```

Next, prepare the data. If you don't feel comfortable giving `get_data.sh` executing permission, have a look at it before you run it (it's just `wget` commands) or download the files yourself.

```bash
cd data
chmod +x get_data.sh
./get_data.sh
```

Then run data preprocessing.

```bash
cd ..
python scripts/preprocess_data.py
```

Almost done. As the Annoy index for the k-nearest neighbour search doesn't fit into a github repo, it has to be created or unzipped.
```bash
gzip -d rhea-drfp.ann.gz
```

If you want to recreate it, be warned that this might take a bit, grab an ice tea.
```bash
python scripts/train_knn.py
```

Now you should be read to run the app.
```bash
chmod +x run.sh
./run.sh
```

The EC prediction models and the TMAP can be regenerated (in case of new data, or for hacking around). Both processes should be pretty fast on a laptop or similar. What does take a bit of time is the fingerprint generation for the ecreact data set during the first run of `train_mlp.py`.
```bash
python scripts/create_map.py
scripts/train_mlp.py --variable ec1
scripts/train_mlp.py --variable ec12
scripts/train_mlp.py --variable ec123
```
v1: rooted atom, radius 3
v2: atom not rooted, radius 3
v3: atom not rooted, radius 2
v4: atom not rooted, radius 2, explicit hydrogens