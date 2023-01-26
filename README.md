# :anchor: Theia
## Quickstart
As you need at least Python 3.9 to get started, I suggest you use conda to create an environment with an up-to-date Python versions (3.11 is really, really fast, so I suggest using this as soon as rdkit supports it). For now, let's go with Python 3.10: `conda create -n theia python==3.10 && conda activate theia` is all you need (ha). Then you can go ahead and install theia using pip (theia was taken, so make sure to install theia-pypi, except if you want to parse log files):
```
pip install theia-pypi
```
Thats pretty much it, now you can start theia by simply typing:
```
theia
```
and open the url `http://127.0.0.1:5000/` in your web browser. In case you don't want or need an UI, you can also use the cli to simply predict an EC number from an arbitrary reaction:
```
theia-cli "rheadb.ec123" "S=C=NCC1=CC=CC=C1>>N#CSCC1=CC=CC=C1"
```
If you want a bit more information than just the predicted EC class, you can also get the top-5 probabilities:
```
theia-cli "rheadb.ec123" "S=C=NCC1=CC=CC=C1>>N#CSCC1=CC=CC=C1" --probs
```
Or, if you want human-readable output, you can make it pretty:
```
theia-cli "rheadb.ec123" "S=C=NCC1=CC=CC=C1>>N#CSCC1=CC=CC=C1" --probs --pretty
```
and you'll get a neat table...
```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Prediction ┃ Probability [%] ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 2.7.4      │           14.22 │
│ 2.3.2      │           11.03 │
│ 2.3.1      │            7.15 │
│ 2.7.8      │            4.62 │
│ 2.6.1      │            4.05 │
└────────────┴─────────────────┘
```
Of course, there are more models than `rhea.ec123`, which we used in the previous examples. Here's a complete list of all the included models:
| Model         | Trained on  | Name            |
|---------------|-------------|-----------------|
| Rhea ECX      | Rhea        | `rheadb.ec1`    |
| Rhea ECXY     | Rhea        | `rheadb.ec12`   |
| Rhea ECXYZ    | Rhea        | `rheadb.ec123`  |
| ECREACT ECX   | ECREACT 1.0 | `ecreact.ec1`   |
| ECREACT ECXY  | ECREACT 1.0 | `ecreact.ec12`  |
| ECREACT ECXYZ | ECREACT 1.0 | `ecreact.ec123` |

## Reproduction & Custom Models
