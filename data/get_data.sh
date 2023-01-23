#!/bin/bash

wget https://raw.githubusercontent.com/rxn4chemistry/biocatalysis-model/main/data/ecreact-nofilter-1.0.csv -O ecreact-nofilter-1.0.csv
wget https://github.com/daenuprobst/data-sets/raw/main/rheadb.csv.gz -O rheadb.csv.gz
gzip ecreact-nofilter-1.0.csv