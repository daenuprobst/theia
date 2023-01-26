MODEL_PATH='../experiments/models'
DATA_PATH='../experiments/data'

python heatmap.py ${MODEL_PATH}/rheadb-0-ec1.cm ${MODEL_PATH}/rheadb-0-ec12.cm ${MODEL_PATH}/rheadb-0-ec123.cm \
${MODEL_PATH}/ecreact-0-ec1.cm ${MODEL_PATH}/ecreact-0-ec12.cm ${MODEL_PATH}/ecreact-0-ec123.cm \
--cols 3 --rows 2 \
--title 'ECX$_{\text{Rhea}}$' \
--title 'ECXY$_{\text{Rhea}}$' \
--title 'ECXYZ$_{\text{Rhea}}$' \
--title 'ECX$_{\text{ECREACT}}$' \
--title 'ECXY$_{\text{ECREACT}}$' \
--title 'ECXYZ$_{\text{ECREACT}}$'

python training_metrics.py ${MODEL_PATH}/rheadb-0-ec1.metrics.pkl ${MODEL_PATH}/rheadb-0-ec12.metrics.pkl ${MODEL_PATH}/rheadb-0-ec123.metrics.pkl \
${MODEL_PATH}/ecreact-0-ec1.metrics.pkl ${MODEL_PATH}/ecreact-0-ec12.metrics.pkl ${MODEL_PATH}/ecreact-0-ec123.metrics.pkl \
--cols 3 --rows 2 \
--title 'ECX$_{\text{Rhea}}$' \
--title 'ECXY$_{\text{Rhea}}$' \
--title 'ECXYZ$_{\text{Rhea}}$' \
--title 'ECX$_{\text{ECREACT}}$' \
--title 'ECXY$_{\text{ECREACT}}$' \
--title 'ECXYZ$_{\text{ECREACT}}$'

python accuracies.py ${MODEL_PATH}/rheadb-0-ec1.cm ${MODEL_PATH}/ecreact-0-ec1.cm ${MODEL_PATH}/rheadb-0-ec12.cm \
${MODEL_PATH}/ecreact-0-ec12.cm ${MODEL_PATH}/rheadb-0-ec123.cm ${MODEL_PATH}/ecreact-0-ec123.cm \
--cols 2 --rows 3 \
--title 'ECX$_{\text{Rhea}}$' \
--title 'ECX$_{\text{ECREACT}}$' \
--title 'ECXY$_{\text{Rhea}}$' \
--title 'ECXY$_{\text{ECREACT}}$' \
--title 'ECXYZ$_{\text{Rhea}}$' \
--title 'ECXYZ$_{\text{ECREACT}}$'

python accuracy_vs_size.py ${DATA_PATH}/rheadb_counts.csv ${MODEL_PATH}/rheadb-0-ec1.cm ${MODEL_PATH}/rheadb-0-ec12.cm ${MODEL_PATH}/rheadb-0-ec123.cm \
--cols 1 --rows 3 --outname "accuracy-vs-size-rheadb" \
--title 'Rhea Classes' \
--title 'Rhea Subclasses' \
--title 'Rhea Sub-Subclasses'

python accuracy_vs_size.py ${DATA_PATH}/ecreact_counts.csv ${MODEL_PATH}/ecreact-0-ec1.cm ${MODEL_PATH}/ecreact-0-ec12.cm ${MODEL_PATH}/ecreact-0-ec123.cm \
--cols 1 --rows 3 --outname "accuracy-vs-size-ecreact" \
--title 'ECREACT Classes' \
--title 'ECREACT Subclasses' \
--title 'ECREACT Sub-Subclasses'

python tmaps.py ../experiments/data
