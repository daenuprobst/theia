SCRIPT_PATH='scripts'
DATA_PATH='data-v5'
MODEL_PATH='models-v5'

python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-0-ec1-train.csv ${DATA_PATH}/rheadb-0-ec1-valid.csv ${DATA_PATH}/rheadb-0-ec1-test.csv ${MODEL_PATH}/rheadb-0-ec1
python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-0-ec12-train.csv ${DATA_PATH}/rheadb-0-ec12-valid.csv ${DATA_PATH}/rheadb-0-ec12-test.csv ${MODEL_PATH}/rheadb-0-ec12
python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-0-ec123-train.csv ${DATA_PATH}/rheadb-0-ec123-valid.csv ${DATA_PATH}/rheadb-0-ec123-test.csv ${MODEL_PATH}/rheadb-0-ec123

python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-0-ec1-train.csv ${DATA_PATH}/ecreact-0-ec1-valid.csv ${DATA_PATH}/ecreact-0-ec1-test.csv ${MODEL_PATH}/ecreact-0-ec1
python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-0-ec12-train.csv ${DATA_PATH}/ecreact-0-ec12-valid.csv ${DATA_PATH}/ecreact-0-ec12-test.csv ${MODEL_PATH}/ecreact-0-ec12
python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-0-ec123-train.csv ${DATA_PATH}/ecreact-0-ec123-valid.csv ${DATA_PATH}/ecreact-0-ec123-test.csv ${MODEL_PATH}/ecreact-0-ec123
