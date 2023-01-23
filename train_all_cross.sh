SCRIPT_PATH='scripts'
DATA_PATH='data-v5'
MODEL_PATH='models-v5'

for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-${v}-ec1-train.csv ${DATA_PATH}/ecreact-${v}-ec1-valid.csv ${DATA_PATH}/ecreact-${v}-ec1-test.csv ${MODEL_PATH}/ecreact-${v}-ec1; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-${v}-ec12-train.csv ${DATA_PATH}/ecreact-${v}-ec12-valid.csv ${DATA_PATH}/ecreact-${v}-ec12-test.csv ${MODEL_PATH}/ecreact-${v}-ec12;python ${SCRIPT_PATH}/train.py ${DATA_PATH}/ecreact-${v}-ec123-train.csv ${DATA_PATH}/ecreact-${v}-ec123-valid.csv ${DATA_PATH}/ecreact-${v}-ec123-test.csv ${MODEL_PATH}/ecreact-${v}-ec123
done

# python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12;python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123