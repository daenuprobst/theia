SCRIPT_PATH='scripts'
DATA_PATH='experiments/data'
MODEL_PATH='experiments/models'

SHUFFLE_FRAC=0.01
for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1 --shuffle-fraction ${SHUFFLE_FRAC}; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12 --shuffle-fraction ${SHUFFLE_FRAC};python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123 --shuffle-fraction ${SHUFFLE_FRAC}
done

SHUFFLE_FRAC=0.05
for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1 --shuffle-fraction ${SHUFFLE_FRAC}; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12 --shuffle-fraction ${SHUFFLE_FRAC};python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123 --shuffle-fraction ${SHUFFLE_FRAC}
done

SHUFFLE_FRAC=0.1
for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1 --shuffle-fraction ${SHUFFLE_FRAC}; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12 --shuffle-fraction ${SHUFFLE_FRAC};python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123 --shuffle-fraction ${SHUFFLE_FRAC}
done

SHUFFLE_FRAC=0.2
for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1 --shuffle-fraction ${SHUFFLE_FRAC}; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12 --shuffle-fraction ${SHUFFLE_FRAC};python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123 --shuffle-fraction ${SHUFFLE_FRAC}
done

SHUFFLE_FRAC=0.5
for v in 1 2 3
do
    python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec1-train.csv ${DATA_PATH}/rheadb-${v}-ec1-valid.csv ${DATA_PATH}/rheadb-${v}-ec1-test.csv ${MODEL_PATH}/rheadb-${v}-ec1 --shuffle-fraction ${SHUFFLE_FRAC}; python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec12-train.csv ${DATA_PATH}/rheadb-${v}-ec12-valid.csv ${DATA_PATH}/rheadb-${v}-ec12-test.csv ${MODEL_PATH}/rheadb-${v}-ec12 --shuffle-fraction ${SHUFFLE_FRAC};python ${SCRIPT_PATH}/train.py ${DATA_PATH}/rheadb-${v}-ec123-train.csv ${DATA_PATH}/rheadb-${v}-ec123-valid.csv ${DATA_PATH}/rheadb-${v}-ec123-test.csv ${MODEL_PATH}/rheadb-${v}-ec123 --shuffle-fraction ${SHUFFLE_FRAC}
done