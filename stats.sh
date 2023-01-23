SCRIPT_PATH='scripts'
MODEL_PATH='models-v5'

echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1.cm ${MODEL_PATH}/rheadb-1-ec1.cm ${MODEL_PATH}/rheadb-2-ec1.cm ${MODEL_PATH}/rheadb-3-ec1.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12.cm ${MODEL_PATH}/rheadb-1-ec12.cm ${MODEL_PATH}/rheadb-2-ec12.cm ${MODEL_PATH}/rheadb-3-ec12.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123.cm ${MODEL_PATH}/rheadb-1-ec123.cm ${MODEL_PATH}/rheadb-2-ec123.cm ${MODEL_PATH}/rheadb-3-ec123.cm

echo 'ECREACT ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/ecreact-0-ec1.cm ${MODEL_PATH}/ecreact-1-ec1.cm ${MODEL_PATH}/ecreact-2-ec1.cm ${MODEL_PATH}/ecreact-3-ec1.cm

echo 'ECREACT ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/ecreact-0-ec12.cm ${MODEL_PATH}/ecreact-1-ec12.cm ${MODEL_PATH}/ecreact-2-ec12.cm ${MODEL_PATH}/ecreact-3-ec12.cm

echo 'ECREACT ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/ecreact-0-ec123.cm ${MODEL_PATH}/ecreact-1-ec123.cm ${MODEL_PATH}/ecreact-2-ec123.cm ${MODEL_PATH}/ecreact-3-ec123.cm