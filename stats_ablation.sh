SCRIPT_PATH='scripts'
MODEL_PATH='experiments/models'


SHUFFLE_FRAC=0.01
echo '=============================================================='
echo ${SHUFFLE_FRAC}
echo '--------------------------------------------------------------'
echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec1-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec12-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec123-shuffle-${SHUFFLE_FRAC}.cm

SHUFFLE_FRAC=0.05
echo '=============================================================='
echo ${SHUFFLE_FRAC}
echo '--------------------------------------------------------------'
echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec1-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec12-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec123-shuffle-${SHUFFLE_FRAC}.cm

SHUFFLE_FRAC=0.1
echo '=============================================================='
echo ${SHUFFLE_FRAC}
echo '--------------------------------------------------------------'
echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec1-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec12-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec123-shuffle-${SHUFFLE_FRAC}.cm

SHUFFLE_FRAC=0.2
echo '=============================================================='
echo ${SHUFFLE_FRAC}
echo '--------------------------------------------------------------'
echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec1-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec12-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec123-shuffle-${SHUFFLE_FRAC}.cm

SHUFFLE_FRAC=0.5
echo '=============================================================='
echo ${SHUFFLE_FRAC}
echo '--------------------------------------------------------------'
echo 'Rhea ECX'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec1-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec1-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXY'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec12-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec12-shuffle-${SHUFFLE_FRAC}.cm

echo 'Rhea ECXYZ'
python ${SCRIPT_PATH}/get_stats.py ${MODEL_PATH}/rheadb-0-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-1-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-2-ec123-shuffle-${SHUFFLE_FRAC}.cm ${MODEL_PATH}/rheadb-3-ec123-shuffle-${SHUFFLE_FRAC}.cm

