#! /bin/sh
export FLASK_CONFIG=PRODUCTION
SCRIPT_NAME=/theia /opt/conda/envs/theia/bin/gunicorn 'theia.web:create_app()' -w 10 -b 0.0.0.0:8000 --log-file /www/theia/logs/dbg.log --log-level debug
