#! /bin/sh
SCRIPT_NAME=/theia gunicorn 'theia.web:create_app()' -w 3 -b 0.0.0.0:8000 --chdir /usr/src/app/theia/src
