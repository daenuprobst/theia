bind = '0.0.0.0:8000'
workers = 3
timeout = 90
raw_env = [
    'SCRIPT_NAME=/theia',
    'FLASK_CONFIGURATION=PRODUCTION'
]
