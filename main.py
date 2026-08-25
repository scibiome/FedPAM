import os
from threading import Thread

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.engine.app import app
from bottle import Bottle

import states  # registers all @app_state decorators

server = Bottle()

path_prefix = os.getenv("PATH_PREFIX")
env = 'fc' if path_prefix else 'native'


def start_app():
    app.register()
    server.mount('/api', api_server)
    server.run(host='localhost', port=5000)


if __name__ == '__main__':
    if env == 'fc':
        print("Starting in fc mode", flush=True)
        start_app()
    else:
        print("Starting in native mode", flush=True)
        # For local dev: run Dash directly without FeatureCloud
        from visualization import app as dash_app
        dash_app.run(host="0.0.0.0", port=8050, debug=True)