"""One process shares the refreshed snapshot across the HTTP and websocket threads."""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '5050')}"
workers = 1
worker_class = "gthread"
control_socket_disable = True
threads = 8
timeout = 60
graceful_timeout = 20


def post_worker_init(worker):
    from ai_server import AUTO_REFRESH, LIVE_STORE

    if LIVE_STORE:
        LIVE_STORE.start(automatic=AUTO_REFRESH)
