import socket
import threading
import time

import webview
from werkzeug.serving import make_server

import app_ai


HOST = "127.0.0.1"
WINDOW_TITLE = "אוצריא AI"
WINDOW_SIZE = (1400, 950)


def pick_free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def boot_engine() -> None:
    cfg = app_ai.load_settings()
    app_ai.ENGINE.last_cfg = cfg
    try:
        app_ai.ensure_offline_assets()
        app_ai.ENGINE.load_resources(
            cfg.get("db_path", app_ai.DEFAULT_DB_PATH),
            cfg.get("edition", "v3"),
            cfg.get("model_source", "zip"),
            cfg.get("zip_path", ""),
        )
        app_ai.ENGINE.build_index(
            cfg.get("db_path", app_ai.DEFAULT_DB_PATH),
            int(cfg.get("max_chunks", 100000)),
            int(cfg.get("ideal_chunk_words", app_ai.IDEAL_CHUNK_WORDS)),
            int(cfg.get("max_chunk_words", app_ai.MAX_CHUNK_WORDS)),
            int(cfg.get("overlap_words", app_ai.DEFAULT_OVERLAP_WORDS)),
        )
    except Exception as exc:
        app_ai.ENGINE._update("error", f"שגיאה בהפעלה: {exc}", 0)


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._server = make_server(host, port, app_ai.app, threaded=True)

    def run(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


def wait_until_ready(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"השרת המקומי לא עלה בזמן על {host}:{port}")


def main() -> None:
    port = pick_free_port()

    threading.Thread(target=boot_engine, daemon=True).start()

    server = ServerThread(HOST, port)
    server.start()
    wait_until_ready(HOST, port)

    window = webview.create_window(
        WINDOW_TITLE,
        f"http://{HOST}:{port}",
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=(1100, 760),
    )

    try:
        webview.start()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
