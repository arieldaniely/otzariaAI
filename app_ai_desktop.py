import os
import sys

# פתרון קריטי לאריזה ב-Windows: הפניית מנוע התצוגה לקובץ הפייתון המובנה בתוכנה שלנו במקום לחפש במחשב של המשתמש
if getattr(sys, 'frozen', False) and sys.platform == 'win32':
    dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    os.environ["PYTHONNET_PYDLL"] = os.path.join(sys._MEIPASS, dll_name)

import socket
import threading
import time

from werkzeug.serving import make_server
import webview

import app_ai


HOST = "127.0.0.1"
WINDOW_TITLE = "אוצריא AI"


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
            [int(b) for b in cfg.get("index_book_ids", []) if str(b).isdigit()]
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

    icon_path = os.path.join(app_ai.STATIC_DIR, "icon.ico")
    # פתיחת החלון הדק שמבוסס על מנוע מערכת ההפעלה
    webview.create_window(WINDOW_TITLE, f"http://{HOST}:{port}", width=1400, height=950, min_size=(1100, 760))
    webview.start(icon=icon_path)
    
    # כשהמשתמש סוגר את החלון, נכבה את השרת ונצא מהתוכנה
    server.shutdown()


if __name__ == "__main__":
    main()
