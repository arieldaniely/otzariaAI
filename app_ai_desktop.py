import inspect
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
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 950
DEFAULT_MIN_SIZE = (1100, 760)
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


def get_centered_window_position(width: int, height: int) -> tuple[int | None, int | None]:
    screens = getattr(webview, "screens", None) or []
    if not screens:
        return None, None

    screen = screens[0]
    screen_x = int(getattr(screen, "x", 0))
    screen_y = int(getattr(screen, "y", 0))
    x = screen_x + max((int(screen.width) - width) // 2, 0)
    y = screen_y + max((int(screen.height) - height) // 2, 0)
    return x, y


def center_window(window, delay: float = 0.0) -> None:
    def task() -> None:
        if delay > 0:
            time.sleep(delay)

        x, y = get_centered_window_position(int(window.width), int(window.height))
        if x is None or y is None:
            return

        try:
            window.move(x, y)
        except Exception:
            pass

    threading.Thread(target=task, daemon=True).start()


def configure_window_behavior(window) -> None:
    def maximize_on_show() -> None:
        try:
            window.maximize()
        except Exception:
            pass

    def center_on_restore() -> None:
        # Give the native window a moment to leave maximized state before moving it.
        center_window(window, delay=0.15)

    if hasattr(window.events, "shown"):
        window.events.shown += maximize_on_show
    if hasattr(window.events, "restored"):
        window.events.restored += center_on_restore


def main() -> None:
    port = pick_free_port()

    threading.Thread(target=boot_engine, daemon=True).start()

    server = ServerThread(HOST, port)
    server.start()
    wait_until_ready(HOST, port)

    icon_path = os.path.join(app_ai.STATIC_DIR, "icon.ico")
    x, y = get_centered_window_position(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    window_kwargs = {}
    if x is not None and y is not None:
        window_kwargs["x"] = x
        window_kwargs["y"] = y
    if "maximized" in inspect.signature(webview.create_window).parameters:
        window_kwargs["maximized"] = True
    # פתיחת החלון הדק שמבוסס על מנוע מערכת ההפעלה
    window = webview.create_window(
        WINDOW_TITLE,
        f"http://{HOST}:{port}",
        width=DEFAULT_WINDOW_WIDTH,
        height=DEFAULT_WINDOW_HEIGHT,
        min_size=DEFAULT_MIN_SIZE,
        **window_kwargs,
    )
    configure_window_behavior(window)
    webview.start(icon=icon_path)
    
    # כשהמשתמש סוגר את החלון, נכבה את השרת ונצא מהתוכנה
    server.shutdown()


if __name__ == "__main__":
    main()
