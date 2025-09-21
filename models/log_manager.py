# utils/log_manager.py
import os
import logging
from logging.handlers import RotatingFileHandler
from collections import deque, defaultdict
import threading
import re
from typing import List, Optional

class LogManager:
    """
    Central logging manager that:
      - Creates named file handlers (rotating)
      - Keeps an in-memory deque per logger for real-time UI poll
      - Provides helpers to append/get/clear logs and to filter noisy lines
    """

    _instance = None
    _instance_lock = threading.Lock()

    ACCESS_LOG_RE = re.compile(
        r'^\d{1,3}(?:\.\d{1,3}){3}\s+-\s+-\s+\[.*?\]\s+"[A-Z]+\s+([^"]+)\s+HTTP/[\d.]+"\s+\d{3}\s+.*$'
    )

    TQDM_RE = re.compile(r'.*\d+%.*\[.*\].*it.*')  # loose match for progressbars

    def __new__(cls, *args, **kwargs):
        # Singleton so modules import same manager
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_log_dir: str = "log", default_max_lines: int = 500):
        # init only once
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.base_log_dir = os.path.abspath(base_log_dir)
        os.makedirs(self.base_log_dir, exist_ok=True)

        self._deques = defaultdict(lambda: deque(maxlen=default_max_lines))
        self._locks = defaultdict(threading.Lock)
        self._handlers = {}  # map logger_name -> file handler
        self._format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    def _file_for(self, logger_name: str):
        # sanitize logger name for filename
        safe = logger_name.replace(".", "_")
        return os.path.join(self.base_log_dir, f"{safe}.log")

    def setup_logger(self, logger_name: str, level=logging.INFO, max_bytes=10 * 1024 * 1024, backup_count=3):
        """
        Create or return a named logger that writes to <base_log_dir>/<logger_name>.log.
        Idempotent — calling multiple times returns same logger.
        """
        if logger_name in self._handlers:
            return logging.getLogger(logger_name)

        fname = self._file_for(logger_name)
        fh = RotatingFileHandler(fname, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(self._format))

        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(fh)
        # prevent propagation to root if you want separation
        logger.propagate = False

        self._handlers[logger_name] = fh
        return logger

    def append(self, logger_name: str, message: str, level=logging.INFO):
        """
        Append a log line to in-memory buffer and to the file via the named logger.
        Use this from code to push UI-visible messages.
        """
        if message is None:
            return
        text = str(message).rstrip()

        # append to in-memory buffer (thread-safe)
        lock = self._locks[logger_name]
        with lock:
            self._deques[logger_name].append(text)

        # also log to file via logger
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            # auto-setup default handler with small file size if not setup
            self.setup_logger(logger_name)
        # use appropriate log level
        logger.log(level, text)

    def get_logs(self, logger_name: str, lines: int = 50, filter_noise: bool = True) -> List[str]:
        """
        Return the last `lines` entries from in-memory buffer.
        If empty, tail the file and return filtered lines.
        """
        lock = self._locks[logger_name]
        with lock:
            buf = list(self._deques[logger_name])[-lines:]
        if buf:
            return buf

        # fallback to file tail (and filter noisy lines)
        out = []
        fname = self._file_for(logger_name)
        if os.path.exists(fname):
            try:
                with open(fname, "r", errors="ignore") as f:
                    for line in reversed(f.readlines()):
                        line = line.strip()
                        if filter_noise and self._is_noise_line(line):
                            continue
                        out.append(line)
                        if len(out) >= lines:
                            break
                return list(reversed(out))
            except Exception:
                return ["Error: failed to read log file."]
        return ["$ Waiting for activity..."]

    def clear(self, logger_name: str):
        lock = self._locks[logger_name]
        with lock:
            self._deques[logger_name].clear()

    def _is_noise_line(self, line: str) -> bool:
        if not line:
            return True
        if self.ACCESS_LOG_RE.match(line):
            return True
        if self.TQDM_RE.match(line):
            return True
        # optionally ignore polling GET lines
        if "/train/logs" in line or "/extract/logs" in line:
            return True
        return False

    # convenience to return a Flask-friendly JSON list
    def get_logs_json(self, logger_name: str, lines: int = 50, filter_noise: bool = True):
        return {"logs": self.get_logs(logger_name, lines=lines, filter_noise=filter_noise)}
