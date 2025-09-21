import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler
from models.log_manager import LogManager

# Use the shared LogManager instance
logmgr = LogManager()
# Ensure extract logger exists (this may create file + in-memory buffer)
logger_obj = logmgr.setup_logger("extract")


class CodeExtractor:
    def __init__(
        self,
        out_path: Path | str = None,
        max_size_bytes: int = 8 * 1024 * 1024,
    ):
        # default output path relative to project if not provided
        base = Path(__file__).resolve().parent.parent
        default_out = base / "data" / "processed" / "train_data.jsonl"
        self.out_path = Path(out_path) if out_path else default_out
        self.max_size_bytes = int(max_size_bytes)

        # dirs and extensions to skip
        self.skip_dirs = {"vendor", ".git", ".svn", ".hg", "__pycache__", "node_modules", "dist", "build"}
        self.skip_exts = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.exe', '.dll',
            '.so', '.bin', '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.mp3',
            '.mp4', '.avi', '.mov', '.wmv', '.iso', '.svg'
        }

        # local references
        self.logmgr = logmgr
        self.logger = logger_obj  # may be a logging.Logger or wrapper

    def _log(self, msg: str, level: str = "info"):
        """
        Centralized logging: write both to system logger and LogManager in-memory store.
        """
        text = str(msg)
        # Use Python logger if available
        try:
            if hasattr(self.logger, level):
                getattr(self.logger, level)(text)
            else:
                # fallback to info
                self.logger.info(text)
        except Exception:
            pass

        # Append to shared LogManager buffer (always)
        try:
            self.logmgr.append("extract", text)
        except Exception:
            # best-effort only
            pass

    def extract_code(self, raw_dir: Path | str, output_file: str | None = None, callback=None, clear_logs: bool = True):
        """
        Main entry expected by Flask: extract from directory `raw_dir` and write to `output_file`.
        - callback: optional function(msg:str) to receive progress lines
        - returns True on success, raises Exception on failure
        """
        self._log(f"Received raw_dir: {raw_dir}")

        if clear_logs:
            try:
                self.logmgr.clear("extract")
            except Exception:
                pass

        if isinstance(raw_dir, str):
            raw_dir = Path(raw_dir)

        if output_file:
            out_path = Path(output_file)
        else:
            out_path = self.out_path

        if callback is None:
            # define default callback that writes both where appropriate
            def callback(msg):
                self._log(msg)

        if not raw_dir.exists() or not raw_dir.is_dir():
            msg = f"Source directory not found or not a directory: {raw_dir}"
            self._log(msg, level="error")
            raise FileNotFoundError(msg)

        # ensure parent exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            return self.extract_from_dir(raw_dir, out_path, callback)
        except Exception as e:
            self._log(f"Extraction failed: {e}", level="error")
            raise

    def extract_all_code(self, file_path: Path, base_dir: Path, callback=None) -> dict | None:
        """
        Extracts a single file into the prompt structure. Returns None to skip.
        """
        try:
            size = file_path.stat().st_size
            if size > self.max_size_bytes:
                self._log(f"SKIP large: {file_path.relative_to(base_dir)} ({size} bytes)", level="warning")
                if callback:
                    callback(f"SKIP large: {file_path.relative_to(base_dir)}")
                return None

            content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                self._log(f"SKIP empty: {file_path.relative_to(base_dir)}", level="debug")
                if callback:
                    callback(f"SKIP empty: {file_path.relative_to(base_dir)}")
                return None

        except Exception as e:
            self._log(f"ERROR reading {file_path}: {e}", level="warning")
            if callback:
                callback(f"ERROR reading {file_path}: {e}")
            return None

        rel = file_path.relative_to(base_dir)
        prompt = (
            f"### Instruction:\nExplain what this file `{rel}` does.\n\n"
            f"### Response:\n{content}"
        )
        self._log(f"EXTRACT: {rel}")
        if callback:
            callback(f"EXTRACT: {rel}")
        return {"text": prompt}

    def clone_repo(self, repo_url: str) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="repo_clone_"))
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(temp_dir)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._log(f"Cloned repository to: {temp_dir}")
        except subprocess.CalledProcessError as e:
            self._log(f"Failed to clone repository {repo_url}: {e}", level="error")
            raise
        return temp_dir

    def extract_from_dir(self, raw_dir: Path, out_path: Path, callback=None) -> bool:
        """
        Walk `raw_dir`, build prompts, write them to out_path (jsonl).
        Returns True when done.
        """
        self._log(f"Using source directory: {raw_dir}")
        prompts = []
        file_count = 0

        for root, dirs, files in os.walk(raw_dir, topdown=True):
            dirs[:] = [d for d in dirs if d.lower() not in self.skip_dirs]
            for fname in files:
                file_count += 1
                fp = Path(root) / fname
                ext = fp.suffix.lower()
                if ext in self.skip_exts or fp.name.startswith('.') or fp.name in {'.gitignore', '.gitattributes'}:
                    self._log(f"SKIP file: {fp.relative_to(raw_dir)}", level="debug")
                    if callback:
                        callback(f"SKIP file: {fp.relative_to(raw_dir)}")
                    continue

                sample = self.extract_all_code(fp, raw_dir, callback=callback)
                if sample:
                    prompts.append(sample)

                if file_count % 100 == 0:
                    msg = f"Processed {file_count} files; extracted {len(prompts)} prompts so far"
                    self._log(msg)
                    if callback:
                        callback(msg)

        self._log(f"Writing {len(prompts)} prompts to {out_path}")
        if callback:
            callback(f"Writing {len(prompts)} prompts to {out_path}")

        # write out
        with out_path.open("w", encoding="utf-8") as out:
            for item in prompts:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

        self._log("✅ Extraction complete")
        if callback:
            callback("✅ Extraction complete")
        return True
