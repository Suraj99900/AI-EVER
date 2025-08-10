import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler

class CodeExtractor:
    def __init__(self,
                 out_path=Path("../data/processed/train_data.jsonl"),
                 log_path=Path("../data/logs/extract_code.log"),
                 max_size_bytes=8 * 1024 * 1024):

        self.out_path = Path(__file__).resolve().parent / "../data/processed/train_data.jsonl"
        self.max_size_bytes = max_size_bytes
        
        self.skip_dirs = {"vendor", ".git", ".svn", ".hg", "__pycache__", "node_modules", "dist", "build"}

        self.skip_exts = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.exe', '.dll',
            '.so', '.bin', '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.mp3',
            '.mp4', '.avi', '.mov', '.wmv', '.iso', '.svg'
        }

        # Setup logging
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("extract_code")
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(ch)

        fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

    def extract_code(self, raw_dir: Path | str, output_file="..data/processed/train_data.jsonl", callback=None, clear_logs: bool = True):
        """Wrapper method expected by Flask app"""
        self.logger.info(f"Converted raw_dir to Path: {raw_dir}")
        if isinstance(raw_dir, str):
            raw_dir = Path(raw_dir)
           
        if output_file is None:
            output_file = self.out_path
        bResult = self.extract_from_dir(raw_dir)
        return bResult



    def extract_all_code(self, file_path: Path, base_dir: Path) -> dict | None:
        try:
            size = file_path.stat().st_size
            if size > self.max_size_bytes:
                self.logger.debug(f"SKIP large: {file_path.relative_to(base_dir)} ({size} bytes)")
                return None

            content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                self.logger.debug(f"SKIP empty: {file_path.relative_to(base_dir)}")
                return None

        except Exception as e:
            self.logger.warning(f"ERROR reading {file_path}: {e}")
            return None

        rel = file_path.relative_to(base_dir)
        prompt = f"### Instruction:\nExplain what this file `{rel}` does.\n\n" \
                 f"### Response:\n{content}"
        self.logger.debug(f"EXTRACT: {rel}")
        return {"text": prompt}

    def clone_repo(self, repo_url: str) -> Path:
        temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
        try:
            subprocess.run([
                "git", "clone", "--depth", "1", repo_url, temp_dir
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.logger.info(f"Cloned repository to: {temp_dir}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone repository {repo_url}: {e}")
            sys.exit(1)
        return Path(temp_dir)

    def extract_from_dir(self, raw_dir: Path):
        self.logger.info(f"Using source directory: {raw_dir}")
        if not raw_dir.exists() or not raw_dir.is_dir():
            self.logger.error(f"Source directory not found or not a directory: {raw_dir}")
            sys.exit(1)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"START extracting from: {raw_dir}")
        prompts = []
        file_count = 0

        for root, dirs, files in os.walk(raw_dir, topdown=True):
            dirs[:] = [d for d in dirs if d.lower() not in self.skip_dirs]
            for fname in files:
                file_count += 1
                fp = Path(root) / fname
                ext = fp.suffix.lower()
                if ext in self.skip_exts or fp.name.startswith('.') or fp.name in {'.gitignore', '.gitattributes'}:
                    self.logger.debug(f"SKIP file: {fp.relative_to(raw_dir)}")
                    continue
                sample = self.extract_all_code(fp, raw_dir)
                if sample:
                    prompts.append(sample)
                if file_count % 100 == 0:
                    self.logger.info(f"Processed {file_count} files; extracted {len(prompts)} prompts so far")

        self.logger.info(f"Writing {len(prompts)} prompts to {self.out_path}")
        with self.out_path.open("w", encoding="utf-8") as out:
            for item in prompts:
                print(self.out_path)
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.logger.info("✅ Extraction complete")
        return True
