#!/usr/bin/env python3
# extract_code.py

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler

# ---------------- Configuration ----------------
DEFAULT_RAW_DIR = Path("data/raw_code")
OUT_PATH = Path("../data/processed/train_data.jsonl")
LOG_PATH = Path("../data/logs/ever_log.log")

SKIP_DIRS = {"vendor", ".git", ".svn", ".hg", "__pycache__"}
SKIP_EXTS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.exe', '.dll',
    '.so', '.bin', '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf',
    '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.mp3',
    '.mp4', '.avi', '.mov', '.wmv', '.iso', '.svg'
}
MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2 MiB

# ---------------- Logging Setup ---------------
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("extract_code")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(fh)


def extract_all_code(file_path: Path, base_dir: Path) -> dict | None:
    """
    Read a text-based file under size limit and package as a prompt.
    Returns None on skip or error.
    """
    try:
        size = file_path.stat().st_size
        if size > MAX_SIZE_BYTES:
            logger.debug(f"SKIP large: {file_path.relative_to(base_dir)} ({size} bytes)")
            return None

        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            logger.debug(f"SKIP empty: {file_path.relative_to(base_dir)}")
            return None

    except Exception as e:
        logger.warning(f"ERROR reading {file_path}: {e}")
        return None

    rel = file_path.relative_to(base_dir)
    prompt = f"### Instruction:\nExplain what this file `{rel}` does.\n\n" \
             f"### Response:\n{content}"
    logger.debug(f"EXTRACT: {rel}")
    return {"text": prompt}


def clone_repo(repo_url: str) -> Path:
    """
    Clone Git repository into a temporary directory and return its path.
    """
    temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info(f"Cloned repository to: {temp_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository {repo_url}: {e}")
        sys.exit(1)
    return Path(temp_dir)


def main():
    p = ArgumentParser(description="Extract code files into JSONL prompts")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--repo",
        type=str,
        help="Git repository URL to clone and extract code from"
    )
    group.add_argument(
        "--path",
        type=Path,
        help="Local directory path to extract code from"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="Output JSONL file path"
    )
    args = p.parse_args()

    # Determine source directory
    if args.repo:
        raw_dir = clone_repo(args.repo)
    else:
        raw_dir = args.path.resolve() if args.path else DEFAULT_RAW_DIR.resolve()

    logger.info(f"Using source directory: {raw_dir}")
    if not raw_dir.exists() or not raw_dir.is_dir():
        logger.error(f"Source directory not found or not a directory: {raw_dir}")
        sys.exit(1)

    # Prepare output path
    out_file = args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"START extracting from: {raw_dir}")
    prompts = []
    file_count = 0

    for root, dirs, files in os.walk(raw_dir, topdown=True):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        for fname in files:
            file_count += 1
            fp = Path(root) / fname
            ext = fp.suffix.lower()
            if ext in SKIP_EXTS or fp.name.startswith('.') or fp.name in {'.gitignore', '.gitattributes'}:
                logger.debug(f"SKIP file: {fp.relative_to(raw_dir)}")
                continue
            sample = extract_all_code(fp, raw_dir)
            if sample:
                prompts.append(sample)
            if file_count % 100 == 0:
                logger.info(f"Processed {file_count} files; extracted {len(prompts)} prompts so far")

    logger.info(f"Writing {len(prompts)} prompts to {out_file}")
    with out_file.open("w", encoding="utf-8") as out:
        for item in prompts:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("✅ Extraction complete")


if __name__ == "__main__":
    main()
