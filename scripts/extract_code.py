import os
import json
import logging
from pathlib import Path

# ---------------- Configuration ----------------
RAW_DIR = Path("../data/raw_code")  # adjust if necessary
OUT_PATH = Path("../data/processed/train_data.jsonl")
# Skip binary or large asset extensions
SKIP_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.exe', '.dll', '.so', '.bin', '.zip'}
# Max file size to read (e.g., 1 MB)
MAX_SIZE_BYTES = 1 * 1024 * 1024

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def extract_all_code(file_path: Path) -> dict | None:
    """
    Read a text-based file under size limit and package as a prompt.
    Returns None on failure or skip.
    """
    try:
        size = file_path.stat().st_size
        if size > MAX_SIZE_BYTES:
            logger.debug(f"Skipping large file: {file_path} ({size} bytes)")
            return None
        content = file_path.read_text(encoding='utf-8', errors='ignore').strip()
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None

    if not content:
        logger.debug(f"Empty content, skipping: {file_path}")
        return None

    rel_path = file_path.relative_to(RAW_DIR)
    instruction = f"Explain what this file `{rel_path}` does."
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{content}"
    )
    return {"text": prompt}


def main():
    # Validate RAW_DIR
    if not RAW_DIR.exists() or not RAW_DIR.is_dir():
        logger.error(f"RAW_DIR not found or not a directory: {RAW_DIR}")
        return

    # Ensure output directory exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = []
    logger.info(f"Starting extraction from {RAW_DIR}...")
    count = 0

    for root, dirs, files in os.walk(RAW_DIR, topdown=True):
        # Skip any 'vendor' folders
        dirs[:] = [d for d in dirs if d.lower() != 'vendor']

        for fname in files:
            count += 1
            file_path = Path(root) / fname
            ext = file_path.suffix.lower()
            if ext in SKIP_EXTS:
                logger.debug(f"Skipping extension {ext}: {file_path}")
                continue

            sample = extract_all_code(file_path)
            if sample:
                dataset.append(sample)

            if count % 100 == 0:
                logger.info(f"Processed {count} files, collected {len(dataset)} samples so far...")

    # Write JSONL
    logger.info(f"Writing {len(dataset)} prompts to {OUT_PATH}...")
    with OUT_PATH.open('w', encoding='utf-8') as out:
        for item in dataset:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✅ Extraction complete: {len(dataset)} files into {OUT_PATH}")


if __name__ == "__main__":
    main()
