# Refactored by: spn on 2025-09-18

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import *
from pipeline import process_file, process_folder
from utils.logging_setup import *

def run(input_path: Path, output_path: Path, **process_kwargs) -> int:
    logger = logging.getLogger(LOGGER_NAME)
    
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 2

    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            logger.error("--output must be a DIRECTORY when --input is a DIRECTORY. Got file: %s", output_path)
            return 2
        output_path.mkdir(parents=True, exist_ok=True)
        process_folder(input_dir=str(input_path), output_dir=str(output_path), **process_kwargs)
        logger.info("All done. Outputs are under: %s", output_path)
        return 0

    if input_path.is_file():
        process_file(input_path=str(input_path), output_path=str(output_path), **process_kwargs)
        logger.info("All done. Output wrote to: %s", output_path)
        return 0

    logger.error("Input path is neither a regular file nor a directory: %s", input_path)
    return 2


def main() -> None:
    setup_logging_original_fix(verbosity=VERBOSITY, log_file=LOG_FILE)
    logger = logging.getLogger(LOGGER_NAME)
    try:
        process_kwargs = {
            "provider": PROVIDER,
            "batch_size": BATCH_SIZE,
            "threshold": CONFIDENCE_THRESHOLD,
            "include_gt_in_prompt": INCLUDE_GT_IN_PROMPT,
        }
        code = run(INPUT_PATH, OUTPUT_PATH, **process_kwargs)
        sys.exit(code)
    except Exception:
        logger = logging.getLogger("pcb-ocr-corrector.main")
        logger.exception("An unhandled error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
