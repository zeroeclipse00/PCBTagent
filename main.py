# Refactored by: spn on 2025-09-18

"""
主程序入口：解析命令行参数并启动处理流程。
"""

import logging
import sys

from config import *
from pipeline import process_file, process_folder
from utils.logging_setup import *
from utils.parser import create_parser


def run(input_path: Path, output_path: Path, **process_kwargs) -> int:
    logger = logging.getLogger(LOGGER_NAME)
    
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return

    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            logger.error("--output must be a DIRECTORY when --input is a DIRECTORY. Got file: %s", output_path)
            return
        output_path.mkdir(parents=True, exist_ok=True)
        process_folder(input_dir=str(input_path), output_dir=str(output_path), **process_kwargs)
        logger.info("All done. Outputs are under: %s", output_path)

    elif input_path.is_file():
        process_file(input_path=str(input_path), output_path=str(output_path), **process_kwargs)
        logger.info("All done. Output wrote to: %s", output_path)

    else:
        logger.error("Input path is neither a regular file nor a directory: %s", input_path)
        return


def main() -> None:
    parser = create_parser(INPUT_PATH, OUTPUT_PATH, PROVIDER, BATCH_SIZE, CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    setup_logging_original_fix(verbosity=args.verbose, log_file=args.log_file)
    logger = logging.getLogger(LOGGER_NAME)

    process_kwargs = {
        "provider": args.provider,
        "batch_size": args.batch_size,
        "conf_threshold": args.threshold,
        "include_gt_in_prompt": args.include_gt,
        "verbosity": args.verbose,
        "log_file": args.log_file,
    }

    try:
        code = run(INPUT_PATH, OUTPUT_PATH, **process_kwargs)
        sys.exit(code)
    except Exception:
        logger = logging.getLogger("pcb-ocr-corrector.main")
        logger.exception("An unhandled error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
