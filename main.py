# -*- coding: utf-8 -*-
"""
主程序入口：解析命令行参数并启动处理流程。
"""

import os
import argparse
import logging

# 导入项目内的模块和配置
from config import PROVIDER, BATCH_SIZE, CONF_THRESHOLD
from pipeline import process_file, process_folder
from utils.logging_setup import setup_logging_original_fix

def main():
    """解析参数并根据输入类型（文件或目录）调用相应的处理函数。"""
    parser = argparse.ArgumentParser(
        description="PCB OCR post-correction with lightweight RAG + LLM batching. "
                    "Input can be a single file OR a directory."
    )
    parser.add_argument("--input", required=True, help="Input path: a txt file OR a directory containing .txt")
    parser.add_argument("--output", required=True,
                        help="Output path: if --input is file -> output txt path; if --input is dir -> output directory")
    parser.add_argument("--provider", default=PROVIDER, choices=["gpt", "deepseek"], help="LLM provider")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size (e.g., 30 or 50)")
    parser.add_argument("--threshold", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--include_gt", action="store_true", help="Include GT in the prompt (default off)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="Increase verbosity: -v (INFO), -vv (DEBUG), -vvv (DEBUG+).")
    parser.add_argument("--log_file", default=None, help="Optional log file path")
    args = parser.parse_args()

    # 在程序开始时设置一次日志
    setup_logging_original_fix(verbosity=args.verbose, log_file=args.log_file)
    logger = logging.getLogger("pcb-ocr-corrector.main")

    # 准备传递给处理函数的参数
    process_kwargs = {
        "provider": args.provider,
        "batch_size": args.batch_size,
        "conf_threshold": args.threshold,
        "include_gt_in_prompt": args.include_gt,
        "verbosity": args.verbose,
        "log_file": args.log_file,
    }

    in_path = args.input
    out_path = args.output

    try:
        if os.path.isdir(in_path):
            # 目录模式
            if os.path.exists(out_path) and not os.path.isdir(out_path):
                logger.error("--output must be a DIRECTORY when --input is a DIRECTORY")
                return
            os.makedirs(out_path, exist_ok=True)
            process_folder(input_dir=in_path, output_dir=out_path, **process_kwargs)
            logger.info(f"All done. Outputs are under: {out_path}")
        elif os.path.isfile(in_path):
            # 单文件模式
            process_file(input_path=in_path, output_path=out_path, **process_kwargs)
            logger.info(f"All done. Output wrote to: {out_path}")
        else:
            logger.error(f"Input path does not exist or is not a file/directory: {in_path}")

    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
