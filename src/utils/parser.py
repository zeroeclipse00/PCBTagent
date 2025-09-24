"""
数据行解析、重建和批处理相关的辅助函数。
"""
import logging
from typing import List, Tuple, Optional

# 获取一个日志记录器实例，用于在本模块中记录日志
logger = logging.getLogger("pcb-ocr-corrector.parser")


def parse_line(line: str) -> Tuple[str, str, float]:
    """
    解析 'gt||ocr conf' 格式的输入行。

    Returns:
        Tuple[str, str, float]: (gt, pred, conf)
    """
    if "||" not in line:
        # 为没有分隔符的行提供备用逻辑
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                # 尝试将最后一个部分解析为置信度
                conf = float(parts[-1])
                pred = " ".join(parts[1:-1])
                gt = parts[0]
                return gt, pred, conf
            except ValueError:
                # 如果最后一个部分不是浮点数，则将其视为 pred 的一部分
                return parts[0], " ".join(parts[1:]), 0.0
        elif len(parts) == 1:
            return parts[0], "", 0.0
        return "", "", 0.0

    gt, right = line.split("||", 1)
    gt = gt.strip()
    right = right.strip()

    r_tokens = right.split()
    if not r_tokens:
        pred, conf = "", 0.0
    else:
        try:
            conf = float(r_tokens[-1])
            pred = " ".join(r_tokens[:-1]).strip()
        except ValueError:
            # 如果右侧最后一个词不是浮点数，则将整个右侧视为 pred
            conf = 0.0
            pred = right

    return gt, pred, conf

def rebuild_line(gt: str, pred: str, conf: float, corrected: str) -> str:
    """
    重建一行，格式为: "gt||pred conf corrected"
    """
    return f"{gt}||{pred} {conf:.4f} {corrected}".strip()

def chunked(seq: List, size: int):
    """将序列切分为指定大小的块。"""
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def postprocess_llm_block(raw_text: str, expected_n: int) -> Optional[List[str]]:
    """
    从 LLM 的原始输出中提取每行的 token，并进行严格的数量检查。

    - 如果行数与预期完全相符，则返回提取的 token 列表。
    - 如果行数不符（过多或过少），记录警告并返回 None，表示此次返回无效，需要重试。
    """
    # 过滤掉空行和常见的、非token的干扰行
    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
    common_intros = ["here are the corrected tokens:", "sure, here are the results:"]
    lines = [ln for ln in lines if ln.lower() not in common_intros]

    if len(lines) != expected_n:
        logger.warning(
            f"LLM returned an incorrect number of lines. "
            f"Expected: {expected_n}, Got: {len(lines)}. This batch will be retried."
        )
        return None

    return lines

## spn: move parser config to .env
# spn: split and wrap parser creation    
# def create_parser(INPUT_PATH: Path, OUTPUT_PATH: Path, PROVIDER: str, BATCH_SIZE: int, CONFIDENCE_THRESHOLD: float):
#     parser = argparse.ArgumentParser(
#         description="PCB OCR post-correction with lightweight RAG + LLM batching. "
#                     "Input can be a single file OR a directory."
#     )
#     parser.add_argument(
#         "--i",
#         "--input",
#         type=str,
#         default=str(INPUT_PATH),
#         help="Input path: a txt file OR a directory containing .txt"
#     )
#     parser.add_argument(
#         "--o",
#         "--output",
#         type=str,
#         default=str(OUTPUT_PATH),
#         help="Output path: if --input is file -> output txt path; if --input is dir -> output directory"
#     )
#     parser.add_argument(
#         "--provider",
#         default=PROVIDER,
#         choices=["gpt", "deepseek"],
#         help="LLM provider"
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=BATCH_SIZE,
#         help="Batch size (e.g., 30 or 50)"
#     )
#     parser.add_argument(
#         "--threshold",
#         type=float,
#         default=CONFIDENCE_THRESHOLD, 
#         help="Confidence threshold"
#     )
#     parser.add_argument(
#         "--include_gt",
#         action="store_true",
#         default=False,
#         help="Include GT in the prompt (default off)"
#     )
#     parser.add_argument(
#         "-v",
#         "--verbose",
#         action="count",
#         default=1,
#         help="Increase verbosity: -v (INFO), -vv (DEBUG), -vvv (DEBUG+)."
#     )
#     parser.add_argument(
#         "--log_file",
#         default=None,
#         help="Optional log file path"
#     )
#     return parser


