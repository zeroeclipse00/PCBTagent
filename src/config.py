# Refactored by: spn on 2025-09-22

"""
Centralized management of API keys, knowledge bases, prompt references, 
and other global configurations.
"""

import logging
import os
from pathlib import Path
from typing import List
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / Path(os.getenv("INPUT_PATH"))
OUTPUT_PATH = PROJECT_ROOT / Path(os.getenv("OUTPUT_PATH"))

RAG_KB_PATH = Path(os.getenv(
    "RAG_KB_PATH",
    PROJECT_ROOT / r"data/resources/knowledge_base_v1.json"
))

REFERENCE_TOKENS_PATH = Path(os.getenv(
    "REFERENCE_TOKENS_PATH",
    PROJECT_ROOT / r"data/resources/sampled_gts_unique_700_long_300_short.txt"
))

# ----------------------------------------------------------------------
# Processing parameters (env override)
# ----------------------------------------------------------------------

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))
INCLUDE_GT_IN_PROMPT = bool(os.getenv("INCLUDE_GT_IN_PROMPT").lower())
REFERENCE_MAX_TOKENS = int(os.getenv("REFERENCE_MAX_TOKENS"))

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

VERBOSITY = int(os.getenv("VERBOSITY"))
LOG_FILE = Path(os.getenv("LOG_FILE"))

# ----------------------------------------------------------------------
# LLM Provider
# ----------------------------------------------------------------------

PROVIDER = os.getenv("LLM_PROVIDER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

# ----------------------------------------------------------------------
# Prompt References
# ----------------------------------------------------------------------

def load_reference_tokens(path: Path, max_n: int) -> List[str]:
    """Load reference word list from given path for few-shot prompts."""
    if not path.exists():
        logging.warning(f"[ref] tokens file not found: {path}")
        return []
    tokens = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            token = line.strip()
            if token:
                tokens.append(token)
                if len(tokens) >= max_n:
                    break
    return tokens

REFERENCE_TOKENS = load_reference_tokens(REFERENCE_TOKENS_PATH, REFERENCE_MAX_TOKENS)

# ----------------------------------------------------------------------
# OCR related
# ----------------------------------------------------------------------

# Used to ignore text that is too small in size. All text smaller than this font size (unit: pixels) will not generate bounding boxes or labels for it.
MIN_FONT_SIZE_TO_DRAW = 7

# Defined the class index (class_id) of the text in the output YOLO tag file.
TEXT_CLASSES = {
    # 显式定义的文本，包括元件的属性、引脚名称/编号、导线网络名等
    'explicit_text': 0,

    # 通过模板匹配识别出的输入端口 (Net In) 关联的文本
    'netin': 0,

    # 通过模板匹配识别出的输出或双向端口 (Net Out/Bidir) 关联的文本
    'netout_netbidir': 0,

    # 通过启发式规则匹配的电源网络 (Power) 文本
    'net_text_power': 0,

    # 通过启发式规则匹配的接地网络 (GND) 文本
    'net_text_gnd': 0,

    # 原理图中的自由文本（例如注释、标题栏信息等）
    'free_text': 1
}
