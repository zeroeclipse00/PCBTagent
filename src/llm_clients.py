"""
The client interacting with the LLM API includes retry and exponential backoff logic.
"""

import re
import time
import random
import logging
import requests
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import *

logger = logging.getLogger(__name__)

def _exp_backoff_sleep(attempt: int, base: float = 1.0, jitter: float = 0.25) -> float:
    """Calculate the number of seconds for exponential backoff with jitter."""
    delay = base * (2 ** attempt)
    return delay + random.uniform(0, base * jitter)

def call_gpt_chat(messages: List[Dict], model: str = OPENAI_MODEL,
                  api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL,
                  timeout: int = 60, max_retries: int = 3) -> str:
    """Invoke OpenAI-compatible chat completion API using logging and backoff mechanisms."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0}

    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            dt = time.time() - t0
            logger.debug(f"GPT call ok in {dt:.2f}s, tokens unknown (no usage field).")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"GPT call failed after {attempt} retries: {e}")
                raise
            sleep_s = _exp_backoff_sleep(attempt)
            logger.warning(f"GPT call error (attempt {attempt+1}/{max_retries+1}), backing off {sleep_s:.1f}s: {e}")
            time.sleep(sleep_s)
    raise ConnectionError("LLM call failed after all retries.")


def call_deepseek_chat(messages: List[Dict], model: str = DEEPSEEK_MODEL,
                       api_key: str = DEEPSEEK_API_KEY, base_url: str = DEEPSEEK_BASE_URL,
                       timeout: int = 60, max_retries: int = 3) -> str:
    """Invoke DeepSeek chat Completion API using the log and backoff mechanisms."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.0}

    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            dt = time.time() - t0
            logger.debug(f"DeepSeek call ok in {dt:.2f}s.")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"DeepSeek call failed after {attempt} retries: {e}")
                raise
            sleep_s = _exp_backoff_sleep(attempt)
            logger.warning(f"DeepSeek call error (attempt {attempt+1}/{max_retries+1}), backing off {sleep_s:.1f}s: {e}")
            time.sleep(sleep_s)
    raise ConnectionError("LLM call failed after all retries.")


def refined_token_postprocess(original_result):
    """
    A more rigorous result correction function corrects new_result according to specific rules.
    """
    SPACE_INSIDE_WORDS = re.compile(r'(?<=\w)\s+(?=\w)')  # 仅替换单词内部空格
    MULTI_UNDERSCORES  = re.compile(r'_{2,}')
    
    # 2) 先做“仅单词内部”的空格 -> 下划线（知识库：Spaces vs. Underscores）
    #    比如 "VCCIO FLASH" -> "VCCIO_FLASH"；"A - B" 中的连字符不变
    candidate = SPACE_INSIDE_WORDS.sub("_", original_result)

    # 3) 若 original_result 出现 “GP” + 非10 的两位数字（如 GP12, GP19），直接信任 original_result（避免误把它改成 GPIO）
    safe_gp_pattern = re.compile(r'(^|[^A-Za-z0-9])GP(\d\b|(?!10)\d{2}\b)')
    if safe_gp_pattern.search(original_result):
        candidate = SPACE_INSIDE_WORDS.sub("_", original_result)  # 也顺手把 original 里的内部空格替换掉

    # 4) 仅修正特例：GP(10|I0|1O)(\d) -> GPIO\3   （保持前导分隔符）
    gp_bug = re.compile(r'(^|[^A-Za-z0-9])GP(10|I0|1O)(\d)')
    candidate = gp_bug.sub(r'\1GPIO\3', original_result)

    # 捕捉各种大小写混乱的 VCC（包括 vCc、VcC、vCC 等），但保留像 AVCC、VCC3V3 这类组合
    candidate = re.sub(r'(?<![A-Z0-9_])v+ ?c+ ?c+(?![A-Z0-9_])', 'VCC', original_result, flags=re.IGNORECASE)
    candidate = re.sub(r'3v3', '3V3', original_result, flags=re.IGNORECASE)

    # 7) 合并重复下划线（有些场景可能产生 "__"）
    candidate = MULTI_UNDERSCORES.sub('_', original_result)
    
    return candidate
