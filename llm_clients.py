"""
与 LLM API 交互的客户端，包含重试和指数退避逻辑。
"""

import time
import random
import logging
import requests
from typing import List, Dict
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)

logger = logging.getLogger(__name__)

def _exp_backoff_sleep(attempt: int, base: float = 1.0, jitter: float = 0.25) -> float:
    """计算带抖动的指数退避秒数。"""
    delay = base * (2 ** attempt)
    return delay + random.uniform(0, base * jitter)

def call_gpt_chat(messages: List[Dict], model: str = OPENAI_MODEL,
                  api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL,
                  timeout: int = 60, max_retries: int = 3) -> str:
    """使用日志和退避机制调用 OpenAI 兼容的聊天补全 API。"""
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
    """使用日志和退避机制调用 DeepSeek 聊天补全 API。"""
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
