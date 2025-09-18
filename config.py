# File: config.py
# Description: Central hub for API keys, RAG knowledge bases, prompt references, and other configs.
# Refactored by: spn on 2025-09-11

"""
Centralized management of API keys, knowledge bases, prompt references, 
and other global configurations.
"""

import logging
import os
from pathlib import Path
from typing import List

# data processing at one time
BATCH_SIZE = 50

# results within cofidence below threshold will be sent to LLM
# TODO: not alining with paper (should be 1.1?)
CONFIDENCE_THRESHOLD = 1.01

# whether to include ground truth labels in prompt
INCLUDE_GT_IN_PROMPT = False

# spn: default file structure configurations
# ease command line interaction
WORKING_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(WORKING_DIR / r"Input/Data/labels_with_text")
OUTPUT_PATH = Path(WORKING_DIR / r"Output/Data")

############################### LLM Provider ##################################
PROVIDER = "deepseek"  # gpt / deepseek

# --- GPT (OpenAI-compatible) placeholders ---
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY", 
    "sk-GwVXGz52FpEeYEHpThpxhQZ3IDHykKAOL1l5cr6p8bwvKvjq"
) 
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", 
    "https://api.chatanywhere.tech/v1/chat/completions"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # gpt-4o / gpt-4o-mini

# --- DeepSeek placeholders ---
DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    "sk-3838df7463fe415caaf5100695ec703c"
)
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com/v1/chat/completions"
)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # deepseek-chat / deepseek-reasoner

############################# RAG Knowledge Base ##########################
RAG_KB_PATH = Path(os.getenv(
    "RAG_KB_PATH",
    WORKING_DIR / r"reference_files/knowledge_base_v1.json"
))

########################### Prompt References #########################
REFERENCE_TOKENS_PATH = os.getenv(
    "REFERENCE_TOKENS_PATH",
    WORKING_DIR / r"references/sampled_gts_unique_700_long_300_short.txt"
)
REFERENCE_MAX_TOKENS = int(os.getenv("REFERENCE_MAX_TOKENS", "120"))

def load_reference_tokens(path: str, max_n: int) -> List[str]:
    """Load reference word list from given path for few-shot prompts."""
    path = Path(path)
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
