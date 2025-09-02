"""
集中管理所有配置、API密钥、模型名称和知识库（RAG_KB）。
"""

import os
from typing import List

# ---- LLM Provider Config ---
PROVIDER = "deepseek"  # "gpt" or "deepseek"

# --- GPT (OpenAI-compatible) placeholders ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")   #  gpt-4o-mini

# --- DeepSeek placeholders ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # deepseek-chat / deepseek-reasoner

BATCH_SIZE = 50  # 一次处理多少条数据
CONF_THRESHOLD = 1.01 # 置信度低于阈值的结果会被发送给llm

# 是否同时发送gt（默认不发送）
INCLUDE_GT_IN_PROMPT = False

# ---- RAG Knowledge Base ---
RAG_KB = {
    "locale": "en-US",
    "overview": "Conservative, character-level guidance for schematic OCR post-correction. Prefer minimal edits. Keep original style unless clearly an OCR confusion.",
    "sections": [
        {
            "id": "spaces_and_underscores",
            "title": "Spaces vs. Underscores",
            "rules": [
                "If OCR produced spaces inside a token, REPLACE internal spaces with underscores (_).",
                "Never introduce new spaces inside tokens.",
                "Do NOT replace hyphen '-' with underscore '_' (they are not interchangeable). Keep '-' if present."
            ],
            "examples": {
                "good": ["USB_CC1", "LED_RED", "HDMI_TX2_N"],
                "bad":  ["USB CC1", "LED-RED (if original was underscore style)"]
            }
        },
        {
            "id": "units_and_symbols",
            "title": "Units & Symbols Integrity",
            "rules": [
                "NEVER drop unit/symbol characters like Ω, µ, °, ±.",
                "Preserve leading '+' or '-' on voltages (e.g., +3.3V).",
                "Do NOT convert between 3.3V and 3V3 styles.",
                "Preserve '/', '(', ')', and domain hints if present."
            ],
            "examples": {
                "good":  ["5.1kΩ", "10kΩ 0603", "+3.3V", "LPDDR4_DQS1N_B"],
                "bad":   ["5.1k", "3.3V -> 3V3", "USB_D- -> USB_DN"]
            }
        },
        {
            "id": "refdes_and_pairs",
            "title": "Refdes & Diff Pairs",
            "rules": [
                "Refdes: uppercase letters + digits (+ optional suffix), e.g., R10, C3, U7, R17A.",
                "Differential pairs must keep *_P/*_N or explicit +/-; DO NOT change '-' to '_' in names like USB_D-."
            ],
            "patterns": {
                "refdes": "[A-Z]{1,3}[0-9]+[A-Z]?",
                "gpio":   "P[A-I][0-9]{1,2}",
                "io":     "IO[0-9]+"
            }
        },
        {
            "id": "character_level_only",
            "title": "Character-Level Only",
            "rules": [
                "Only fix typical OCR confusions: O<->0, I/l<->1, S<->5, B<->8, Z<->2, g/q<->9.",
                "Do NOT rename or normalize; keep casing and style unless it's an obvious OCR confusion.",
                "Short tokens protection: if token length <= 3, RETURN UNCHANGED (to avoid over-correction)."
            ]
        },
        {
            "id": "confidence_guidance",
            "title": "Use Confidence (CONF) to limit edits",
            "rules": [
                "If CONF >= 0.92: at most 0–2 character substitutions from the allowed confusion set; never change length except replacing spaces with underscores.",
                "If 0.80 <= CONF < 0.92: minimal edits; length changes only when replacing spaces with underscores.",
                "If CONF < 0.80: still conservative; apply only clearly justified character-level fixes."
            ]
        }
    ],
    "notes": "When in doubt, keep the OCR token unchanged. Prefer the original style (case, separators) unless a clear OCR confusion exists."
}

# ---- Prompt References ---
REFERENCE_TOKENS_PATH = os.getenv(
    "REFERENCE_TOKENS_PATH",
    r"D:\Project\sampled_gts_unique_700_long_300_short.txt"  # 你上传的文件
)
REFERENCE_MAX_TOKENS = int(os.getenv("REFERENCE_MAX_TOKENS", "120"))

def _load_reference_tokens(path: str, max_n: int) -> List[str]:
    """从给定路径加载参考词表，用于 few-shot 提示。"""
    refs: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                refs.append(t)
                if len(refs) >= max_n:
                    break
    except Exception:
        # 如果文件不存在或无法读取，则静默失败，返回空列表
        pass
    return refs

# 在模块加载时读取一次参考词表
REF_TOKENS = _load_reference_tokens(REFERENCE_TOKENS_PATH, REFERENCE_MAX_TOKENS)
