"""
Build a Prompt to send to the LLM.
"""

import json
from typing import List, Dict
from config import REFERENCE_TOKENS, REFERENCE_MAX_TOKENS, RAG_KB_PATH
from prompt_texts import SYSTEM_MSG, HEADER_TEMPLATE

def _type_mask_string(tok: str) -> str:
    """When include_gt is True, A type mask (A=alpha, D=digit, S=symbol) is generated for the token to guide character-level corrections."""
    if not tok:
        return ""
    mask = []
    for ch in tok:
        if ch.isdigit():
            mask.append('D')
        elif ch.isalpha():
            mask.append('A')
        else:
            mask.append('S')
    return ''.join(mask)

# spn: 10.12 add
def _lock_len_tag(conf: float | None, L: int) -> str:
    """
    基于长度与置信度给出长度锁定提示：
      - L <= 2:        HARD  （严格不改长度）
      - conf >= 0.92:  HARD
      - 0.80 <= conf < 0.92: SOFT
      - else:          NONE
    """
    if L <= 2:
        return "HARD"
    if conf is None:
        return "NONE"
    if conf >= 0.92:
        return "HARD"
    if conf >= 0.80:
        return "SOFT"
    return "NONE"

def build_prompt(batch_items: List[Dict], include_gt: bool) -> List[Dict]:
    """
    Build a Prompt for OCR post-processing.
    """
    kb_snippet = json.dumps(str(RAG_KB_PATH), ensure_ascii=False)

    # Reference word list (sampled from the file, possibly empty)
    ref_head = ""
    if REFERENCE_TOKENS:
        # Control the length of the context and put a line with commas to save more tokens
        ref_head = "Reference tokens (correct examples; mimic style when similar):\n" + ", ".join(
            REFERENCE_TOKENS[:REFERENCE_MAX_TOKENS]
        ) + "\n\n"

    header = ref_head + HEADER_TEMPLATE.format(KB_SNIPPET=kb_snippet)

    lines = []
    for item in batch_items:
        pred = item.get("pred", "")
        L = len(pred)
        conf = item.get("conf", None)
        gt = item.get("gt", None)

        lock_len = _lock_len_tag(conf, L)
        ocr_mask = _type_mask_string(pred)

        if include_gt and item.get("gt"):
            gt_mask = _type_mask_string(gt)
            lines.append(
                f"- OCR: {pred} ; LEN: {L} ; CONF: {conf if conf is not None else 'N/A'} ; "
                f"GT: {gt} ; TYPE_MASK_OCR: {ocr_mask} ; TYPE_MASK_GT: {gt_mask} ; LOCK_LEN: {lock_len}"
            )
        else:
            lines.append(
                f"- OCR: {pred} ; LEN: {L} ; CONF: {conf if conf is not None else 'N/A'} ; "
                f"TYPE_MASK_OCR: {ocr_mask} ; LOCK_LEN: {lock_len}"
            )

    user_msg = header + "\n" + "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
