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

def build_prompt(batch_items: List[Dict], include_gt: bool) -> List[Dict]:
    """
    Build a Prompt for OCR post-processing.
    """
    with open(RAG_KB_PATH, "r", encoding="utf-8") as f:
        kb_content = json.load(f)
    kb_snippet = json.dumps(kb_content, ensure_ascii=False, indent=2)

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

        if include_gt and item.get("gt"):
            tm = _type_mask_string(item["gt"])
            if conf is None:
                lines.append(f"- OCR: {pred} ; LEN: {L} ; GT: {item['gt']} ; TYPE_MASK: {tm}")
            else:
                lines.append(f"- OCR: {pred} ; LEN: {L} ; CONF: {conf:.4f} ; GT: {item['gt']} ; TYPE_MASK: {tm}")
        else:
            if conf is None:
                lines.append(f"- OCR: {pred} ; LEN: {L}")
            else:
                lines.append(f"- OCR: {pred} ; LEN: {L} ; CONF: {conf:.4f}")

    user_msg = header + "\n" + "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
