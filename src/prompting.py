"""
Build a Prompt to send to the LLM.
"""

import json
from typing import List, Dict
from config import REFERENCE_TOKENS, REFERENCE_MAX_TOKENS, RAG_KB_PATH
# from references import *

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
    kb_snippet = json.dumps(str(RAG_KB_PATH), ensure_ascii=False)

    # Reference word list (sampled from the file, possibly empty)
    ref_head = ""
    if REFERENCE_TOKENS:
        # Control the length of the context and put a line with commas to save more tokens
        ref_head = "Reference tokens (correct examples; mimic style when similar):\n" + ", ".join(
            REFERENCE_TOKENS[:REFERENCE_MAX_TOKENS]
        ) + "\n\n"

    system_msg = (
        "You are a senior PCB schematic engineer and OCR correction expert.\n"
        "Task: Fix noisy OCR tokens for schematic labels using CONSERVATIVE, CHARACTER-LEVEL edits only.\n"
        "Allowed swaps: O<->0, I/l<->1, S<->5, B<->8, Z<->2, g/q<->9.\n"
        "Hard constraints:\n"
        " - Replace internal spaces with underscores (_); never introduce spaces.\n"
        " - Do NOT replace '-' with '_' or vice versa.\n"
        " - Do NOT remove unit/symbols like Ω, µ, °, ±.\n"
        " - Do NOT convert 3.3V <-> 3V3.\n"
        " - Keep *_P/*_N suffixes and explicit +/- in diff pairs.\n"
        "Length rule: if token length <= 2, RETURN THE OCR TOKEN UNCHANGED.\n"
        "Confidence rule:\n"
        " - If CONF >= 0.92: at most 0–2 confusable-character substitutions; length must not change (except spaces->underscores).\n"
        " - If 0.80 <= CONF < 0.92: minimal edits; length change only for spaces->underscores.\n"
        " - If CONF < 0.80: still conservative; only clear OCR confusions are allowed.\n"
        "If the OCR token already looks valid or you're unsure, return it unchanged.\n"
        "Additional guidance:\n"
        " - If OCR token matches 'PAB', correct it to 'PA6' or 'PA8' based on the number in context.\n"
        " - For any 'GP' or 'GPIO' confusion, prioritize 'GPIO' over 'GP' and fix errors like 'GP108' to 'GPIO8'.\n"
        " - If the OCR token matches patterns like 'PAB', 'PAC', or 'PAA', consider it a potential misreading of 'PA6' or 'PA8'.\n"
        "Output MUST be tokens only, one per line, exactly matching input order. No numbering, no quotes, no extra text."
    )

    header = (
        ref_head +
        "Knowledge Base (context only; do not over-normalize):\n" +
        f"{kb_snippet}\n\n" +
        "Correct the following OCR tokens.\n"
        "If GT is provided, use it only to guide character types/positions (TYPE_MASK = A/D/S).\n"
        "Return ONE token per line, same order as input.\n"
    )

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

    user_msg = header + "\n".join(lines)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
