"""
The core processing flow includes the logic for handling individual files and entire folders.
"""

import os
import time
import math
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from utils.parser import *
from llm_clients import call_gpt_chat, call_deepseek_chat
from prompting import build_prompt

logger = logging.getLogger("pcb-ocr-corrector.pipeline")

BATCH_RETRY_ATTEMPTS = 3

def correct_batch(items: List[Dict], provider: str, include_gt: bool) -> List[Optional[str]]:
    """
    Correct the data of a batch.
    If the number of lines returned by the LLM is incorrect, it will automatically retry up to BATCH_RETRY_ATTEMPTS.
    """
    messages = build_prompt(items, include_gt)
    expected_n = len(items)

    for attempt in range(BATCH_RETRY_ATTEMPTS):
        if attempt > 0:
            logger.info(f"Retrying batch... (Attempt {attempt + 1}/{BATCH_RETRY_ATTEMPTS})")

        if provider == "gpt":
            raw_output = call_gpt_chat(messages)
        elif provider == "deepseek":
            raw_output = call_deepseek_chat(messages)
        else:
            raise ValueError("provider must be 'gpt' or 'deepseek'")

        corrected_list = postprocess_llm_block(raw_output, expected_n=expected_n)

        if corrected_list is not None:
            return corrected_list
        
        # Failure: corrected_list is None, and the loop will continue with the next retry
        # Before retrying, briefly sleep to avoid making requests too frequently
        if attempt < BATCH_RETRY_ATTEMPTS - 1:
            time.sleep(1)

    # If all retries fail
    logger.error(
        f"Failed to get a valid response for a batch after {BATCH_RETRY_ATTEMPTS} attempts. "
        "Falling back to original OCR values for this entire batch."
    )
    # Return a None list to allow the upstream to handle the rollback logic
    return [None] * expected_n


def process_file(input_path: str, output_path: str,
                 provider: str, batch_size: int,
                 threshold: float,
                 include_gt_in_prompt: bool):
    """
    Read the input txt file, correct the low-confidence items in batches, and write the output with additional columns to the txt file.
    """
    logger.info(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    n_lines = len(raw_lines)
    logger.info(f"Total lines: {n_lines}")

    # Analyze and collect low-confidence items
    parsed: List[Dict] = []
    to_fix: List[Dict] = []

    iterable = enumerate(raw_lines)
    iterable = tqdm(iterable, total=n_lines, desc="Parse", leave=False)
    
    for i, line in iterable:
        # Updated: The parse_line return value does not contain left_prefix
        gt, pred, conf = parse_line(line)
        parsed.append({
            "i": i, "gt": gt, "pred": pred, "conf": conf
        })
        if conf < threshold:
            to_fix.append({"idx": i, "pred": pred, "gt": gt, "conf": conf})

    n_fix = len(to_fix)
    logger.info(f"Low-confidence items (<{threshold:.2f}): {n_fix}")
    if n_fix == 0:
        logger.info("Nothing to correct. Writing passthrough output.")
        with open(output_path, "w", encoding="utf-8") as file:
            for rec in parsed:
                # Updated: The rebuild_line call does not contain left_prefix
                file.write(rebuild_line(rec["gt"], rec["pred"], rec["conf"], rec["pred"]) + "\n")
        logger.info(f"Done. Wrote: {output_path}")
        return

    # Batch processing progress bar
    n_batches = math.ceil(n_fix / batch_size)
    batch_iter = list(chunked(to_fix, batch_size))

    pbar = None
    pbar = tqdm(total=n_fix, desc=f"LLM({provider})", unit="tok", leave=True)

    idx_to_corrected: Dict[int, str] = {}
    start_ts = time.time()

    try:
        for b_idx, batch in enumerate(batch_iter, start=1):
            t0 = time.time()
            corrected = correct_batch(batch, provider=provider, include_gt=include_gt_in_prompt)
            for item, corr in zip(batch, corrected):
                # If corr is None (due to retry failure) or an empty string, fall back to the original pred
                idx_to_corrected[item["idx"]] = corr if corr is not None and corr != "" else item["pred"]
            dt = time.time() - t0
            
            if pbar:
                pbar.update(len(batch))
                pbar.set_postfix({"batch": f"{b_idx}/{n_batches}", "sec": f"{dt:.1f}"})
            else:
                done = b_idx * batch_size if b_idx < n_batches else n_fix
                logger.info(f"Batch {b_idx}/{n_batches} done in {dt:.1f}s ({done}/{n_fix} items)")

        total_dt = time.time() - start_ts
        ips = n_fix / total_dt if total_dt > 0 else 0.0
        logger.info(f"LLM correction finished in {total_dt:.1f}s, throughput {ips:.2f} item/s")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl-C). Writing partial results...")
    finally:
        if pbar:
            pbar.close()

    # Build the output line
    out_lines = []
    for rec in parsed:
        corrected_token = idx_to_corrected.get(rec["i"], rec["pred"])
        # Updated: The rebuild_line call does not contain left_prefix
        out_lines.append(rebuild_line(rec["gt"], rec["pred"], rec["conf"], corrected_token))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    logger.info(f"Wrote output: {output_path} (processed {len(idx_to_corrected)}/{n_fix} low-confidence items)")

def process_folder(input_dir: str, output_dir: str, **kwargs):
    """
    Iterate over all.txt files under input_dir and run process_file for each file.
    Write the output to output_dir with the same file name.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted([
        f for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(".txt")
    ])

    if not txt_files:
        logger.warning(f"No .txt files found in: {input_dir}")
        return

    logger.info(f"Found {len(txt_files)} txt file(s) in {input_dir}. Outputting to {output_dir}.")

    for idx, fn in enumerate(txt_files, start=1):
        in_path = os.path.join(input_dir, fn)
        out_path = os.path.join(output_dir, fn)
        logger.info(f"--- Processing file {idx}/{len(txt_files)}: {fn} ---")
        process_file(input_path=in_path, output_path=out_path, **kwargs)

        if idx < len(txt_files):
            sleep_duration = 10
            logger.info(f"Sleeping {sleep_duration}s before next file...")
            time.sleep(sleep_duration)
            
    logger.info("Folder processing complete.")
