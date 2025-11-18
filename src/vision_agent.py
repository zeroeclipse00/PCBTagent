import os
import base64
import logging
import time
import random
import re
from io import BytesIO
from typing import Dict, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --- API and basic config---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-ysUfU0FU4sA5eT1thCcgyEa4Z9q7Hy524Mtkvkib10h0kJFs")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1/chat/completions")
VISION_MODEL = "gpt-4o"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Aspect ratio threshold for orientation
ASPECT_RATIO_THRESHOLD = 1.18

def _exp_backoff_sleep(attempt: int, base: float = 1.0, jitter: float = 0.25) -> float:
    """
    Calculate exponential backoff seconds with jitter for API request retries.
    """
    delay = base * (2 ** attempt)
    return delay + random.uniform(0, base * jitter)

def call_gpt_vision(base64_image: str, max_retries: int = 3, timeout: int = 90) -> Tuple[str, int, int, float]:
    """
    Call the OpenAI Vision API and return the recognized text, token consumption, and backoff sleep time.

    Args:
        base64_image (str): The Base64 encoded string of the image.
        max_retries (int): Maximum number of retries.   
        timeout (int): Request timeout (seconds).

    Returns:
        Tuple[str, int, int, float]: 
            (recognized text, prompt_tokens, completion_tokens, total_sleep_time)
            if fail, return ("", 0, 0, total_sleep_time)
    """
    if not OPENAI_API_KEY:
        logger.error("The environment variable OPENAI_API_KEY is not set, API cannot be called.")
        return "", 0, 0, 0.0

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in OCR for PCB schematics. Accurately recognize the text in the image provided."
                            "Pay close attention to distinguishing between confusable characters, such as 'O' and '0', 'I'/'l' and '1', and 'S' and '5'."
                            "You must preserve all special symbols, such as Ω, µ, °, ±, _, and -."
                            "Return only the recognized text content. Do not add any explanations, quotation marks, newlines, or other extra characters."
                            )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100  # Limit the maximum output length to save tokens.
    }

    total_sleep_time = 0.0

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract the recognized content.
            content = data["choices"][0]["message"]["content"]
            content = content.strip().replace("`", "")
            
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            return content, prompt_tokens, completion_tokens, total_sleep_time
            
        except requests.RequestException as e:
            if attempt == max_retries:
                logger.error(f"Vision API request ultimately failed after {attempt} retries: {e}")
                return "", 0, 0, total_sleep_time # if fail
            
            sleep_s = _exp_backoff_sleep(attempt)
            total_sleep_time += sleep_s
            
            logger.warning(f"Vision API request failed (try {attempt+1}/{max_retries+1}), retry after {sleep_s:.1f}s: {e}")
            time.sleep(sleep_s)
            
    return "", 0, 0, total_sleep_time

# Image and data processing
def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode the Pillow image object into a Base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_line(line: str) -> Optional[Dict]:
    if "||" not in line:
        return None
    
    left_part, right_part = line.split("||", 1)
    
    tokens = left_part.strip().split()
    if len(tokens) < 6:
        return None
        
    try:
        class_id = tokens[0]
        bbox = tuple(map(float, tokens[1:5])) # (cx, cy, w, h)
        gt = " ".join(tokens[5:])
        original_prefix = " ".join(tokens[:5])
    except (ValueError, IndexError):
        return None

    right_part_stripped = right_part.strip()
    right_tokens = right_part_stripped.split()
    confidence = 0.0  # default
    ocr_text = ""     # default

    if not right_tokens:
        ocr_text = ""
        confidence = 0.0
    else:
        try:
            confidence = float(right_tokens[-1])
            ocr_text = " ".join(right_tokens[:-1])
        except ValueError:
            ocr_text = right_part_stripped
            confidence = 0.0
    
    return {
        "class_id": class_id, 
        "bbox": bbox, 
        "gt": gt, 
        "original_prefix": original_prefix,
        "ocr_text": ocr_text, 
        "confidence": confidence
    }

# Rule 1 (Hard Veto): Veto known, high-frequency systematic errors from the LLM
LLM_VETO_PATTERNS = [
    # Hard Veto (1): LLM easily misidentifies ...6 as ...8 
    # This rule matches patterns like R6, C6, R16, PC6, LED16, etc.
    re.compile(r'^[A-Z]{1,4}\d*6$'),
    
    # Hard Veto (2): LLM tends to 'correct' 'uF'/'UF' to 'µF'
    re.compile(r'\d+(\.\d+)?(uF|UF)$', re.IGNORECASE),
    
    # Hard Veto (3): LLM tends to add 'Ω' or change case for K-suffix resistors 
    re.compile(r'^\d+(\.\d+)?K$'),
    
    # Hard Veto (4): LLM tends to add a '0' to 2-digit resistor values (20Ω -> 200Ω)
    re.compile(r'^\d{2}Ω$'),
]

# Rule 2 (General Trigger): Only trigger on the most common, high-frequency, general error patterns from OCR
OCR_TRIGGER_PATTERNS = [
    # Trigger: O/0 confusion (PAO -> PA0, BOOTO -> BOOT0) 
    re.compile(r'\b(PA|PB|PC|BOOT|IIC|ADC|DP|DN|D|IO|BIT|FSMC_D|SDIO_D)O\b', re.IGNORECASE),
    
    # Trigger: Missing underscore (space or concatenation) 
    re.compile(r'^(USB|UART|SPI|IIC|LCD|FLASH|FSMC|EMMC|SDIO|GPIO) ', re.IGNORECASE),
    re.compile(r'\b(UART\dTX|UART\dRX|QSPISD|QSPISS|I2CSDA|SPISCK)\b', re.IGNORECASE),

    # Trigger: I/1 confusion (1019 -> IO19, GPI03 -> GPIO3)
    re.compile(r'^(10|1O|I0)\d{1,2}$'), # Matches 10xx, 1Oxx, I0xx
    re.compile(r'\b(GPI)O\d', re.IGNORECASE), # Matches GPlO...
    
    # Trigger: Common GND/VCC noise (GND -> PGND, -, c, etc. VCC -> vcc, cc)
    re.compile(r'^(PGND|vcc|vCC|cc|cC)$'),
    re.compile(r'^[-+cnom.]$'), # Single noise characters
]


def should_use_vision_api(ocr_text: str, confidence: float, confidence_threshold: float) -> bool:
    """
    Decides whether to call the Vision API or keep the OCR result based on confidence and rules.
    
    Args:
        ocr_text (str): Text recognized by PaddleOCR.
        confidence (float): Confidence score provided by PaddleOCR.
        confidence_threshold (float): User-set confidence threshold.

    Returns:
        bool: True to call Vision API, False to keep OCR result.
    """
    
    # Rule 1: Check "Veto" Rules (based on Cat 2)
    # Use the stricter Veto list to precisely target high-frequency LLM errors.
    for pattern in LLM_VETO_PATTERNS:
        if pattern.search(ocr_text):
            # Hit an LLM error-prone pattern (e.g., R6, 10uF, 10K, 20Ω)
            # Veto the API, use OCR.
            return False

    # Rule 2: Check "Low Confidence" Trigger (Core safety net)
    # If not vetoed and confidence is below threshold, OCR is unreliable, call API.
    if confidence < confidence_threshold:
        return True

    # Rule 3: Check "OCR Error-Prone" Triggers (based on Cat 3)
    # Reaching here means confidence is *high* (>= threshold) and *not* vetoed.
    # Use the general Trigger list to capture high-frequency OCR errors.
    for pattern in OCR_TRIGGER_PATTERNS:
        if pattern.search(ocr_text):
            # Hit an OCR error-prone pattern (e.g., PAO, vcc, "USB DP")
            # Even with high confidence, it's worth calling the API to check.
            return True

    # Default case: High confidence and no special rules were hit.
    # Trust the high-confidence OCR, do not call API.
    return False

# --- [END OF REPLACEMENT] ---

# Function now returns a statistics dictionary
def run_vision_agent(input_txt_path: str, image_path: str, output_txt_path: str, confidence_threshold: float) -> Optional[Dict]:
    """Main execution function: processes a single annotation file and its corresponding image, returning statistics."""
    
    start_time = time.time() # Record start time for processing a single file
    
    # Initialize statistics for the single file
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_sleep_time = 0.0 # API backoff sleep time
    
    # Check if input files exist (already checked in main loop, this is a double-check)
    if not os.path.exists(input_txt_path):
        logger.error(f"Input file does not exist: {input_txt_path}")
        return None
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return None

    # Load the original image
    try:
        with Image.open(image_path) as original_image:
            img_width, img_height = original_image.size
            logger.info(f"Successfully loaded image: {image_path} (Size: {img_width}x{img_height})")

            # Read all annotation lines
            with open(input_txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            output_lines = []
            
            # Use tqdm to create a progress bar
            for line in tqdm(lines, desc=f"Processing {os.path.basename(input_txt_path)}"):
                line = line.strip()
                if not line:
                    continue

                parsed_data = parse_line(line)
                if not parsed_data:
                    logger.warning(f"Skipping malformed line: {line}")
                    # Keep unparsable lines
                    output_lines.append(line)
                    continue
                
                # Get confidence and OCR text
                confidence = parsed_data.get("confidence", 0.0)
                ocr_text = parsed_data.get("ocr_text", "")

                # Use the new intelligent decision function to decide whether to call API
                if should_use_vision_api(ocr_text, confidence, confidence_threshold):
                    # Trigger API call (Reason: Low confidence or hit OCR error-prone rule)
                    # logger.info(f"OCR: '{ocr_text}' (Conf: {confidence:.2f}) -> Triggering Vision API") # Optional log
                    
                    # Unpack coordinates and convert to absolute pixel coordinates (left, top, right, bottom)
                    cx, cy, w, h = parsed_data["bbox"]
                    left = (cx - w / 2) * img_width
                    top = (cy - h / 2) * img_height
                    right = (cx + w / 2) * img_width
                    bottom = (cy + h / 2) * img_height

                    # Crop image
                    cropped_img = original_image.crop((left, top, right, bottom))
                    
                    # Check orientation and rotate
                    crop_w, crop_h = cropped_img.size
                    if crop_w > 0 and crop_h > 0 and crop_h / crop_w > ASPECT_RATIO_THRESHOLD:
                        # Rotate 90 degrees clockwise
                        cropped_img = cropped_img.rotate(angle=-90, expand=True)
                    
                    # Encode and call API
                    base64_image = encode_image_to_base64(cropped_img)
                    
                    # Receive all return values
                    vision_result, p_tokens, c_tokens, sleep_time = call_gpt_vision(base64_image)

                    # Accumulate statistics
                    total_api_calls += 1
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens
                    total_sleep_time += sleep_time

                    if not vision_result:
                        logger.warning(f"Failed to get Vision OCR result for line '{line}', will keep empty output.")
                    new_line = f"{parsed_data['original_prefix']} {parsed_data['gt']}||{vision_result}"
                    output_lines.append(new_line)
                
                else:
                    # Keep OCR (Reason: High confidence and no rules hit, or hit LLM error-prone rule)
                    new_line = f"{parsed_data['original_prefix']} {parsed_data['gt']}||{ocr_text}"
                    output_lines.append(new_line)
        
        # Ensure output directory exists (created in main function)
        output_dir = os.path.dirname(output_txt_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Write results file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
        
        end_time = time.time() # Record end time
        
        # Calculate and report statistics for the single file
        total_wall_time = end_time - start_time
        net_processing_time = total_wall_time - total_sleep_time
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        logger.info(f"Processing complete! Results written to: {output_txt_path}")
        logger.info(f"--- [File Statistics: {os.path.basename(input_txt_path)}] ---")
        logger.info(f"  Total API Calls: {total_api_calls}")
        logger.info(f"  Total Tokens: {total_tokens} (Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens})")
        logger.info(f"  Total Wall Time: {total_wall_time:.2f} s")
        logger.info(f"  API Sleep Time (Excluded): {total_sleep_time:.2f} s")
        logger.info(f"  Net Processing Time: {net_processing_time:.2f} s")
        
        # Return statistics dictionary
        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "api_calls": total_api_calls,
            "sleep_time": total_sleep_time,
            "wall_time": total_wall_time
        }

    except FileNotFoundError:
        logger.error(f"Could not find or open image file: {image_path}")
        return None
    except UnidentifiedImageError:
        logger.error(f"Unidentified image format, please ensure it is a valid image file: {image_path}")
        return None
    except Exception as e:
        logger.critical(f"An unknown error occurred while processing file '{input_txt_path}': {e}", exc_info=True)
        return None
if __name__ == "__main__":
    # --- Folder Paths and Threshold Configuration ---
    # --- Paths can now point to a folder (batch processing) or a single file (.txt / .png) ---
    
    # Input annotation path (can be a folder or a single .txt file)
    INPUT_TEXT_PATH = r"data/agent_input_text"
    
    # Corresponding original image path (can be a folder or a single .png file)
    INPUT_IMAGE_PATH = r"data/agent_input_picture"
    
    # Output folder path for processed results
    OUTPUT_DIR = r"data/vision_output"

    # Confidence Threshold:
    # This is the base threshold.
    # 1. When PaddleOCR confidence < this value, trigger API (Rule 2)
    # 2. When PaddleOCR confidence >= this value, it will trigger Veto (Rule 1) or Trigger (Rule 3) checks.
    CONFIDENCE_THRESHOLD = 0.95

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize global statistics variables
    files_processed = 0
    grand_total_prompt_tokens = 0
    grand_total_completion_tokens = 0
    grand_total_tokens = 0
    grand_total_api_calls = 0
    grand_total_sleep_time = 0.0
    grand_total_wall_time = 0.0
    
    # Determine if input path is a folder or a file
    # Mode 1: If input is a folder, execute batch processing
    if os.path.isdir(INPUT_TEXT_PATH):
        logger.info(f"Detected folder as input path, starting batch processing mode: '{INPUT_TEXT_PATH}'")
        
        # In batch mode, the image path must also be a folder
        if not os.path.isdir(INPUT_IMAGE_PATH):
            logger.error(f"Input text is a folder, but image path is not a folder: '{INPUT_IMAGE_PATH}'")
            logger.error("Batch processing mode failed.")
        else:
            # --- Start of batch processing logic (same as original script) ---
            try:
                text_files = [f for f in os.listdir(INPUT_TEXT_PATH) if f.endswith(".txt")]
                if not text_files:
                    logger.info(f"No .txt files found in directory '{INPUT_TEXT_PATH}'.")
                else:
                    logger.info(f"Found {len(text_files)} .txt files in '{INPUT_TEXT_PATH}', starting batch processing...")
                    
                    # Iterate over all .txt files in the input text directory
                    for txt_filename in sorted(text_files): # sorted() ensures processing order
                        base_name = os.path.splitext(txt_filename)[0]
                        
                        # Construct full file paths
                        input_txt_path = os.path.join(INPUT_TEXT_PATH, txt_filename)
                        image_path = os.path.join(INPUT_IMAGE_PATH, f"{base_name}.png")
                        output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
                        
                        # Check if the corresponding image file exists
                        if not os.path.exists(image_path):
                            logger.warning(f"Could not find corresponding image file '{image_path}', skipping processing for '{input_txt_path}'.")
                            continue
                        
                        # Call main function and receive statistics
                        logger.info(f"--- Starting processing: {txt_filename} ---")
                        stats = run_vision_agent(input_txt_path, image_path, output_txt_path, CONFIDENCE_THRESHOLD)
                        
                        # Accumulate global statistics
                        if stats:
                            files_processed += 1
                            grand_total_prompt_tokens += stats["prompt_tokens"]
                            grand_total_completion_tokens += stats["completion_tokens"]
                            grand_total_tokens += stats["total_tokens"]
                            grand_total_api_calls += stats["api_calls"]
                            grand_total_sleep_time += stats["sleep_time"]
                            grand_total_wall_time += stats["wall_time"]
                        
                        logger.info(f"--- Finished processing: {txt_filename} ---\n")
                    
                    logger.info("All files processed!")

            except Exception as e:
                logger.critical(f"A fatal error occurred during batch processing: {e}", exc_info=True)
            # --- End of batch processing logic ---

    # Mode 2: If input is a file, execute single file processing
    elif os.path.isfile(INPUT_TEXT_PATH):
        logger.info(f"Detected file as input path, starting single file processing mode: '{INPUT_TEXT_PATH}'")
        
        # In single file mode, the image path must also be a file
        if not os.path.isfile(INPUT_IMAGE_PATH):
            logger.error(f"Input text is a file, but image path is not a file: '{INPUT_IMAGE_PATH}'")
            logger.error("Single file processing mode failed.")
        else:
            # --- Start of single file processing logic ---
            try:
                # 1. Get the filename from the full path
                txt_filename = os.path.basename(INPUT_TEXT_PATH)
                # 2. Construct the full output file path
                output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
                
                logger.info(f"--- Starting processing single file: {txt_filename} ---")
                
                # Call main function and receive statistics
                stats = run_vision_agent(
                    input_txt_path=INPUT_TEXT_PATH,     # Use the configured .txt file path
                    image_path=INPUT_IMAGE_PATH,      # Use the configured .png file path
                    output_txt_path=output_txt_path, 
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )
                
                # Accumulate global statistics
                if stats:
                    files_processed += 1
                    grand_total_prompt_tokens = stats["prompt_tokens"]
                    grand_total_completion_tokens = stats["completion_tokens"]
                    grand_total_tokens = stats["total_tokens"]
                    grand_total_api_calls = stats["api_calls"]
                    grand_total_sleep_time = stats["sleep_time"]
                    grand_total_wall_time = stats["wall_time"]
                
                logger.info(f"--- Finished processing: {txt_filename} ---")
                logger.info("Single file processing complete!")
                
            except Exception as e:
                logger.critical(f"A fatal error occurred while processing file '{txt_filename}': {e}", exc_info=True)
            # --- End of single file processing logic ---

    # Mode 3: If path is invalid
    else:
        logger.error(f"Input path is neither a valid file nor a valid directory: '{INPUT_TEXT_PATH}'")
        logger.error("Program exited.")

    # --- Final Global Statistics Report ---
    if files_processed > 0:
        grand_total_net_time = grand_total_wall_time - grand_total_sleep_time
        
        logger.info("==========================================================")
        logger.info(f"--- Task Summary (Processed {files_processed} files) ---")
        logger.info("==========================================================")
        logger.info(f"  [Calls] Total API Calls: {grand_total_api_calls}")
        logger.info(f"  [Tokens] Total Consumed: {grand_total_tokens}")
        logger.info(f"      - Prompt: {grand_total_prompt_tokens}")
        logger.info(f"      - Completion: {grand_total_completion_tokens}")
        logger.info(f"  [Time] Total Wall Time: {grand_total_wall_time:.2f} s")
        logger.info(f"  [Time] API Sleep Time (Excluded): {grand_total_sleep_time:.2f} s")
        logger.info(f"  [Time] Net Processing Time: {grand_total_net_time:.2f} s")
        avg_net_time = grand_total_net_time / files_processed
        logger.info(f"  [Perf] Avg. Net Processing Time: {avg_net_time:.2f} s/file")
        if grand_total_api_calls > 0:
            avg_token_per_call = grand_total_tokens / grand_total_api_calls
            logger.info(f"  [Perf] Avg. Tokens/Call: {avg_token_per_call:.1f} tokens")
        logger.info("==========================================================")
    elif not os.path.isdir(INPUT_TEXT_PATH) and not os.path.isfile(INPUT_TEXT_PATH):
        pass # Invalid path, already reported error
    else:
        logger.info("--- Task Summary ---")
        logger.info("No files were processed successfully.")