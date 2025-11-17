# -*- coding: utf-8 -*-
"""
Vision Agent 批量处理脚本

功能：
- 遍历指定的标注文件夹（agent_input_text），处理其中所有的 .txt 文件。
- 自动从图像文件夹（agent_input_picture）中查找与 .txt 文件同名的 .png 图像。
- 对每个标注文件中的每一行，解析 PaddleOCR 的识别结果和置信度。
- **当置信度低于设定的阈值时**，根据标注的边界框（bounding box）从对应的原始图像中裁剪出小图。
- 自动检测小图方向，并将竖排文本旋转至水平。
- 调用 GPT-4o-mini API 对校正后的小图进行文字识别。
- 将识别结果（或置信度达标的原始结果）以指定格式写入输出文件夹。
- [!! 新增] 统计每个文件的处理时间（排除API等待）、API调用次数和Token消耗，并在最后进行汇总。

使用前准备：
1. 修改下方的 '文件路径配置' 和 '置信度阈值' 部分，填入你的实际参数。
2. 安装必要的 Python 库:
   pip install Pillow requests tqdm
3. 设置环境变量:
   需要设置您的 OpenAI API 密钥。建议创建一个 .env 文件，内容如下：
   OPENAI_API_KEY="sk-..."
   OPENAI_BASE_URL="https://api.openai.com/v1" # 或者您的代理地址
"""

import os
import base64
import logging
import time
import random
import re  # 导入正则表达式库
from io import BytesIO
from typing import Dict, Optional, Tuple # [!! 修改] 导入 Tuple

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --- API 与基础配置 ---

# 从环境变量加载 API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-ysUfU0FU4sA5eT1thCcgyEa4Z9q7Hy524Mtkvkib10h0kJFs")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1/chat/completions")
VISION_MODEL = "gpt-4o" # 脚本此处模型写的是 gpt-4o
# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

# 方向判断的宽高比阈值
ASPECT_RATIO_THRESHOLD = 1.18


# --- API 调用封装 ---

def _exp_backoff_sleep(attempt: int, base: float = 1.0, jitter: float = 0.25) -> float:
    """计算带抖动的指数退避秒数，用于 API 请求重试。"""
    delay = base * (2 ** attempt)
    return delay + random.uniform(0, base * jitter)

# [!! 修改] 函数签名和返回值已更新，以包含 Token 计数和睡眠时间
def call_gpt_vision(base64_image: str, max_retries: int = 3, timeout: int = 90) -> Tuple[str, int, int, float]:
    """
    调用 OpenAI Vision API 并返回识别的文本、Token 消耗和退避睡眠时间。

    Args:
        base64_image (str): 图像的 Base64 编码字符串。
        max_retries (int): 最大重试次数。
        timeout (int): 请求超时时间（秒）。

    Returns:
        Tuple[str, int, int, float]: 
            (识别的文本, prompt_tokens, completion_tokens, total_sleep_time)
            失败则返回 ("", 0, 0, total_sleep_time)
    """
    if not OPENAI_API_KEY:
        logger.error("环境变量 OPENAI_API_KEY 未设置，无法调用 API。")
        return "", 0, 0, 0.0 # [!! 修改] 返回带统计的元组

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
        "max_tokens": 100  # 限制最大输出长度，节省 token
    }

    total_sleep_time = 0.0 # [!! 新增] 初始化总睡眠时间

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENAI_BASE_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            # 提取识别内容
            content = data["choices"][0]["message"]["content"]
            content = content.strip().replace("`", "") # 清理 API 可能返回的额外字符
            
            # [!! 新增] 提取 usage 统计
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # [!! 修改] 返回所有信息
            return content, prompt_tokens, completion_tokens, total_sleep_time
            
        except requests.RequestException as e:
            if attempt == max_retries:
                logger.error(f"Vision API 请求在 {attempt} 次重试后最终失败: {e}")
                return "", 0, 0, total_sleep_time # [!! 修改] 失败时返回
            
            sleep_s = _exp_backoff_sleep(attempt)
            total_sleep_time += sleep_s # [!! 新增] 累加睡眠时间
            
            logger.warning(f"Vision API 请求错误 (尝试 {attempt+1}/{max_retries+1})，{sleep_s:.1f}s 后重试: {e}")
            time.sleep(sleep_s)
            
    return "", 0, 0, total_sleep_time # [!! 修改] 循环结束仍失败时返回

# --- 图像与数据处理 ---

def encode_image_to_base64(image: Image.Image) -> str:
    """将 Pillow 图像对象编码为 Base64 字符串。"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_line(line: str) -> Optional[Dict]:
    """
    【已修改】
    解析 'class_id x y w h gt||ocr 置信度' 格式的行。
    现在会额外返回 ocr_text 字段。
    """
    if "||" not in line:
        return None
    
    left_part, right_part = line.split("||", 1)
    
    # 解析左侧标注信息
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

    # --- 【新】解析右侧 OCR 结果和置信度 ---
    right_part_stripped = right_part.strip()
    right_tokens = right_part_stripped.split()
    confidence = 0.0  # 默认置信度
    ocr_text = ""       # 默认 OCR 文本

    if not right_tokens:
        # '||' 后面为空，ocr_text 为空，confidence 为 0.0
        ocr_text = ""
        confidence = 0.0
    else:
        try:
            # 尝试将最后一个 token 转换为浮点数作为置信度
            confidence = float(right_tokens[-1])
            # 如果成功，前面的就是 ocr_text
            ocr_text = " ".join(right_tokens[:-1])
        except ValueError:
            # 如果最后一个 token 不是数字，说明整行都是 ocr_text，没有置信度。
            # 当作低置信度处理，触发API调用。
            ocr_text = right_part_stripped
            confidence = 0.0
    
    return {
        "class_id": class_id, 
        "bbox": bbox, 
        "gt": gt, 
        "original_prefix": original_prefix,
        "ocr_text": ocr_text, # <-- 新增字段
        "confidence": confidence
    }

# --- 【新】智能决策规则 ---
# [基于用户反馈，采用“平衡型”规则集]
# [结合强硬的Veto规则和通用的Trigger规则]

# 规则 1 (强硬-否决): 否决 LLM 已知的、极高频的系统性错误 (来自 Cat 2)
LLM_VETO_PATTERNS = [
    # 强硬否决 (1): LLM 极易将 ...6 识别为 ...8 
    # 此规则匹配如 R6, C6, R16, PC6, LED16 等模式
    re.compile(r'^[A-Z]{1,4}\d*6$'),
    
    # 强硬否决 (2): LLM 易将 'uF'/'UF' 修正为 'µF'
    re.compile(r'\d+(\.\d+)?(uF|UF)$', re.IGNORECASE),
    
    # 强硬否决 (3): LLM 易给 K 结尾的电阻值加 'Ω' 或改大小写 
    re.compile(r'^\d+(\.\d+)?K$'),
    
    # 强硬否决 (4): LLM 易给 2 位电阻值加 '0' (20Ω -> 200Ω)
    re.compile(r'^\d{2}Ω$'),
]

# 规则 2 (通用-触发): 仅触发 OCR 最常见、最高频、最通用的错误模式 (来自 Cat 3, Cat 4)
# [此列表与上一版 "保守型" 策略相同，保持稳定性]
OCR_TRIGGER_PATTERNS = [
    # 触发: O/0 混淆 (PAO -> PA0, BOOTO -> BOOT0) 
    re.compile(r'\b(PA|PB|PC|BOOT|IIC|ADC|DP|DN|D|IO|BIT|FSMC_D|SDIO_D)O\b', re.IGNORECASE),
    
    # 触发: 缺少下划线 (空格或粘连) 
    re.compile(r'^(USB|UART|SPI|IIC|LCD|FLASH|FSMC|EMMC|SDIO|GPIO) ', re.IGNORECASE),
    re.compile(r'\b(UART\dTX|UART\dRX|QSPISD|QSPISS|I2CSDA|SPISCK)\b', re.IGNORECASE),

    # 触发: I/1 混淆 (1019 -> IO19, GPI03 -> GPIO3)
    re.compile(r'^(10|1O|I0)\d{1,2}$'), # 匹配 10xx, 1Oxx, I0xx
    re.compile(r'\b(GPI)O\d', re.IGNORECASE), # 匹配 GPlO...
    
    # 触发: 常见的 GND/VCC 噪声 (GND -> PGND, -, c, etc. VCC -> vcc, cc)
    re.compile(r'^(PGND|vcc|vCC|cc|cC)$'),
    re.compile(r'^[-+cnom.]$'), # 单个噪声字符
]


def should_use_vision_api(ocr_text: str, confidence: float, confidence_threshold: float) -> bool:
    """
    【"平衡型"智能决策函数】
    根据置信度和"平衡型"规则，决定是否应调用 Vision API 或保留 OCR 结果。
    
    Args:
        ocr_text (str): PaddleOCR 识别出的文本。
        confidence (float): PaddleOCR 提供的置信度。
        confidence_threshold (float): 用户设置的置信度阈值。

    Returns:
        bool: True 表示调用 Vision API，False 表示保留 OCR 结果。
    """
    
    # 规则 1：检查“否决”规则 (Veto Rules, 基于 Cat 2)
    # [新] 采用更强硬的 Veto 列表，精准狙击 LLM 的高频错误。
    for pattern in LLM_VETO_PATTERNS:
        if pattern.search(ocr_text):
            # 命中 LLM 易错模式 (如 R6, 10uF, 10K, 20Ω)
            # 否决 API，使用 OCR。
            return False

    # 规则 2：检查“低置信度”触发器 (核心安全网)
    # 如果没被否决，且置信度低于阈值，说明 OCR 不可信，调用 API。
    if confidence < confidence_threshold:
        return True

    # 规则 3：检查“OCR易错模式”触发器 (基于 Cat 3)
    # 运行到这里，说明置信度 *高* (>= threshold)，且 *未* 命中否决规则。
    # [旧] 采用通用的 Trigger 列表，捕获 OCR 的高频错误。
    for pattern in OCR_TRIGGER_PATTERNS:
        if pattern.search(ocr_text):
            # 命中 OCR 易错模式 (如 PAO, vcc, "USB DP")
            # 即使置信度高，也值得调用 API 试一试。
            return True

    # 默认情况：置信度高，且没有命中任何特殊规则。
    # 相信高置信度的 OCR，不调用 API。
    return False

# --- 【替换结束】 ---

# [!! 修改] 函数现在返回一个统计字典
def run_vision_agent(input_txt_path: str, image_path: str, output_txt_path: str, confidence_threshold: float) -> Optional[Dict]:
    """主执行函数：处理单个标注文件及其对应的图像，并返回统计数据。"""
    
    start_time = time.time() # [!! 新增] 记录单个文件处理开始时间
    
    # [!! 新增] 初始化单个文件的统计数据
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_sleep_time = 0.0 # API 退避睡眠时间
    
    # 检查输入文件是否存在 (在主循环中已检查，此处为双重保险)
    if not os.path.exists(input_txt_path):
        logger.error(f"输入文件不存在: {input_txt_path}")
        return None # [!! 修改] 返回 None 表示处理失败
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return None # [!! 修改] 返回 None 表示处理失败

    # 加载原始图像
    try:
        with Image.open(image_path) as original_image:
            img_width, img_height = original_image.size
            logger.info(f"成功加载图像: {image_path} (尺寸: {img_width}x{img_height})")

            # 读取所有标注行
            with open(input_txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            output_lines = []
            
            # 使用 tqdm 创建进度条
            for line in tqdm(lines, desc=f"Processing {os.path.basename(input_txt_path)}"):
                line = line.strip()
                if not line:
                    continue

                parsed_data = parse_line(line)
                if not parsed_data:
                    logger.warning(f"跳过格式不正确的行: {line}")
                    # 保留无法解析的行
                    output_lines.append(line)
                    continue
                
                # 【新】获取置信度和 OCR 文本
                confidence = parsed_data.get("confidence", 0.0)
                ocr_text = parsed_data.get("ocr_text", "") # 新增

                # --- 【核心逻辑修改】---
                # 使用新的智能决策函数判断是否调用 API
                if should_use_vision_api(ocr_text, confidence, confidence_threshold):
                    # 触发 API 调用 (原因: 置信度低 或 命中OCR易错规则)
                    # logger.info(f"OCR: '{ocr_text}' (Conf: {confidence:.2f}) -> 触发 Vision API") # 可选日志
                    
                    # 解包坐标并转换为绝对像素坐标 (left, top, right, bottom)
                    cx, cy, w, h = parsed_data["bbox"]
                    left = (cx - w / 2) * img_width
                    top = (cy - h / 2) * img_height
                    right = (cx + w / 2) * img_width
                    bottom = (cy + h / 2) * img_height

                    # 裁剪图像
                    cropped_img = original_image.crop((left, top, right, bottom))
                    
                    # 判断方向并旋转
                    crop_w, crop_h = cropped_img.size
                    if crop_w > 0 and crop_h > 0 and crop_h / crop_w > ASPECT_RATIO_THRESHOLD:
                        # 顺时针旋转90度
                        cropped_img = cropped_img.rotate(angle=-90, expand=True)
                    
                    # 编码并调用 API
                    base64_image = encode_image_to_base64(cropped_img)
                    
                    # [!! 修改] 接收所有返回值
                    vision_result, p_tokens, c_tokens, sleep_time = call_gpt_vision(base64_image)

                    # [!! 新增] 累加统计数据
                    total_api_calls += 1
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens
                    total_sleep_time += sleep_time

                    if not vision_result:
                        logger.warning(f"未能获取行 '{line}' 的 Vision OCR 结果，将保留空输出。")
                    
                    # 重建输出行: class_id x y w h gt||vision_result
                    # 这会用 vision_result 替换掉原有的 'ocr 置信度'
                    new_line = f"{parsed_data['original_prefix']} {parsed_data['gt']}||{vision_result}"
                    output_lines.append(new_line)
                
                else:
                    # 保留 OCR (原因: 置信度高且未命中规则，或命中LLM易错规则)
                    
                    # --- 【Bug 修复】 ---
                    # 之前: output_lines.append(line) # 错误：这会保留原始行，包含置信度
                    # 修复: 同样使用解析出的数据重建行，但使用 ocr_text，以去除置信度
                    new_line = f"{parsed_data['original_prefix']} {parsed_data['gt']}||{ocr_text}"
                    output_lines.append(new_line)
                    # --- 【修复结束】 ---
        
        # 确保输出目录存在 (在主函数中已创建)
        output_dir = os.path.dirname(output_txt_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 写入结果文件
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
        
        end_time = time.time() # [!! 新增] 记录结束时间
        
        # [!! 新增] 计算并报告单个文件的统计数据
        total_wall_time = end_time - start_time
        net_processing_time = total_wall_time - total_sleep_time
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        logger.info(f"处理完成！结果已写入: {output_txt_path}")
        logger.info(f"--- [文件统计: {os.path.basename(input_txt_path)}] ---")
        logger.info(f"  总API调用次数: {total_api_calls} 次")
        logger.info(f"  总Token消耗: {total_tokens} (Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens})")
        logger.info(f"  总墙上时间: {total_wall_time:.2f} 秒")
        logger.info(f"  API等待时间: {total_sleep_time:.2f} 秒 (已排除)")
        logger.info(f"  净处理时间: {net_processing_time:.2f} 秒")
        
        # [!! 新增] 返回统计字典
        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "api_calls": total_api_calls,
            "sleep_time": total_sleep_time,
            "wall_time": total_wall_time
        }

    except FileNotFoundError:
        logger.error(f"无法找到或打开图像文件: {image_path}")
        return None # [!! 修改] 返回 None
    except UnidentifiedImageError:
        logger.error(f"无法识别的图像文件格式，请确保是有效的图片文件: {image_path}")
        return None # [!! 修改] 返回 None
    except Exception as e:
        logger.critical(f"处理文件 '{input_txt_path}' 过程中发生未知错误: {e}", exc_info=True)
        return None # [!! 修改] 返回 None

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # --- 文件夹路径与阈值配置 ---
    # --- 请在这里修改为您自己的参数 ---
    # --- [!!] 路径现在可以指向文件夹 (批量处理) 或单个文件 (.txt / .png)
    # ------------------------------------------------------------------
    
    # 输入的标注路径 (可以是文件夹, 也可以是单个 .txt 文件)
    INPUT_TEXT_PATH = r"E:\Code\data\all\30samples\agent_input_text"
    
    # 对应的原始图像路径 (可以是文件夹, 也可以是单个 .png 文件)
    INPUT_IMAGE_PATH = r"E:\Code\data\all\30samples\agent_input_picture"
    
    # 处理结果的输出文件夹路径
    OUTPUT_DIR = r"E:\Code\data\all\30samples\output_gpt_4o_mini"

    # 【!!】置信度阈值：
    # 这是一个基础阈值。
    # 1. 当 PaddleOCR 置信度 < 此值时，触发 API (规则 2)
    # 2. 当 PaddleOCR 置信度 >= 此值时，将触发 否决(规则1) 或 触发(规则3) 的检查。
    #
    # 你之前设置的 1.01 会导致所有内容都被发送到 API，无法利用新规则。
    # 推荐设置为 0.95 左右的“高可信”门槛。
    CONFIDENCE_THRESHOLD = 0.95

    # ------------------------------------------------------------------
    # --- 配置结束 ---
    # ------------------------------------------------------------------

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # [!! 新增] 初始化全局统计变量
    files_processed = 0
    grand_total_prompt_tokens = 0
    grand_total_completion_tokens = 0
    grand_total_tokens = 0
    grand_total_api_calls = 0
    grand_total_sleep_time = 0.0
    grand_total_wall_time = 0.0
    
    # --- 核心逻辑：判断输入路径是文件夹还是文件 ---
    
    # 模式一：如果输入是文件夹，则执行批量处理
    if os.path.isdir(INPUT_TEXT_PATH):
        logger.info(f"检测到输入路径为文件夹，启动批量处理模式: '{INPUT_TEXT_PATH}'")
        
        # 批量模式下，图像路径也必须是文件夹
        if not os.path.isdir(INPUT_IMAGE_PATH):
            logger.error(f"输入文本是文件夹，但图像路径不是文件夹: '{INPUT_IMAGE_PATH}'")
            logger.error("批量处理模式失败。")
        else:
            # --- 开始批量处理逻辑 (与原脚本相同) ---
            try:
                text_files = [f for f in os.listdir(INPUT_TEXT_PATH) if f.endswith(".txt")]
                if not text_files:
                    logger.info(f"在目录 '{INPUT_TEXT_PATH}' 中没有找到 .txt 文件。")
                else:
                    logger.info(f"在 '{INPUT_TEXT_PATH}' 中发现 {len(text_files)} 个 .txt 文件，开始批量处理...")
                    
                    # 遍历输入文本目录中的所有 .txt 文件
                    for txt_filename in sorted(text_files): # sorted() 保证处理顺序
                        base_name = os.path.splitext(txt_filename)[0]
                        
                        # 构建完整的文件路径
                        input_txt_path = os.path.join(INPUT_TEXT_PATH, txt_filename)
                        image_path = os.path.join(INPUT_IMAGE_PATH, f"{base_name}.png")
                        output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
                        
                        # 检查对应的图像文件是否存在
                        if not os.path.exists(image_path):
                            logger.warning(f"找不到对应的图像文件 '{image_path}'，已跳过 '{input_txt_path}' 的处理。")
                            continue
                        
                        # [!! 修改] 调用主函数并接收统计数据
                        logger.info(f"--- 开始处理: {txt_filename} ---")
                        stats = run_vision_agent(input_txt_path, image_path, output_txt_path, CONFIDENCE_THRESHOLD)
                        
                        # [!! 新增] 累加全局统计
                        if stats:
                            files_processed += 1
                            grand_total_prompt_tokens += stats["prompt_tokens"]
                            grand_total_completion_tokens += stats["completion_tokens"]
                            grand_total_tokens += stats["total_tokens"]
                            grand_total_api_calls += stats["api_calls"]
                            grand_total_sleep_time += stats["sleep_time"]
                            grand_total_wall_time += stats["wall_time"]
                        
                        logger.info(f"--- 完成处理: {txt_filename} ---\n")
                    
                    logger.info("所有文件处理完毕！")

            except Exception as e:
                logger.critical(f"批量处理过程中发生严重错误: {e}", exc_info=True)
            # --- 批量处理逻辑结束 ---

    # 模式二：如果输入是文件，则执行单个文件处理
    elif os.path.isfile(INPUT_TEXT_PATH):
        logger.info(f"检测到输入路径为文件，启动单个文件处理模式: '{INPUT_TEXT_PATH}'")
        
        # 单文件模式下，图像路径也必须是文件
        if not os.path.isfile(INPUT_IMAGE_PATH):
            logger.error(f"输入文本是文件，但图像路径不是文件: '{INPUT_IMAGE_PATH}'")
            logger.error("单个文件处理模式失败。")
        else:
            # --- 开始单个文件处理逻辑 ---
            try:
                # 1. 从完整路径中获取文件名
                txt_filename = os.path.basename(INPUT_TEXT_PATH)
                # 2. 构建输出文件的完整路径
                output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
                
                logger.info(f"--- 开始处理单个文件: {txt_filename} ---")
                
                # [!! 修改] 调用主函数并接收统计数据
                stats = run_vision_agent(
                    input_txt_path=INPUT_TEXT_PATH,     # 使用配置的 .txt 文件路径
                    image_path=INPUT_IMAGE_PATH,      # 使用配置的 .png 文件路径
                    output_txt_path=output_txt_path, 
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )
                
                # [!! 新增] 累加全局统计
                if stats:
                    files_processed += 1
                    grand_total_prompt_tokens = stats["prompt_tokens"]
                    grand_total_completion_tokens = stats["completion_tokens"]
                    grand_total_tokens = stats["total_tokens"]
                    grand_total_api_calls = stats["api_calls"]
                    grand_total_sleep_time = stats["sleep_time"]
                    grand_total_wall_time = stats["wall_time"]
                
                logger.info(f"--- 完成处理: {txt_filename} ---")
                logger.info("单个文件处理完毕！")
                
            except Exception as e:
                logger.critical(f"处理文件 '{txt_filename}' 过程中发生严重错误: {e}", exc_info=True)
            # --- 单个文件处理逻辑结束 ---

    # 模式三：如果路径无效
    else:
        logger.error(f"输入路径既不是有效的文件也不是有效的目录: '{INPUT_TEXT_PATH}'")
        logger.error("程序已退出。")

    # --- [!! 新增] 最终全局统计报告 ---
    if files_processed > 0:
        grand_total_net_time = grand_total_wall_time - grand_total_sleep_time
        
        logger.info("==========================================================")
        logger.info(f"--- 任务总结 (共处理 {files_processed} 个文件) ---")
        logger.info("==========================================================")
        logger.info(f"  [调用] 总API调用次数: {grand_total_api_calls} 次")
        logger.info(f"  [Token] 总消耗: {grand_total_tokens}")
        logger.info(f"      - Prompt: {grand_total_prompt_tokens}")
        logger.info(f"      - Completion: {grand_total_completion_tokens}")
        logger.info(f"  [时间] 总墙上时间: {grand_total_wall_time:.2f} 秒")
        logger.info(f"  [时间] API等待时间 (已排除): {grand_total_sleep_time:.2f} 秒")
        logger.info(f"  [时间] 净处理时间: {grand_total_net_time:.2f} 秒")
        avg_net_time = grand_total_net_time / files_processed
        logger.info(f"  [性能] 平均净处理时间: {avg_net_time:.2f} 秒/文件")
        if grand_total_api_calls > 0:
            avg_token_per_call = grand_total_tokens / grand_total_api_calls
            logger.info(f"  [性能] 平均Token/调用: {avg_token_per_call:.1f} tokens")
        logger.info("==========================================================")
    elif not os.path.isdir(INPUT_TEXT_PATH) and not os.path.isfile(INPUT_TEXT_PATH):
        pass # 路径无效，之前已报错
    else:
        logger.info("--- 任务总结 ---")
        logger.info("未成功处理任何文件。")