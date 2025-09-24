# PCBTagent:A Knowledge-Driven Multi-Agent Framework for High-Precision Text Recognition in Complex PCB Schematics
本项目是一个使用大语言模型（LLM）对 PCB 原理图进行光学字符识别（OCR）的纠正工具。它采用了一种轻量级的检索增强生成（RAG）策略，通过向模型提供领域知识和修正规则，以批处理的方式修正 OCR 结果。
## 数据格式
.txt文件，每行格式为class_id x y w h gt||ocr conf

## Usage

### 1. Run with default paths
By default, input/output paths are defined in `config.py`:
```python
# config.py
INPUT_PATH = Path(PROJECT_ROOT / r"data/input/raw_data/labels")
OUTPUT_PATH = Path(PROJECT_ROOT / r"data/output")
```

So you can simply run:

```
python main.py
```

The program will read from `data/input/raw_data/labels` and write to `output`.

### 2. Run with user-specified paths

You can override the defaults using command-line arguments:

```
python main.py \
  --input /path/to/custom/input \
  --output /path/to/custom/output \
  --provider deepseek \
  --batch_size 50 \
  --threshold 1.01 \
  --include_gt False \
  --verbose 1
```

This way, `--input` and `--output` will take precedence over the default values from `config.py`.

## 文件结构
下面是项目主要文件的功能说明：  
main.py:程序主入口。  
负责解析命令行参数（如输入/输出路径、LLM 服务商等），并根据参数调用相应的处理流程。

### pipeline.py:核心处理管道。  
包含处理单个文件 (process_file) 和整个文件夹 (process_folder) 的核心逻辑。它会读取数据，将它们分批（batching），调用 LLM 进行修正，最后将修正结果写入输出文件。

### llm_clients.py:LLM API 客户端。  
封装了与不同 LLM 服务（如 GPT、DeepSeek）API 的交互逻辑。内置了指数退避（exponential backoff）的自动重试机制，以提高网络请求的稳定性。

### prompting.py:提示语构建模块。  
负责构建发送给 LLM 的详细指令（Prompt）。它会将 RAG 知识库、参考词表（few-shot examples）和待修正的数据项动态地组合成一个结构化的、高效的 Prompt。

### config.py:全局配置文件。  
集中管理项目的所有可配置项，包括：API 密钥和 URL。默认的 LLM 模型名称。批处理大小、置信度阈值等参数。用于 RAG 的核心知识库（RAG_KB）。

### utils/logging_setup.py:日志配置模块。  
用于初始化和配置项目的日志系统，支持不同级别的日志输出（INFO, DEBUG）和日志文件记录。

### utils/parser.py:解析工具模块。  
提供一系列辅助函数，用于：解析输入文件中特定格式的行 (parse_line)。将处理结果重新构建为输出格式 (rebuild_line)。对 LLM 返回的文本块进行后处理，提取出修正结果 (postprocess_llm_block)。