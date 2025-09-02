# PCBTagent:A Knowledge-Driven Multi-Agent Framework for High-Precision Text Recognition in Complex PCB Schematics
本项目是一个使用大语言模型（LLM）对 PCB 原理图进行光学字符识别（OCR）的纠正工具。它采用了一种轻量级的检索增强生成（RAG）策略，通过向模型提供领域知识和修正规则，以批处理的方式修正 OCR 结果。
## 数据格式
.txt文件，每行格式为class_id x y w h gt||ocr conf
## 命令行启动
python main.py --input <你的输入路径> --output <你的输出路径> [--provider {gpt,deepseek}] [--batch_size BATCH_SIZE] [--threshold THRESHOLD] [--include_gt] [-v] [--log_file LOG_FILE]

处理单个文件命令格式:  
python main.py --input data/sample_01.txt --output output/sample_01_corrected.txt [其他选项]

处理整个文件夹命令格式:  
python main.py --input data --output output [其他选项]

运行程序之前，配置 config.py 文件：  
选择 LLM 服务商:修改 PROVIDER 变量为 "deepseek" 或 "gpt"。  
设置 API 密钥:通过环境变量来设置 API 密钥。对于 DeepSeek, 设置 DEEPSEEK_API_KEY。对于 OpenAI/GPT, 设置 OPENAI_API_KEY。
### 必需参数
--input <你的输入路径>  
指定输入文件的路径或包含多个输入文件的文件夹路径。

--output <你的输出路径>  
指定处理结果的保存路径。如果输入是单个文件，这里应该是一个完整的文件路径（例如 output/result.txt）。
如果输入是一个文件夹，这里应该是一个文件夹路径（例如 output/），程序会在该文件夹下生成与输入文件同名的结果文件。
### 可选参数
[-h] 或 --help  
显示帮助信息。

[--provider {gpt,deepseek}]  
选择使用的大语言模型（LLM）服务商 API。可选值: gpt 或 deepseek。  
默认值: 如果不指定，会使用 config.py 文件中 PROVIDER 变量设定的值。

[--batch_size BATCH_SIZE]  
设置批处理大小。
决定一次性打包多少个待修正的词条发送给 LLM API。较大的批处理大小可以提高处理效率，但可能会增加单次请求的等待时间。  
默认值: 如果不指定，会使用 config.py 文件中 BATCH_SIZE 变量设定的值。

[--threshold THRESHOLD]  
设置置信度阈值。
OCR 识别结果的置信度低于这个设定的值时，程序会将其发送给 LLM 进行修正。  
默认值: 如果不指定，会使用 config.py 文件中 CONF_THRESHOLD 变量设定的值（默认为 1.01，意味着默认情况下会处理所有词条）。

[--include_gt]  
在提示词（Prompt）中包含真实标签（Ground Truth）。  
这是一个开关参数。如果使用它，程序会在发送给 LLM 的指令中附上每个词条的正确答案（GT），用于引导模型进行更精确的字符级修正。  
默认值: 默认不启用。主要用于调试或特定场景，默认关闭以避免“数据泄露”。

[-v]  
增加日志的详细程度（Verbose）。  
不使用 -v 或只使用一个 -v：显示标准的 INFO 级别信息。
使用 -vv：显示更详细的 DEBUG 级别信息，用于排查问题。  
默认值: 默认级别为 1 (INFO)。

[--log_file LOG_FILE]  
指定一个日志文件路径。  
除了在屏幕上显示日志外，还将所有日志信息额外保存到一个指定的文件中，方便后续回顾和分析。  
默认值: 默认不保存到文件。
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