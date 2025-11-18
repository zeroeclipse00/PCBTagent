# PCBOCR: A Knowledge-Driven Multi-Agent Framework for High-Precision Text Recognition in Complex PCB Schematics
This project is a correction tool that uses LLM to perform OCR on PCB schematics. It adopts a lightweight RAG strategy, which corrects the OCR results by providing domain knowledge and correction rules to the model.

## File Structure
```
 PCBTagent/
 │
 ├── data/
 │ 
 ├── resources/
 │   ├── knowledge_base_v2.json
 │   └── sampled_gts_unique_700_long_300_short.txt
 │
 ├── src/
 │   ├── utils/
 │   │   ├── logging_setup.py
 │   │   └── parser.py
 │   │
 │   ├── config.py
 │   ├── llm_clients.py
 │   ├── main.py
 │   ├── pipeline.py
 │   ├── vision_agent.py
 │   └── prompting.py
 │
 ├── .env
 ├── .gitignore
 ├── LICENSE
 ├── README.md
 ├── run.bat
 └── run.sh
```

## Quick Start
Linux/macOS:
```
run.sh
```

Windows:
```
run.bat
```

**(Optional) Run the Vision Agent**:
    * Ensure the API key and model settings within `src/vision_agent.py` are correct.
    * Execute the script:  
    ```
    python src/vision_agent.py
    ```

## Complete dataset link
https://pan.baidu.com/s/1hNsExQIoFlTr-J23q2pfXw?pwd=qkx6