# PCBTagent:A Knowledge-Driven Multi-Agent Framework for High-Precision Text Recognition in Complex PCB Schematics
This project is a correction tool that uses LLM to perform OCR on PCB schematics. It adopts a lightweight RAG strategy, which corrects the OCR results by providing domain knowledge and correction rules to the model.

## File Structure
```
 PCBTagent/
 │
 ├── resources/
 │   ├── knowledge_base_v1.json
 │   └── sampled_gts_unique_700_long_...
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
 │   └── prompting.py
 │
 ├── .env.example
 ├── .gitignore
 ├── LICENSE
 ├── README.md
```



